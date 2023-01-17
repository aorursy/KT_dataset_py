import numpy as np
import pandas as pd

import gc
import os
from collections import Counter
from datetime import datetime as dt
import time
from numba import jit

# Plots
import matplotlib.pyplot as plt
import seaborn as sns 

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# PyTorch
import torch
from torch import autograd, nn, optim
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# TSFresh
from tsfresh.feature_extraction import extract_features

# Path to data directory
data_dir = '../input'

train = pd.read_csv(data_dir + '/training_set.csv')
train.head()

@jit
def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) from 
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
                          np.multiply(np.cos(lat1), 
                                      np.multiply(np.cos(lat2), 
                                                  np.power(np.sin(np.divide(dlon, 2)), 2))))
    
    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine, 
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)), 
   }

# TSFresh features
fcp = {
    'flux': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None
    },

    'flux_by_flux_ratio_sq': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,      
    },

    'flux_passband': {
        'kurtosis' : None,
        'skewness' : None,
        'standard_deviation': None
    },
    'fluxerr_passband': {
        'kurtosis' : None,
        'skewness' : None,
        'standard_deviation': None
    },
    'mjd': {
        'maximum': None, 
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None,
    },
}
def process_meta(filename):
    meta_df = pd.read_csv(filename)
    
    meta_dict = dict()
    
    # Distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values, 
                   meta_df['gal_l'].values, meta_df['gal_b'].values))
    
    meta_dict['hostgal_photoz_certain'] = np.multiply(
            meta_df['hostgal_photoz'].values, 
             np.exp(meta_df['hostgal_photoz_err'].values))
    
    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df

def featurize(df, df_meta, fcp, n_jobs=4):
    """
    Extracting Features from train or test set
    Features from olivier's kernel
    very smart and powerful feature that is generously given here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    per passband features with tsfresh library. fft features added to capture periodicity https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """
    
    # Add more features with
    agg_df_ts_flux_passband = extract_features(df, 
                                               column_id='object_id', 
                                               column_sort='mjd', 
                                               column_kind='passband', 
                                               column_value='flux', 
                                               default_fc_parameters=fcp['flux_passband'], n_jobs=n_jobs)

    agg_df_ts_fluxerr_passband = extract_features(df, 
                                               column_id='object_id', 
                                               column_sort='mjd', 
                                               column_kind='passband', 
                                               column_value='flux_err', 
                                               default_fc_parameters=fcp['flux_passband'], n_jobs=n_jobs)

    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected']==1].copy()
    agg_df_mjd = extract_features(df_det, 
                                  column_id='object_id', 
                                  column_value='mjd',
                                  default_fc_parameters=fcp['mjd'], n_jobs=n_jobs)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']
    
    # Rename the index for merging with metadata (tsfresh renames the index)
    agg_df_ts_flux_passband.index.rename('object_id', inplace=True) 
    agg_df_ts_fluxerr_passband.index.rename('object_id', inplace=True)
    agg_df_mjd.index.rename('object_id', inplace=True)

      
    agg_df_ts = pd.concat([
                           agg_df_ts_flux_passband, 
                           agg_df_ts_fluxerr_passband,
                            agg_df_mjd
                          ], axis=1).reset_index()

    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    return result
meta_train = process_meta(data_dir + '/training_set_metadata.csv')
full_train = featurize(train, meta_train, fcp, n_jobs=6)

if 'target' in full_train:
    y = full_train['target']
    del full_train['target']
full_train.head()
if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'], full_train['hostgal_specz'],  full_train['hostgal_photoz']
    del full_train['ra'], full_train['decl'], full_train['gal_l'],full_train['gal_b'],full_train['ddf']
    
train_mean = full_train.mean(axis=0)
pd.set_option('display.max_rows', 500)

full_train.fillna(train_mean, inplace=True)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
ss = StandardScaler()

# @todo figure out something to replace this
full_train_values = np.nan_to_num(full_train.values)
full_train_ss = ss.fit_transform(full_train_values)
del full_train_values
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# Get unique target classes
unique_y = np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i

y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
y_categorical = to_categorical(y_map, len(unique_y))
#device = 'cuda'
device = 'cpu'
class PlasticcNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the class and call the parent init method.
        Create two hidden layers. Derived classes from nn.Module
        will assign learnable params to self.parameters
        """
        super(PlasticcNet, self).__init__()
        
        dropout_proba = 0.6
        bias = False
        
        
        # define hidden linear layers, with optional batch norm on their outputs
        # layers with batch_norm applied have no bias term
        self.dense1 = nn.Linear(input_size, hidden_size, bias=bias)
        
        # It seems that even Christian Szegedy now likes to perform BatchNorm after the ReLU (not before it). 
        # Quote by F. Chollet, the author of Keras: "I haven't gone back to check what they are suggesting 
        # in their original paper, but I can guarantee that recent code written by Christian 
        # applies relu before BN. It is still occasionally a topic of debate, though."
        self.batch1 = nn.BatchNorm1d(hidden_size)
        
        self.dense2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.batch2 = nn.BatchNorm1d(hidden_size)
        
        self.dense3 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.batch3 = nn.BatchNorm1d(hidden_size)
        
        self.dense4 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.batch4 = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(p=dropout_proba)
        
        self.dense5 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Feed data forward through the layers.
        """
        x = self.dense1(x)
        x = self.batch1(x)
        x = self.dropout(torch.relu(x))
        
        x = self.dense2(x)
        x = self.batch2(x)
        x = self.dropout(torch.relu(x))
        
        x = self.dense3(x)
        x = self.batch3(x)
        x = self.dropout(torch.relu(x))
        
        x = self.dense4(x)
        x = self.batch4(x)
        x = self.dropout(torch.relu(x))
        
        x = self.dense5(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class PlasticcDataset(Dataset):
    def __init__(self, x, y_categorical, mode='train'):
        self.x = x
        self.y_categorical = y_categorical
        self.mode = mode
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x, y = self.x[index], self.y_categorical[index]
        if self.mode == 'validate':
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
            y = torch.tensor(y, dtype=torch.int64).to(device)
            return x, y
        else:
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.int64).to(device)
            return x, y
num_classes = len(unique_y)
classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1,
                64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
weight_tensor = torch.tensor(list(class_weight.values()),
                             requires_grad=False, dtype=torch.float32).to(device)
def multi_weighted_logloss(y_ohe, y_p, class_weight):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    #class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    
    # Transform to log
    y_p_log = np.log(y_p)
    
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss
def criterion(ln_p, y_h, wt):
    """
    Modified implementation of wloss_objective competiton weighted log loss
    from @mithrillion Know Your Objective kernel https://www.kaggle.com/mithrillion/know-your-objective
    Main changes:
        - Moved softmax back to net and take log probs as first arg
        - Second arg is one hot matrix of target classes so we don't need to call scatter to expand the labels
        back into a one hot matrix
        - Divide everything by the sum of the weights as is shown in competition evaluation equation. Olivier's
        function also does this
        - Removed gradient computations as PyTorch loss function does not need them
    """
    y_h = y_h.float()
    ys = y_h.sum(dim=0, keepdim=True)
    
    # Replace 0's with 1's so we don't get nans from division by 0
    ys[ys==0] = 1
    y_h = y_h / ys
    wll = torch.sum(y_h * ln_p, dim=0)
    wsum = torch.sum(wt, dim=0)
    loss = -(torch.dot(wt, wll) / wsum)
    return loss

def batch_accuracy(probs, labels):
    """
    Batch accuracy from https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch
    part 5
    """
    top_p_validate, top_class_validate = probs.topk(1, dim=1)
    equals_validate = top_class_validate == labels.view(*top_class_validate.shape)
    return torch.mean(equals_validate.type(torch.FloatTensor))

def check_probs(probs):
    """
    Check that predictions sum to 1
    """
    preds_batch_sum = torch.sum(torch.sum(probs, dim=1), dim=0).item()
    sum_diff = (np.absolute(batch_size - preds_batch_sum))
    if (sum_diff > 0.1) and (preds_batch_sum != probs.shape[0]):

        # Should be the same size as the number of samples
        print("preds_sum: ", preds_batch_sum)

def fold_validation(model, xv, yv):
    """
    Validate on all samples in current fold validate split
    """
    x_valid_mwll = torch.tensor(xv, dtype=torch.float32, requires_grad=False).to(device)
    y_valid_mwll = torch.tensor(yv, dtype=torch.int64).to(device)
    olivier_loss = multi_weighted_logloss(yv, torch.exp(model(x_valid_mwll)).cpu().numpy(), class_weight)
    pytorch_loss = criterion(model(x_valid_mwll), y_valid_mwll, weight_tensor).cpu().item()
    print('Olivier loss: {:.6f}'.format(olivier_loss))
    print('PyTorch criterion loss: {:.6f}'.format(pytorch_loss))
    return pytorch_loss
hidden_size = 256
batch_size = 100
epochs = 100

best_model = {
    'epoch': 0,
    'loss': np.Inf,
    'fold': 0
}

for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
    x_train, y_train = full_train_ss[trn_], y_categorical[trn_]
    x_valid, y_valid = full_train_ss[val_], y_categorical[val_]
    
    # Create datasets (Make sure trn_ and val_ indices are correct)
    ds_train = PlasticcDataset(full_train_ss[trn_], y_categorical[trn_], mode='train')
    ds_val = PlasticcDataset(full_train_ss[val_], y_categorical[val_], mode='validate')
    
    # Create dataloaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    validate_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    
    # Create a new model instance
    model = PlasticcNet(full_train_ss.shape[1], hidden_size, num_classes)
    
    # Run model on gpu
    model.to(device)
    
    # Create optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    validate_losses = []
    validate_accuracy = []
    
    # Initialize tracker for minimum validation loss
    # Set initial min to infinity because any loss we get will be less than infinity
    validate_loss_min = np.Inf
    
    # Epochs loop
    for e in range(epochs):
        running_train_loss = 0
        running_validate_accuracy = 0
        running_validate_loss = 0
        all_validate_probs = []
        
        # Switch to train mode
        model.train()
        for train_bx, train_by in train_loader:
            log_probs = model(train_bx)
            loss = criterion(log_probs, train_by, weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            
        ##### Validation (After each epoch) #####
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            
            # Switch to eval mode
            model.eval()
            
            for validate_bx, validate_by in validate_loader:
                validate_log_probs = model(validate_bx)
                validate_probs = torch.exp(validate_log_probs)
                    
                # Check the sum of the predicted class probabilities
                check_probs(validate_probs)
                    
                loss = criterion(validate_log_probs, validate_by, weight_tensor)
                
                # Get target class index from one hot vector
                validate_labels = torch.argmax(validate_by, 1)
                
                # Sum batch loss and accuracy
                running_validate_loss += loss.item()
                running_validate_accuracy += batch_accuracy(validate_probs, validate_labels)
        
            # Validate on all samples in current fold validate split
            pytorch_loss = fold_validation(model, x_valid, y_valid)
            
        vln = len(validate_loader)
        tln = len(train_loader)
        train_loss = (running_train_loss / tln)
        validate_loss = (running_validate_loss / vln)
        validate_accuracy = (running_validate_accuracy / vln)
        
        train_losses.append(train_loss)
        validate_losses.append(validate_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Train Avg Batch Loss: {:.4f}.. ".format(train_loss),
              "Val Avg Batch Loss: {:.4f}.. ".format(validate_loss),
              "Val Avg Batch Accuracy: {:.4f}".format(validate_accuracy))
        
        # save model if validation loss has decreased
        if pytorch_loss <= validate_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            validate_loss_min,
            validate_loss))
            torch.save(model.state_dict(), "pt_fold{}_model.pt".format(fold_))
            validate_loss_min = pytorch_loss
            
        # Record best model across all folds
        if pytorch_loss <= best_model['loss']:
            best_model['loss'] = pytorch_loss
            best_model['epoch'] = e
            best_model['fold'] = fold_
            print("best_model updated: ", best_model)
            
    plt.plot(train_losses, label='Training loss')
    plt.plot(validate_losses, label='Validation loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()
# Get the column names from the sample submission
sample_sub = pd.read_csv(data_dir + '/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
del sample_sub;gc.collect()
# meta_test is 285M so we can load that into memory in one go
meta_test = pd.read_csv(data_dir + '/test_set_metadata.csv')
def predict_chunk(df_, model_eval_, ss_, meta_, features, featurize_configs, train_mean):
    
    # Process all features
    full_test = featurize(df_, meta_, featurize_configs['fcp'])
    full_test.fillna(train_mean, inplace=True)
    
    partial_test = full_test.copy()
    
    # Delete unused features
    if 'object_id' in full_test:
        del full_test['object_id'], full_test['hostgal_specz'],  full_test['hostgal_photoz']
        del full_test['ra'], full_test['decl'], full_test['gal_l'],full_test['gal_b'],full_test['ddf']

    test_chunk_values = np.nan_to_num(full_test.values)
    
    # Only do transform on test set, scaler paramters have been learned
    # on the train set
    test_chunk_ss = ss.transform(test_chunk_values)
    x_test = torch.tensor(test_chunk_ss, dtype=torch.float32).to('cuda')

    test_log_probs = model_eval(x_test)
    test_probs = torch.exp(test_log_probs)
    preds_ = test_probs.cpu().numpy()

    check_probs(test_probs)
    
    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    # Create an empty vector the size of the number of samples
    preds_99 = np.ones(preds_.shape[0])
    
    # Loop over the number of classes (should be 14)
    for i in range(preds_.shape[1]):
        preds_99 *= (1 - preds_[:, i])
        
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]

    # Create DataFrame from predictions
    preds_df_ = pd.DataFrame(preds_, 
                             columns=['class_{}'.format(s) for s in classes])
    
    # Add the object ids back (this is why we select instead of delete cols above)
    preds_df_['object_id'] = partial_test['object_id']
    
    # Add probability for class 99
    mean_preds_99 = 1
    if np.mean(preds_99) > 0:
        mean_preds_99 = np.mean(preds_99)
        
    preds_df_['class_99'] = 0.14 * preds_99 / mean_preds_99
    return preds_df_


def process_test(model_eval_,
                 ss_,
                 features, 
                 featurize_configs,
                 train_mean,
                 filename='predictions.csv',
                 chunks=5000000):
    start = time.time()

    meta_test = process_meta(data_dir + '/test_set_metadata.csv')
    
    remain_df = None
    for i_c, df in enumerate(pd.read_csv(data_dir + '/test_set.csv', chunksize=chunks, iterator=True)):
        
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df['object_id'])
        
        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df
        
        preds_df = predict_chunk(df_=df,
                                 model_eval_=model_eval_,
                                 ss_=ss,
                                 meta_=meta_test,
                                 features=features,
                                 featurize_configs=featurize_configs,
                                 train_mean=train_mean)
    
        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)
    
        del preds_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes' .format(
                chunks * (i_c + 1), (time.time() - start) / 60), flush=True)
        
    # Compute last object in remain_df
    preds_df = predict_chunk(df_=remain_df,
                             model_eval_=model_eval_,
                             ss_=ss,
                             meta_=meta_test,
                             features=features,
                             featurize_configs=featurize_configs,
                             train_mean=train_mean)
        
    preds_df.to_csv(filename, header=False, mode='a', index=False)
    return
# with torch.no_grad():
    
#     model_eval = PlasticcNet(full_train_ss.shape[1], hidden_size, num_classes)
#     mpt = 'pt_fold{}_model.pt'.format(best_model['fold'])
#     print('Loading best model state: {}...'.format(mpt))
#     model_eval.load_state_dict(torch.load(mpt))
#     model_eval.eval().to(device)

#     filename = 'pt_wll_subm_{:.6f}_{}.csv'.format(best_model['loss'], dt.now().strftime('%Y-%m-%d-%H-%M'))
#     print('save to {}'.format(filename))
#     process_test(model_eval_=model_eval,
#                  ss_=ss,
#                  features=full_train.columns, 
#                  featurize_configs={'fcp': fcp}, 
#                  train_mean=train_mean, 
#                  filename=filename,
#                  chunks=5000000)

#     z = pd.read_csv(filename)
#     print("Shape BEFORE grouping: {}".format(z.shape))
#     z = z.groupby('object_id').mean()
#     print("Shape AFTER grouping: {}".format(z.shape))
#     z.to_csv('single_{}'.format(filename), index=True)
