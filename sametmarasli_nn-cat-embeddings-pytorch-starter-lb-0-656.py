import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import gc
import time
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import os
import random
import  warnings
pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = [c for c,v in dtypes.items() if v in numerics]
categorical_columns = [c for c,v in dtypes.items() if v not in numerics]
nrows = 500000
retained_columns = numerical_columns + categorical_columns
train = pd.read_csv('../input/train.csv',
                    nrows = nrows,
                    usecols = retained_columns,
                    dtype = dtypes
                   )

retained_columns.remove('HasDetections')
test = pd.read_csv('../input/test.csv',
                   nrows=nrows,
                   usecols = retained_columns,
                   dtype = dtypes
                  )
target = train['HasDetections']
del train['HasDetections']

train_ids = train['MachineIdentifier']
test_ids = test['MachineIdentifier']

df_all = pd.concat((train,test),axis=0)

del train,test
gc.collect()
# In practice, among the numerical variables, many corresponds to identifiers. *In the current dataset, the truly numerical variables are in fact rare*. Below, I make a list of the variables which are truly numerical, according the the description of the data.

true_numerical_columns = [
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges'
]
binary_variables = [c for c in df_all.columns if df_all[c].nunique() == 2]

categorical_columns = [c for c in df_all.columns 
                       if (c not in true_numerical_columns) & (c not in binary_variables)]
# For variables with large cardinality, an efficient encoding consists in ranking the categories with respect to their frequencies. These variables are then treated as numerical.

def frequency_encoding(variable):
    # t = pd.concat([train[variable], test[variable]]).value_counts().reset_index()
    t = df_all[variable].value_counts().reset_index()
    t = t.reset_index()
    t.loc[t[variable] == 1, 'level_0'] = np.nan
    t.set_index('index', inplace=True)
    max_label = t['level_0'].max() + 1
    t.fillna(max_label, inplace=True)
    return t.to_dict()['level_0']

frequency_encoded_variables = [
    'Census_OEMModelIdentifier',
    'CityIdentifier',
    'Census_FirmwareVersionIdentifier',
    'AvSigVersion',
    'Census_ProcessorModelIdentifier',
    'Census_OEMNameIdentifier',
    'DefaultBrowsersIdentifier',
    'AVProductStatesIdentifier',
    'OsBuildLab',
]

for variable in tqdm(frequency_encoded_variables):
    freq_enc_dict = frequency_encoding(variable)
    df_all[variable] = df_all[variable].map(lambda x: freq_enc_dict.get(x, np.nan))
    categorical_columns.remove(variable)

for col in tqdm(categorical_columns):
    if str(df_all[col].dtypes)=='category':
        df_all[col] = df_all[col].cat.add_categories(['isna'])
    df_all.loc[df_all[col].isna(),col]  ='isna'

indexer = {}
for col in tqdm(categorical_columns+binary_variables+frequency_encoded_variables):
    if col == 'MachineIdentifier': continue
    _, indexer[col] = pd.factorize(df_all[col])
    df_all[col] = indexer[col].get_indexer(df_all[col])

del indexer
gc.collect()
embed_cols = []
len_embed_cols = []
for c in categorical_columns[1:]:
    if df_all[c].nunique()>2:
        embed_cols.append(c)
        len_embed_cols.append(df_all[c].nunique())
        print(c + ': %d values' % df_all[c].nunique()) #look at value counts to know the embedding dimensions
print('\n Number of embed features :', len(embed_cols))
# some feature engineering with high cardinalty categorical features
fe_frequency_encoded_variables = []
for var in frequency_encoded_variables:
    gr = df_all.groupby(var)[true_numerical_columns].agg(['mean','sum'])
    gr.columns = ['{}_{}_{}'.format(var,e[0],e[1]) for e in gr.columns]
    fe_frequency_encoded_variables.extend(gr.columns)
    df_all = df_all.merge(gr,how='left',on=var)
    df_all.head()
    del gr
    gc.collect()

df_all[fe_frequency_encoded_variables] = df_all[fe_frequency_encoded_variables].replace([np.inf, -np.inf], np.nan)

df_all = reduce_mem_usage(df_all)
# set index to unique identifier
df_all = df_all.set_index('MachineIdentifier')

# Select the numeric features
other_cols = [x for x in df_all.columns if x not in embed_cols]

# Impute missing values in order to scale
df_all[other_cols] = df_all[other_cols].fillna(value=0)


# Fit the scaler only on df_all data
scaler = MinMaxScaler().fit(df_all[other_cols])
df_all.loc[:, other_cols] = scaler.transform(df_all[other_cols])

# other_cols = [c for c in df_all.columns if (not c in embed_cols)]
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_layers = nn.ModuleList()
        self.dropout = nn.Dropout(.25)
        self.num_categorical = len(len_embed_cols)
        self.num_numeric = len(other_cols)
        
        for embed_col, len_embed_col in zip(embed_cols, len_embed_cols):
            self.emb_layers.append(nn.Embedding(len_embed_col, len_embed_col // 2))

        ff_inp_dim = sum(e.embedding_dim for e in self.emb_layers) + self.num_numeric
        self.ff = nn.Sequential(
            nn.Linear(ff_inp_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )


    def forward(self, x_batch):
        emb_indices = x_batch[:, :self.num_categorical].long()
        emb_outs = []
        for i, emb_layer in enumerate(self.emb_layers):
            emb_out = emb_layer(emb_indices[:, i])
            emb_out = self.dropout(emb_out)
            emb_outs.append(emb_out)
        
        embs = torch.cat(emb_outs, dim=1)

        x_numerical = x_batch[:, self.num_categorical:]
        embs_num = torch.cat([embs, x_numerical], dim=1)
        out = self.ff(embs_num)
        return out
train = df_all.loc[train_ids,embed_cols+other_cols]
test = df_all.loc[test_ids,embed_cols+other_cols]

del df_all
gc.collect()
X_train, X_val, y_train, y_val = train_test_split(train, target , test_size=0.33, random_state=42)

torch_X_train = torch.FloatTensor(X_train.values)
torch_X_val = torch.FloatTensor(X_val.values)
torch_y_train = torch.FloatTensor(y_train.values.astype(np.int32))
torch_y_val = torch.FloatTensor(y_val.values.astype(np.int32))
torch_test  = torch.FloatTensor(test.values)
# always call this before training for deterministic results
seed_everything()

batch_size = 512
n_epochs = 6

# init model
model = Net()
# init Binary Cross Entropy loss
loss_fn = torch.nn.BCELoss(reduction='mean')
# init optimizer
optimizer = Adam(model.parameters())

#prepare iterators for training
torch_train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
train_loader = torch.utils.data.DataLoader(torch_train, batch_size=batch_size, shuffle=True)
torch_val = torch.utils.data.TensorDataset(torch_X_val, torch_y_val)
valid_loader = torch.utils.data.DataLoader(torch_val, batch_size=batch_size, shuffle=False)

# init predictions
train_preds = np.zeros((torch_X_train.size(0)))
valid_preds = np.zeros((torch_X_val.size(0)))
for epoch in range(n_epochs): 
    start_time = time.time()
    avg_loss = 0.  
    # set the module in training mode.
    model.train()

    for x_batch, y_batch in tqdm(train_loader, disable=True):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x_batch)
        # Compute and print loss.
        loss = loss_fn(y_pred, y_batch)
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    # set evaluation mode of the model. This disabled operations which are only applied during training like dropout
    model.eval()

    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        # detach returns a new Tensor, detached from the current graph whose result will never require gradient
        y_val_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_val_pred, y_batch).item() / len(valid_loader)

        valid_preds[i * batch_size:(i+1) * batch_size] = y_val_pred.cpu().numpy()[:, 0]
    elapsed_time = time.time() - start_time 
    print('\nEpoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
    print('AUC_VAL{} '.format(roc_auc_score(torch_y_val.cpu(),valid_preds).round(3)))
torch_test = torch.utils.data.TensorDataset(torch_test)
test_loader = torch.utils.data.DataLoader(torch_test, batch_size=batch_size, shuffle=False)
test_preds = np.zeros((len(torch_test)))


for i, (x_batch,) in enumerate(test_loader):
    y_pred = model(x_batch).detach()
    test_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]
fpr, tpr, _ = roc_curve(torch_y_val.cpu(),valid_preds)

roc_auc = auc(fpr,tpr)

plt.figure(figsize=(10,6))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
submission = pd.DataFrame({'MachineIdentifier':test_ids,'HasDetections':test_preds})
submission.head()
# submission.to_csv('nn_embeddings.csv.gz', index=False, ,compression='gzip')