import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import torch

import torch.nn as nn

from torch import tensor

import torch.optim as torch_optim

import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split
# Read in data

train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

dep_var = 'FVC'



# Drop the dependent variable

# CURRENTLY DROPPING PERCENT. NEED TO LOOK INTO IF WE NEED TO KEEP OR NOT

X_df = train.drop(labels=[dep_var, 'Percent'], axis=1)

y_df = train[dep_var]



def encode_objects(df):

    obj_cols = []

    le = LabelEncoder()

    for col, dtype in zip(df, df.dtypes):

        if dtype == 'object':

            df[col] = le.fit_transform(df[col])

            df[col] = df[col].astype('category')

            obj_cols.append(col)

    

    print("Converted {0} columns from objects to categories".format(obj_cols))



# Encode 'object' dtype columns

encode_objects(X_df)
X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.10, shuffle=True)

X_train.head()
def create_embeddings(df):

    embedded_cols = {}

    for n,col in X_df.items():

        if str(X_df[n].dtype) == 'category':

            embedded_cols[n] = len(col.cat.categories)

            

    # Create dict of emb names

    embedded_col_names = embedded_cols.keys()



    embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]

    

    return embedded_col_names, embedding_sizes

    
embedded_col_names, embedding_sizes = create_embeddings(X_df)
## Create Dataset



class OSICTabularDataset(Dataset):

    """

    Args:

        csv_file : str 

            Path to the csv file with annotations.

        y_name : str

            Name of dependent variable

    """

    def __init__(self, X, Y=None, embedded_col_names=None):

        # Break it into the numberical and categoricals for the model later

        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64) #categorical columns

        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32) #numerical columns

        if Y is not None:

            self.y = Y.values.astype(np.float32)

        else:

            self.y = None

                        

    def __len__(self):

        if self.y is None:

            return len(self.X1)

        else:

            return len(self.y)

    

    def __getitem__(self, idx):

        if self.y is None:

            return self.X1[idx], self.X2[idx]

        else:

            return self.X1[idx], self.X2[idx], self.y[idx]
train_ds = OSICTabularDataset(X_train, y_train, embedded_col_names)
valid_ds = OSICTabularDataset(X_val, y_val, embedded_col_names)
# Sanity check loop through train_dataset

for i in range(10):

    sample = train_ds[i] 

    print(sample)
# Sanity check loop through train_dataset

for i in range(10):

    sample = valid_ds[i] 

    print(sample)
z_cat = torch.randn(50, 53)

z_cont = torch.randn(50, 3)

print(z_cont.size(), z_cat.size())



comb = torch.cat((z_cat, z_cont), 1)

print(comb.size())
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')
def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)
class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)
device = get_default_device()

device
# Network Architecture



class OSICModel(nn.Module):

    def __init__(self, embedding_sizes, n_cont):

        super().__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])

        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined

        self.n_emb, self.n_cont = n_emb, n_cont

        self.lin1 = nn.Linear(55, 100)

        self.lin2 = nn.Linear(100, 100)

        self.lin3 = nn.Linear(100, 1)

#         self.bn1 = nn.BatchNorm1d(self.n_cont)

#         self.bn2 = nn.BatchNorm1d(200)

#         self.bn3 = nn.BatchNorm1d(70)

#         self.emb_drop = nn.Dropout(0.6)

#         self.drops = nn.Dropout(0.3)



        self.sigma = nn.Linear(100, 1)

        

    def forward(self, x_cat, x_cont):

        out = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]

        # Concat into one tensor

        out = torch.cat(out, 1)

        out = torch.cat((x_cont, out), 1)

        

        out = F.relu(self.lin1(out))

        out = F.relu(self.lin2(out))

        fvc = F.relu(self.lin3(out))

        sigma = F.relu(self.sigma(out))

        return fvc,sigma

        

    def metric_loss(self,pred_fvc,true_fvc,pred_sigma):

        true_fvc=torch.reshape(true_fvc,pred_fvc.shape)

        sigma_clipped=torch.clamp(pred_sigma,min=1e-3)

        delta=torch.clamp(torch.abs(pred_fvc-true_fvc),max=1000)

        metric=torch.div(-torch.sqrt(torch.tensor([2.0]).to(device))*delta,sigma_clipped)-torch.log(torch.sqrt(torch.tensor([2.0]).to(device))*sigma_clipped)

        return -metric

    

    def fvc_loss(self,pred_fvc,true_fvc):

        true_fvc=torch.reshape(true_fvc,pred_fvc.shape)

        fvc_err=torch.abs(pred_fvc-true_fvc)

        return fvc_err
model = OSICModel(embedding_sizes, 3)



to_device(model, device)
# I have a dataloader waiting in the wings

dataloader_valid = DataLoader(train_ds, batch_size=6, shuffle=True, num_workers=4)



for i_batch, (cats, nums, dep_var) in enumerate(dataloader_valid):

    print("Batch:", i_batch)

    print("Categoricals:\n {0}".format(cats))

    print("Numericals:\n {0}".format(nums))

    print("Ground Truth:\n {0}".format(dep_var))

    print("-------------------\n")

    
# Optimizer (ADAM)

def get_optimizer(model, lr = 0.01, wd = 0.0):

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)

    return optim
def train_model(model, optim, data):

    model.train()

    total = 0

    train_loss=0

    train_metric=0

    for x1, x2, y in data:

        batch = y.shape[0]

        fvc, sigma = model(x1, x2)

        

        fvc_loss = model.fvc_loss(fvc, y.to(device)).mean()

        metric_loss = model.metric_loss(fvc, y.to(device),sigma).mean()

        loss = metric_loss

        optim.zero_grad()

        loss.backward()

        

        train_loss += fvc_loss.item()

        train_metric += metric_loss.item()

        optim.step()

        total += batch

        

    return train_loss, train_metric
def val_result(model, valid_dl):

    model.eval()

    val_loss=0

    val_metric=0

    for x1, x2, y in valid_dl:

        # FVC and Sigma are the "predictions"

        fvc, sigma = model(x1, x2)

        fvc_loss = model.fvc_loss(fvc, y.to(device)).mean()

        metric_loss = model.metric_loss(fvc, y.to(device),sigma).mean()

        

        loss = metric_loss

        val_loss += fvc_loss.item()

        val_metric += metric_loss.item()

    return val_loss, val_metric
epoch_train_metric=[]

epoch_val_metric=[]

epoch_train_loss=[]

epoch_val_loss=[]



def train_loop(model, train_data, valid_data=None, epochs=50, lr=0.01, wd=0.0):

    optim = get_optimizer(model, lr = lr, wd = wd)

    for epoch in range(epochs): 

        train_loss, train_metric = train_model(model, optim, train_data)

        print('\n====> Epoch: {}'.format(epoch))

        print('-------------------------------')

        

        print('Average TRAIN fvc loss: {:.4f}'.format(

              train_loss / len(train_dl)))

        print('Average TRAIN metric: {:.4f}'.format(

              train_metric / len(train_dl)))

        

        val_loss, val_metric = val_result(model, valid_dl)

        print('Average VALIDATION fvc loss: {:.4f}'.format(

              val_loss / len(valid_dl)))

        print('Average VALIDATION metric: {:.4f}'.format(

              val_metric / len(valid_dl)))

        

        epoch_train_loss.append(train_loss/ len(train_dl))

        epoch_val_loss.append(val_loss / len(valid_dl))

        epoch_train_metric.append(train_metric/ len(train_dl))

        epoch_val_metric.append(val_metric / len(valid_dl))

        

    print("Min TRAIN metric:", min(epoch_train_metric))

    print("Min VALID metric:", min(epoch_val_metric))
batch_size = 50

train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)

valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)
train_dl = DeviceDataLoader(train_dl, device)

valid_dl = DeviceDataLoader(valid_dl, device)
train_loop(model, train_dl, valid_dl, epochs=50, lr=0.05, wd=0.00001)
# Plot results

def plot_training_loss(train, val,title='loss'):

    plt.figure()

    plt.plot(train, label='Train')

    plt.plot(val, label='Val')

    if title=='loss':

        plt.title('Model Training Loss')

    else:

        plt.title('Model Metric Loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.yscale('log')

    plt.legend()

    plt.savefig('training_loss')
# plot_training_loss(epoch_train_loss, epoch_val_loss)
# plot_training_loss(epoch_train_metric, epoch_val_metric, title='metric')
# Dataframe Cleanup



test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test = test.rename(columns={'Weeks': 'base_Weeks', 'FVC': 'base_FVC','Percent': 'base_Percent'})



# Adding Sample Submission

submission = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")



# In submisison file, format: ID_'week', using lambda to split the ID

submission['Patient'] = submission['Patient_Week'].apply(lambda x:x.split('_')[0])



# In submisison file, format: ID_'week', using lambda to split the Week

submission['Weeks'] = submission['Patient_Week'].apply(lambda x:x.split('_')[1]).astype(int)



test = submission.drop(columns = ["FVC", "Confidence"]).merge(test, on = 'Patient')



test['Week_passed'] = test['Weeks'] - test['base_Weeks']



test=test[train.columns.drop(['FVC','Percent'])]
test.head()
# Cleanup dataframe

encode_objects(test)

embedded_test_col_names, embedding_test_sizes = create_embeddings(test)
test.head()
# Create dataset

test_ds = OSICTabularDataset(test, embedded_col_names=embedded_test_col_names)
test_ds[0]
# Sanity check loop through test_dataset

for i in range(5):

    sample = test_ds[i] 

    print(sample)
test_dl = DataLoader(test_ds, batch_size=batch_size,shuffle=True)

test_dl = DeviceDataLoader(test_dl, device)
def test_eval(model, test_data):

    model.eval()

    fvc_pred = []

    sigma_pred = []

    with torch.no_grad():

        for x1, x2 in test_dl:

            # FVC and Sigma are the "predictions"

            fvc, sigma = model(x1, x2)

            fvc_pred.append(fvc)

            sigma_pred.append(sigma)

    fvc_pred=torch.cat(fvc_pred, dim=0)

    sigma_pred=torch.cat(sigma_pred, dim=0)

    

    return fvc_pred, sigma_pred
fvc_pred, sigma_pred = test_eval(model, test_dl)
try:

    test['FVC']=fvc_pred.cpu().numpy()

    test['Confidence']=sigma_pred.cpu().numpy()



    test['Patient_Week']=test["Patient"]



    final_submission = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

    test['FVC']



    kaggle_submission = pd.concat([final_submission['Patient_Week'], test['FVC'], test['Confidence']], axis=1)

    kaggle_submission.to_csv('./submission.csv', index=False, float_format='%.1f')

    

except: 

    raise RuntimeError
kaggle_submission = pd.concat([final_submission['Patient_Week'], test['FVC'], test['Confidence']], axis=1)
kaggle_submission.to_csv('./submission.csv', index=False, float_format='%.1f')
import os



for dirname, _, filenames in os.walk('/kaggle/working/'):

   for filename in filenames:

       print(os.path.join(dirname, filename))