
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
train=pd.read_csv('../input/into-the-future/train.csv')
train.head(10)
test=pd.read_csv('../input/into-the-future/test.csv')
test.head(10)
train.isnull().sum()
test.isnull().sum()
test.describe()
train['time']=pd.to_datetime(train['time'])
train.info()

test['time']=pd.to_datetime(test['time'])
test.info()
train.shape
test.shape
test.head()
train.head()
train.plot(x='feature_1', y='feature_2')

all_cols = []
for col in train.columns:
    all_cols.append(col)
input_cols = all_cols[1:-1]
input_cols

output_cols = [all_cols[-1]]
output_cols
def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in input_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array
inputs_array, targets_array = dataframe_to_arrays(train)
inputs_array, targets_array
inputs = torch.from_numpy(inputs_array).type(torch.float32)

targets = torch.from_numpy(targets_array).type(torch.float32)
inputs.size()
targets.size()
from torch.utils.data import DataLoader, TensorDataset, random_split
dataset = TensorDataset(inputs, targets)
num_rows = len(train)
num_rows
val_percent = 0.1 
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset, [train_size, val_size]) # Use the random_split function to split dataset into 2 parts of the desired length

len(train_ds), len(val_ds)
batch_size = 60
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
for xb, yb in train_loader:
    print('inputs',xb)
    print('targets',yb)
    break
input_cols
input_size = len(input_cols)
input_size
output_size = len(output_cols)
output_size
import torch.nn.functional as F
class time_series_analysis(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 282)              
        self.linear2 = nn.Linear(282,141)
        self.linear3 = nn.Linear(141,64)
        self.linear4 = nn.Linear(64,32)
        self.linear5 = nn.Linear(32,output_size)
        
    def forward(self, xb):
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.linear1(out)
        out = F.relu(out)
        
        out = self.linear2(out)
        out = F.relu(out)
        
        out = self.linear3(out)
        out = F.relu(out)
        
        out = self.linear4(out)
        out = F.relu(out)
        
        out = self.linear5(out)
        out = F.relu(out)
        
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out, targets)                         
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out, targets)                          
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
model = time_series_analysis()
model
list(model.parameters())
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history
result = evaluate(model, val_loader) 
print(result)
epochs = 1000
lr = 1e-1
history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 100
lr = 0.001
history2 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 100
lr = 0.01
history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 150
lr = 1e-5
history4 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 150
lr = 1e-10
history5 = fit(epochs, lr, model, train_loader, val_loader)
loss = [result] + history1 + history2 + history3 + history5

losses = [result['val_loss'] for result in loss]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
jovian.log_metrics(val_loss=val_loss)
!pip install jovian --upgrade --quiet

project_name = 'time-series-analysis'
import jovian
jovian.commit(project=project_name, enviroment=None)
def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)                # fill this
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)
    

inputt, target = val_ds[0]
predict_single(inputt, target, model)
all_cols_t = []
for col in test.columns:
    all_cols_t.append(col)
    
all_cols_t
input_cols_test = all_cols_t[1:]
input_cols_test
def dataframe_to_arrays_test(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in dataframe.columns:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols_test].to_numpy()
    
    return inputs_array
input_array = dataframe_to_arrays_test(test)
input_array
inputs = torch.from_numpy(input_array).type(torch.float32)
test_ds = TensorDataset(inputs)
test_loader = DataLoader(test_ds, batch_size)
def predict_dl(dl, model):
    batch_probs = []
    for xb in dl:
        inputs = xb.unsqueeze(0)
        targets = model(inputs)
        batch_probs.append(targets.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return batch_probs
test_preds = predict_dl(inputs, model)
test_preds
len(test_preds)
len(test)
px = test_preds.numpy()
test_preds = pd.DataFrame(px)
submission_df = pd.read_csv('../input/into-the-future/test.csv')
print(submission_df.head(3))
submission_df['feature_2'] = test_preds
submission_df.head()
sub_fname = 'submission_1.csv'
submission_df.to_csv(sub_fname, index=False)

import seaborn as sns
sns.boxplot(x='feature_1',data=train)
# There are outliers in the feature 1 of the train data
sns.boxplot(x='feature_2',data=train)
# There are outliers in the feature_2 in the train data
def predict_dl(dl, model):
    batch_probs = []
    for xb in dl:
        probs = model(xb)
    return probs 
    
   
test_preds = predict_dl(test_loader, model)
sns.boxplot(x='feature_1',data=test)
# There are no outlier in the test data set
sns.jointplot(x='feature_1',y='feature_2',data=train)
# Accorfing ti the dependencie we ca see that there is one outlier so it is not oin to affect our data
# treating thte outlier
np.percentile(train.feature_1,[99])
np.percentile(train.feature_1,[99])[0]
uv=np.percentile(train.feature_1,[99])[0]
train[(train.feature_1)>uv]
train.feature_1[(train.feature_1)>3*uv]=3*uv
train[(train.feature_1)>uv]
# Correlation 
train.corr()
del train['id']
train.head()
import statsmodels.api as sm
X=sm.add_constant(train['feature_1'])
X.head()
lm=sm.OLS(train['feature_2'],X).fit()
lm.summary()
from sklearn.linear_model import LinearRegression
X=train['feature_1']
X.head()
X_1=pd.DataFrame(X)
y=train['feature_2']
y_1=pd.DataFrame(y)
y.head()
lr=LinearRegression()
lr.fit(X_1,y_1)
print(lr.intercept_)# Beta nod
print(lr.coef_)
lr.predict(X_1)
sns.jointplot(x=train['feature_1'],y=train['feature_2'],data=train,kind='reg')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_1,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(y_train.shape)
lr2=LinearRegression()
lr2.fit(X_1,y)
#Now let get the predicted value of test set
y_test_a=lr2.predict(X_test)
##Now let get the predicted value of train set
y_train_a=lr2.predict(X_train)
from sklearn.metrics import r2_score

r2_score(y_test,y_test_a)
r2_score(y_train,y_train_a)
