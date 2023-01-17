import warnings

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = None

pd.options.display.max_rows = None

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_auc_score

%matplotlib inline

warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

from sklearn.model_selection import GridSearchCV,StratifiedKFold

# Pytorch Imports

import torch

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader

from torch import nn, optim

import torch.nn.functional as F

from torch import optim

torch.manual_seed(0)

np.random.seed(0)

import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
print('Train set shape:', train_df.shape)

print('Test set shape:', test_df.shape)

#print('Train set overview:')

#display(train_df.head())
train_df['magic_count'] = train_df.groupby(['wheezy-copper-turtle-magic'])['id'].transform('count')

test_df['magic_count'] = test_df.groupby(['wheezy-copper-turtle-magic'])['id'].transform('count')

train_df = pd.concat([train_df, pd.get_dummies(train_df['wheezy-copper-turtle-magic'])], axis=1, sort=False)

test_df = pd.concat([test_df, pd.get_dummies(test_df['wheezy-copper-turtle-magic'])], axis=1, sort=False)
train_df.shape , test_df.shape
cols_to_drop = ['target','id']

X= train_df.drop(cols_to_drop, axis = 1)

X_test = test_df.drop(['id'],axis = 1)

Y = train_df['target']
X.shape , X_test.shape , Y.shape
#### Scaling feature #####

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

X_test = sc.transform(X_test)
X_test = torch.tensor(X_test , dtype=torch.float)
X_test.shape
class Model(nn.Module):

    def __init__(self):

        super(Model,self).__init__()

        self.fc1 = nn.Linear(769,512)

        self.bn1 = nn.BatchNorm1d(num_features=512)

        self.fc2 = nn.Linear(512,256)

        self.bn2 = nn.BatchNorm1d(num_features=256)

        self.fc3 = nn.Linear(256,100)

        self.bn3 = nn.BatchNorm1d(num_features=100)

        self.fc4 = nn.Linear(100,1)

        

        self.dropout1 = nn.Dropout(p=0.4)

        self.dropout2 = nn.Dropout(p=0.4)

        self.dropout3 = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()

        

    def forward(self,x):

        # input tensor is flattened 

        x = x.view(x.shape[0], -1)

        #print(x.shape)

        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))

        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))

        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))

        x = self.sigmoid(self.fc4(x))

        

        return x
print(torch.cuda.is_available())

# Creating a device object 

device = torch.device(

    "cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)
n_splits = 7

kf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=0)

pred_test_full =0

cv_score =[]

i=1

for train_index,test_index in kf.split(X,Y):

    print('{} of KFold {}'.format(i,kf.n_splits))

    

    # Train & Val Split

    

    #X_train,X_val = X.loc[train_index],X.loc[test_index]

    #Y_train,Y_val = Y.loc[train_index],Y.loc[test_index]

    

    X_train,X_val = X[train_index , :],X[test_index , :]

    Y_train,Y_val = Y[train_index],Y[test_index]

    

    

    # Train & Val Tensors

    X_train = torch.tensor(X_train , dtype=torch.float)

    Y_train = torch.tensor(Y_train.values, dtype=torch.float)

    X_val = torch.tensor(X_val , dtype=torch.float)

    Y_val = torch.tensor(Y_val.values, dtype=torch.float)

    

    # Tensor DataSet

    train = torch.utils.data.TensorDataset(X_train, Y_train)

    val = torch.utils.data.TensorDataset(X_val, Y_val)

    

    # how many samples per batch to load

    batch_size = 120

    

    # data loaders preparation

    train_loader = DataLoader(train, batch_size=batch_size)

    valid_loader = DataLoader(val, batch_size=batch_size)

    

    model = Model()

    criterion = nn.BCELoss() 

    optimizer = optim.Adam(model.parameters(), lr=0.0001 , weight_decay=0.003)

    model.to(device)

    

    for epoch in range(1, 31): 

        train_loss, valid_loss = [], []

        ## training part 

        

        model.train()

        for data, target in train_loader:

            # Move input and label tensors to the avialable device

            data, target = data.to(device), target.to(device)

            #print(data.shape)

            optimizer.zero_grad()

            ## 1. forward propagation

            output = model(data)

            output = output.view(output.shape[0])



            ## 2. loss calculation

            loss = criterion(output, target)



            ## 3. backward propagation

            loss.backward()



            ## 4. weight optimization

            optimizer.step()



            train_loss.append(loss.item())

        #print('train complete')

        ## evaluation part

        with torch.no_grad():

            model.eval()

            for data, target in valid_loader:

                # Move input and label tensors to the avialable device

                data, target = data.to(device), target.to(device)

                output = model(data)

                output = output.view(output.shape[0])

                loss = criterion(output, target)

                valid_loss.append(loss.item())

        #print('validation complete')

        if epoch > 0 and epoch % 10 == 0:

            print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

    

    prediction = model(X_test.cuda())

    prediction = prediction.cpu()

    prediction = prediction.data.numpy()

    pred_test_full +=prediction

    i+=1
y_pred = pred_test_full/n_splits

sub['target'] = y_pred

sub.to_csv('submission_pytorch_KFold_v1.csv', index = False, header = True)
sub.head()