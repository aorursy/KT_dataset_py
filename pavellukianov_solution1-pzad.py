# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tqdm import tqdm

import torch.optim as optim

import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_features.head()
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

test_features.head()
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_scored.head()
train_features = train_features.drop(columns = ['sig_id'])

y_test = pd.DataFrame(test_features['sig_id'])

test_features = test_features.drop(columns = ['sig_id'])
cat_columns = ['cp_type', 'cp_time', 'cp_dose']

cat_train_features = train_features[cat_columns]

cat_test_features = test_features[cat_columns]

train_features = train_features.drop(columns = cat_columns)

test_features = test_features.drop(columns = cat_columns)

X1_train = np.array(train_features)

X1_test = np.array(test_features)
st = StandardScaler()

X1_train = st.fit_transform(X1_train)

X1_test = st.transform(X1_test)
one_hot = OneHotEncoder()

X2_train = one_hot.fit_transform(cat_train_features)

X2_test = one_hot.transform(cat_test_features)
X_train = np.concatenate((X2_train.toarray(), X1_train), 1)

X_test = np.concatenate((X2_test.toarray(), X1_test), 1)
def train_epoch(model, optimizer, train_loader, criterion, device, scheduler):

    """

    for each batch 

    performs forward and backward pass and parameters update 

    

    Input:

    model: instance of model (example defined above)

    optimizer: instance of optimizer (defined above)

    train_loader: instance of DataLoader

    

    Returns:

    nothing

    

    Do not forget to set net to train mode!

    """

    model.train()

    for x_batch, y_batch in train_loader:

        x_batch = x_batch.to(device)

        y_batch = y_batch.to(device)



        optimizer.zero_grad()

        output = model(x_batch)

        

        loss = criterion(output, y_batch)

        loss.backward()

        optimizer.step()

        scheduler.step()

    



def evaluate_loss(loader, model, criterion, device):

    """

    Evaluates loss and accuracy on the whole dataset

    

    Input:

    loader:  instance of DataLoader

    model: instance of model (examle defined above)

    

    Returns:

    (loss, accuracy)

    

    Do not forget to set net to eval mode!

    """

    model.eval()

    with torch.no_grad():

        cumloss, cumacc = 0, 0

        num_objects = 0

        model.eval()

        for x_batch, y_batch in loader:

            x_batch = x_batch.to(device)

            y_batch = y_batch.to(device)

            output = model(x_batch)

            loss = criterion(output, y_batch)

            cumloss += loss.item()

            num_objects += len(x_batch)

    return cumloss / num_objects

    

    

def train(model, opt, train_loader, test_loader, criterion, n_epochs, device, scheduler, verbose=True):

    """

    Performs training of the model and prints progress

    

    Input:

    model: instance of model (example defined above)

    opt: instance of optimizer 

    train_loader: instance of DataLoader

    test_loader: instance of DataLoader (for evaluation)

    n_epochs: int

    

    Returns:

    4 lists: train_log, train_acc_log, val_log, val_acc_log

    with corresponding metrics per epoch

    """

    train_log = []

    val_log = []



    for epoch in range(n_epochs):

        train_epoch(model, opt, train_loader, criterion, device, scheduler)

        train_loss = evaluate_loss(train_loader, 

                                                  model, criterion, 

                                                  device)

        #val_loss = evaluate_loss(test_loader, model, 

        #                                      criterion, device)



        train_log.append(train_loss)



        #val_log.append(val_loss)

        

        if verbose:

             print ('Epoch', epoch+1, '/', n_epochs, 'Loss (train): ', train_loss)

            

    return train_log#, val_log
class Data(torch.utils.data.Dataset):

    def __init__(self, X, Y):

        self.X = X

        self.Y = Y

        

    def __getitem__(self, idx):

        return torch.Tensor((self.X)[idx, :]), torch.Tensor((self.Y)[idx, :])

    

    def __len__(self):

        return self.X.shape[0]

    

class DataTest(torch.utils.data.Dataset):

    def __init__(self, X):

        self.X = X

        

    def __getitem__(self, idx):

        return torch.Tensor(self.X[idx, :])

    

    def __len__(self):

        return self.X.shape[0]
if torch.cuda.is_available():

    device = 'cuda'

device
class SimpleNet(nn.Module):

    def __init__(self, num_features, num_targets, hidden_size):

        super(SimpleNet, self).__init__()

        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.layer1 = nn.Linear(num_features, hidden_size)

        

        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.layer2 = nn.Linear(hidden_size, hidden_size)

        

        self.layer3 = nn.Linear(hidden_size, num_targets)

    

    def forward(self, x):

        x = F.relu(self.bn1(self.layer1(x)))

        x = F.relu(self.bn2(self.layer2(x)))

        return self.layer3(x)
y_train = np.array(train_targets_scored[train_targets_scored.columns[1:]])

train_data = Data(X_train, y_train)

test_data = DataTest(X_test)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
criterion = nn.BCEWithLogitsLoss()

model = SimpleNet(879, 206, 1024)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, gamma = 0.5, step_size=55)

loss_stat = train(model, optimizer, train_loader, test_loader, criterion, n_epochs = 20, device = device, scheduler = scheduler, verbose=True)
model.eval()

with torch.no_grad():

    tensors = []

    for x_batch in test_loader:

        #x_batch.to(device)

        #print(x_batch.shape)

        output = model(x_batch.to(device))

        tensors.append(output)

    y_pred = torch.cat(tensors, 0)
pred = (nn.Sigmoid()(y_pred.cpu())).numpy()
i = 0

for column in train_targets_scored.columns[1:]:

    y_test[column] = pred[:, i]

    i += 1
y_test.to_csv('submission.csv', index = False)