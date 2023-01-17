import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
raw_train = pd.read_csv('../input/application_train.csv')
raw_train.head()
target = 'TARGET'
id_col = 'SK_ID_CURR'

#Delete target column as it is not useful for prediction
del raw_train[id_col]

#Distribution of target variable
raw_train[target].value_counts()

#Store target variable to be used later
target_val = raw_train[target]

#Delete target column from features
del raw_train[target]
#### Fetch all continous variables
cont_vars = []
for col in raw_train.columns:
    if raw_train[col].dtype == 'int64' or raw_train[col].dtype == 'float64':
        cont_vars.append(col)
#Store number of continous variable. This will be equivalent to number of neurons in input layer
cont_train = raw_train.loc[:, cont_vars]
curr_dim = cont_train.shape[1]
#Fill NAs with mean value of column. Lot of scope of improvement here :)
cont_train = cont_train.fillna(cont_train.mean())

#Normalize features using standard scaler. We will use same standard scaler object to normalize test data
std_scale = preprocessing.StandardScaler().fit(cont_train[cont_vars])
cont_train[cont_vars] = std_scale.transform(cont_train[cont_vars])

class basic_model(torch.nn.Module):
    def __init__(self, i_dim):
        super(basic_model, self).__init__()
        self.linear1 = torch.nn.Linear(i_dim, 40)
        self.linear2 = torch.nn.Linear(40, 20)
        self.linear3 = torch.nn.Linear(20, 1)
        self.out_act = torch.nn.Sigmoid()
    
    def forward(self, x):
        h_relu = self.linear2(self.linear1(x).clamp(min=0))
        y_pred = self.out_act(self.linear3(h_relu))
        return y_pred
batch_size = 256
in_dim = curr_dim
model = basic_model(in_dim)
criterion = torch.nn.BCELoss()
learning_rate = .001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  betas=(0.9, 0.999))
def train_epoch(model, criterion, optimizer, batch_size=batch_size):
    #model.train()
    losses = []
    
    for beg_i in range(0, cont_train.shape[0], batch_size):
        x_batch = cont_train.loc[beg_i:beg_i + batch_size, :]
        y_batch = target_val.loc[beg_i:beg_i + batch_size]
        input_data = torch.from_numpy(np.array(x_batch, dtype=np.float32))
        target_data = torch.from_numpy(np.array(y_batch, dtype=np.float32))
        y_pred = model(input_data)
        loss = criterion(y_pred, target_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.numpy())
        

    return losses

e_losses = []
num_epochs = 2
for e in range(num_epochs):
    #losses = train_epoch(model, criterion, optimizer)
    e_losses += train_epoch(model, criterion, optimizer)
plt.plot(e_losses)
test_data = pd.read_csv('../input/application_test.csv')

#Store ids
test_id = test_data.SK_ID_CURR

#Use only continous features
test_data = test_data.loc[:, cont_vars]

#Fill NA values
test_data[cont_vars] = test_data[cont_vars].fillna(raw_train[cont_vars].mean())

#Normalize data
test_data[cont_vars] = std_scale.transform(test_data[cont_vars])

#Convert to tensor
torch_test_data = torch.from_numpy(np.array(test_data, dtype=np.float32))

#Make predictions
probs = model(torch_test_data)

#Convert to numpy
probs = probs.detach().numpy()

#Prepare results
result = test_id.to_frame()
result[target] = probs
result.to_csv('pytorch_first.csv', index=False)
