import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/training/training.csv')

print(train_data.shape)

train_data.isna().sum()

# train_data.isnull().any().value_counts()
# Using ffill to fill the na values:

train_data.fillna(method='ffill', inplace=True)
img_dt = []



for i in range(len(train_data)):

  img_dt.append(train_data['Image'][i].split(' '))

  

X = np.array(img_dt, dtype='float')
# Visualizing one of the images:

plt.imshow(X[1].reshape(96,96), cmap='gray')
facial_pts_data = train_data.drop(['Image'], axis=1)

facial_pts = []



for i in range(len(facial_pts_data)):

  facial_pts.append(facial_pts_data.iloc[i])

  

y = np.array(facial_pts, dtype='float')
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data_utils

from torch.utils.data.sampler import SubsetRandomSampler





# class Basic_CNN(nn.Module):

#     def __init__(self):

#         super(Basic_CNN, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5) # (1,1,96,96) to (1,4,92,92)

# #         self.conv1_bn = nn.BatchNorm2d(16)

#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5) # (1,4,46,46) to (1,8,42,42)

# #         self.conv2_bn = nn.BatchNorm2d(32)

#         self.fc1 = nn.Linear(32*21*21, 250)

#         self.fc2 = nn.Linear(250, 30)

#         self.dp1 = nn.Dropout(p=0.4)

    

        

    

#     def forward(self, x, verbose=False):

# #         x = self.conv1_bn(self.conv1(x))

#         x = self.conv1(x)

#         x = F.relu(x)

#         x = F.max_pool2d(x, kernel_size=2)

#         x = self.dp1(x)

# #         x = self.conv2_bn(self.conv2(x))

#         x = self.conv2(x)

#         x = F.relu(x)

#         x = F.max_pool2d(x, kernel_size=2)

#         x = self.dp1(x)

#         x = x.view(-1, 32*21*21)

#         x = self.fc1(x)

#         x = F.relu(x)

#         x = self.dp1(x)

#         x = self.fc2(x)

#         return x

      

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5) # (b,1,96,96) to (b,4,92,92)

        self.conv1_bn = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3) # (b,4,46,46) to (b,64,44,44)

        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3) # (b,64,22,22) to (b,128,20,20)

        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3) # (b,128,10,10) to (b,256,8,8)

        self.conv4_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*4*4, 1024)

        self.fc2 = nn.Linear(1024, 256)

        self.fc3 = nn.Linear(256, 30)

        self.dp1 = nn.Dropout(p=0.4)

    

        

    

    def forward(self, x, verbose=False):

        # apply conv1, relu and maxpool2d

        x = self.conv1_bn(self.conv1(x))

        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2)

        x = self.dp1(x)

        

        # apply conv2, relu and maxpool2d

        x = self.conv2_bn(self.conv2(x))

        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2)

        x = self.dp1(x)

        

        # apply conv3, relu and maxpool2d

        x = self.conv3_bn(self.conv3(x))

        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2)

        x = self.dp1(x)

        

        # apply conv4, relu and maxpool2d

        x = self.conv4_bn(self.conv4(x))

        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2)

        

        # apply dropout

        x = self.dp1(x)

        

        x = x.view(-1, 256*4*4)

        

        # now use FC layer with relu

        x = self.fc1(x)

        x = F.relu(x)

        x = self.dp1(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.dp1(x)

        x = self.fc3(x)

        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def testing(model, device, valid_loader):

  model.eval()

  test_loss = 0

  for data, target in valid_loader:

    data, target = data.to(device), target.to(device)

    data = data.view(-1, 96*96)

    data = data.view(-1, 1, 96, 96)

    output = model(data)

    loss = criterion(output, target)

    test_loss += loss.item()

    

  test_loss /= len(valid_loader.dataset)

  return test_loss



def training(epochs, model, criterion, device, train_loader, valid_loader, optimizer):

  train_error_list = []

  val_error_list = []

  for epoch in range(epochs):

    model.train()

    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

      data, target = data.to(device), target.to(device)

      data = data.view(-1, 96*96)

      data = data.view(-1, 1, 96, 96)

      optimizer.zero_grad()

      output = model(data)

      loss = criterion(output, target)

      train_loss += loss.item()

      loss.backward()

      optimizer.step()

    

    train_loss /= len(train_loader.dataset)

    eval_loss = testing(model, device, valid_loader)

    train_error_list.append(train_loss)

    val_error_list.append(eval_loss)

    if (epoch+1) % 25 == 0:

      print("End of epoch {}: \nTraining error = [{}]\tValidation error = [{}]".format(epoch+1, train_loss, eval_loss))

  return train_error_list, val_error_list

def train_test_split(X, validation_split):

  dataset_size = len(X)

  indices = list(range(dataset_size))

  val_num = int(np.floor(validation_split*dataset_size))

  np.random.shuffle(indices)

  train_indices, val_indices = indices[val_num:], indices[:val_num]



  train_sampler = SubsetRandomSampler(train_indices)

  valid_sampler = SubsetRandomSampler(val_indices)



  loader_object = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())

  train_loader = data_utils.DataLoader(loader_object, batch_size=32, sampler=train_sampler)

  valid_loader = data_utils.DataLoader(loader_object, batch_size=32, sampler=valid_sampler)

  return train_loader, valid_loader
def get_n_params(model):

    np=0

    for p in list(model.parameters()):

        np += p.nelement()

    return np



n_hidden = 128 # number of hidden units

output_size = 30

train_loader, valid_loader = train_test_split(X, 0.2)



model = CNN()

model.to(device)

criterion = torch.nn.MSELoss() 

optimizer = optim.Adam(model.parameters())



print('Number of parameters: {}'.format(get_n_params(model)))



train_error_list, valid_error_list = training(500, model, criterion, device, train_loader, valid_loader, optimizer)
def plot_samples(X, y, model, num_samples):

  fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(12,12))

  

  for row in range(num_samples):

    sample_idx = np.random.choice(len(X))

    x = X[sample_idx]

    x = torch.from_numpy(x).float().view(1,1,96,96).to(device)

    actual_y = y[sample_idx]

    pred_y = model(x)

    img = X[sample_idx].reshape(96,96)

    

    actual_y = np.vstack(np.split(actual_y, 15)).T

    pred_y = pred_y.cpu().data.numpy()[0]

    pred_y = np.vstack(np.split(pred_y, 15)).T

    

    axes[row, 0].imshow(img, cmap='gray')

    axes[row, 0].plot(actual_y[0], actual_y[1], 'o', color='red', label='actual')

    axes[row, 0].legend()

    axes[row, 1].imshow(img, cmap='gray')

    axes[row, 1].plot(actual_y[0], actual_y[1], 'o', color='red', label='actual')

    axes[row, 1].plot(pred_y[0], pred_y[1], 'o', color='green', label='predicted')

    axes[row, 1].legend()



  

plot_samples(X, y, model, 3)
test_data = pd.read_csv('../input/test/test.csv')



img_dt = []



for i in range(len(test_data)):

  img_dt.append(test_data['Image'][i].split(' '))

  

test_X = np.array(img_dt, dtype='float')
test_X_torch = torch.from_numpy(test_X).float().view(len(test_X),1,96,96).to(device)

test_predictions = model(test_X_torch)

test_predictions = test_predictions.cpu().data.numpy()



keypts_labels = train_data.columns.tolist()
# Visualizing the outputs:



def plot_samples_test(X, y, num_samples):

  fig, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=(20,12))

  

  for row in range(num_samples):

    sample_idx = np.random.choice(len(X))

    img = X[sample_idx].reshape(96,96)

    predicted = y[sample_idx]

    

    predicted = np.vstack(np.split(predicted, 15)).T

#     print(img, predicted)

    axes[row].imshow(img, cmap='gray')

    axes[row].plot(predicted[0], predicted[1], 'o', color='green', label='predicted')

    axes[row].legend()

  

plot_samples_test(test_X, test_predictions, 6)
id_lookup = pd.read_csv('../input/IdLookupTable.csv')

id_lookup_features = list(id_lookup['FeatureName'])

id_lookup_image = list(id_lookup['ImageId'])



for i in range(len(id_lookup_features)):

  id_lookup_features[i] = keypts_labels.index(id_lookup_features[i])



location = []

for i in range(len(id_lookup_features)):

  location.append(test_predictions[id_lookup_image[i]-1][id_lookup_features[i]])
id_lookup['Location'] = location
submission = id_lookup[['RowId', 'Location']]

submission.head(5)
submission.to_csv('../input/SubmissionFile.csv')