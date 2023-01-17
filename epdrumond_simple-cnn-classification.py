import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
#Load train data
data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
data.head()
#Get simple description of the data set
data.describe()
x_data = data.drop(columns = 'label')
y_data = data['label']
fig, axs = plt.subplots(2, 2)
fig.tight_layout(pad = 3.0)
for i in range(2):
    for j in range(2):
        img = np.random.randint(0, len(x_data))
        axs[i,j].imshow(x_data.loc[img].values.reshape(28, 28))
        axs[i,j].set_title(y_data[img])
#Define function to scale the dataset
def scale_data(data, scaler, opt = 0): #opt: 0 -> apply fit_trandform, 1 -> apply only transform
    if opt == 0:
        return scaler.fit_transform(data)
    else:
        return scaler.transform(data)
#Scale train dataset
my_scaler = MinMaxScaler()
scaled_x_data = scale_data(x_data, my_scaler, 0)

print('Original dataset: \n', x_data.values[0:3], '\n\n')
print('Scaled dataset: \n', scaled_x_data[0:3])
#Convert data entries from 1D to 2D
scaled_x_data = np.asarray([x.reshape(28, 28) for x in scaled_x_data])
print('Sample converted image: \n')
print(scaled_x_data[0])
#Selecting device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Device: ', device)
#Split data into train, test and validation
x_train, x_test, y_train, y_test = train_test_split(scaled_x_data, y_data,
                                                    test_size = 0.25,
                                                    stratify = y_data)

print('Train size: %d \n Test size: %d' %(len(x_train), len(x_test)))
#Define model hyperparameters
param = {
    'num_jobs': 2,
    'batch_size': 128,
    'num_epochs': 100
}
#Define class to load the dataset as a tensor
class MnistData(Dataset):
    
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        sample = torch.from_numpy(self.x_data[index].astype(np.float32).reshape(1, 28, 28))
        target = torch.from_numpy(self.y_data[index].astype(np.float32))
        return (sample, target)
    
    def __len__(self):
        return len(self.x_data)
#Load data as tensors
train_data = MnistData(x_train, y_train.values.reshape(len(y_train), 1))
test_data = MnistData(x_test, y_test.values.reshape(len(y_test), 1))
#validation_data = MnistData(x_validation, y_validation.values.reshape(len(y_validation), 1))
#Create DataLoader
train_loader = DataLoader(train_data,
                          batch_size = param['batch_size'],
                          shuffle = True,
                          num_workers = param['num_jobs'])
test_loader = DataLoader(test_data,
                         batch_size = param['batch_size'],
                         shuffle = True,
                         num_workers = param['num_jobs'])
class MnistClassification(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        #Inicializar classe pai
        super(MnistClassification, self).__init__()
        self.input_dim = input_dim
        
        #Preprocessing layers
        self.conv_01 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 6, stride = 1, padding = 1)
        self.conv_bn_01 = nn.BatchNorm2d(num_features = 16)
        self.pool_01 = nn.MaxPool2d(kernel_size = 3, stride = 1)
        
        self.conv_02 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 1)
        self.conv_bn_02 = nn.BatchNorm2d(num_features = 32)
        self.pool_02 = nn.MaxPool2d(kernel_size = 3, stride = 1)
        
        self.conv_03 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 1)
        self.conv_bn_03 = nn.BatchNorm2d(num_features = 64)
        self.pool_03 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        #Linear an normalization layers
        self.norm_01 = nn.BatchNorm1d(num_features = 2048)
        self.layer_01 = nn.Linear(in_features = 4096, out_features = 2048)
        self.norm_02 = nn.BatchNorm1d(num_features = 1024)
        self.layer_02 = nn.Linear(in_features = 2048, out_features = 1024)
        self.norm_03 = nn.BatchNorm1d(num_features = 512)
        self.layer_03 = nn.Linear(in_features = 1024, out_features = 512)
        self.norm_04 = nn.BatchNorm1d(num_features = 256)
        self.layer_04 = nn.Linear(in_features = 512, out_features = 256)
        self.norm_05 = nn.BatchNorm1d(num_features = 128)
        self.layer_05 = nn.Linear(in_features = 256, out_features = 128)
        self.norm_06 = nn.BatchNorm1d(num_features = 64)
        self.layer_06 = nn.Linear(in_features = 128, out_features = 64)
        self.output_layer = nn.Linear(in_features = 64, out_features = output_dim)
        
        #Activation and dropout layers
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
            
    def forward(self, x_data):
        
        x_data = x_data.to(device)
        response = self.relu(self.conv_bn_01(self.conv_01(x_data)))
        response = self.pool_01(response)
        response = self.relu(self.conv_bn_02(self.conv_02(response)))
        response = self.pool_02(response)
        response = self.relu(self.conv_bn_03(self.conv_03(response)))
        response = self.pool_03(response)
        
        response = self.dropout(self.relu(self.norm_01(self.layer_01(response.reshape(len(x_data),4096)))))
        response = self.dropout(self.relu(self.norm_02(self.layer_02(response))))
        response = self.dropout(self.relu(self.norm_03(self.layer_03(response))))
        response = self.dropout(self.relu(self.norm_04(self.layer_04(response))))
        response = self.dropout(self.relu(self.norm_05(self.layer_05(response))))
        response = self.dropout(self.relu(self.norm_06(self.layer_06(response))))
        
        response = self.softmax(self.output_layer(response))
        
        return response
#Create MLP instance
param['input_dim'] = x_data.shape[1]
param['output_dim'] = len(y_data.unique())
param['num_layers'] = 3

model = MnistClassification(input_dim = param['input_dim'],
                            output_dim = param['output_dim']).to(device)
#Define loss function and optimizer
param['learning_rate'] = 1e-4
param['weight_decay'] = 5e-3

loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params = model.parameters(),
                       lr = param['learning_rate'],
                       weight_decay = param['weight_decay'])
#Define training model function
def train_model(estimator, train_data, epoch):
    
    #Toogle training mode
    model.train()
    
    epoch_error = []
    epoch_accuracy = []
    
    for batch in train_data:
        
        #Update learning rate 
        optimizer = optim.Adam(params = model.parameters(),
                               lr = param['learning_rate'] * (1/(10**((epoch/10) + 1))),
                               weight_decay = param['weight_decay'])

        #Forward process
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_response = model(x_batch)
                
        #Compute error
        batch_error = loss_function(batch_response, torch.squeeze(y_batch, 1).type(torch.LongTensor).to(device))
        epoch_error.append(batch_error.cpu().data)
        
        batch_accuracy = accuracy_score(torch.squeeze(y_batch, 1).type(torch.LongTensor).cpu().numpy(), 
                                        np.argmax(batch_response.detach().cpu().numpy(), 1))
        epoch_accuracy.append(batch_accuracy)
        
        #Backward process
        optimizer.zero_grad()
        batch_error.backward()
        optimizer.step()
        
    epoch_error = np.asarray(epoch_error)
    epoch_accuracy = np.asarray(epoch_accuracy)
    
    print('Epoch %d TRAIN error: %.4f +/- %.4f / accuracy: %.4f' %(epoch+1, epoch_error.mean(), 
                                                                   epoch_error.std(), epoch_accuracy.mean()))
    
    return [epoch_error.mean(), epoch_accuracy.mean()] 
#Define testing model function
def test_model(estimator, test_data, epoch):
    
    #Toogle training mode
    model.eval()
    with torch.no_grad():
    
        epoch_error = []
        epoch_accuracy = []
        for batch in test_data:
            #Forward process
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_response = model(x_batch)

            #Compute error
            batch_error = loss_function(batch_response, torch.squeeze(y_batch, 1).type(torch.LongTensor).to(device))
            epoch_error.append(batch_error.cpu().data)
            
            batch_accuracy = accuracy_score(torch.squeeze(y_batch, 1).type(torch.LongTensor).cpu().numpy(), 
                                            np.argmax(batch_response.detach().cpu().numpy(), 1))
            epoch_accuracy.append(batch_accuracy)

        epoch_error = np.asarray(epoch_error)
        epoch_accuracy = np.asarray(epoch_accuracy)
        
        print('Epoch %d TEST error: %.4f +/- %.4f / accuracy: %.4f' %(epoch+1, epoch_error.mean(), 
                                                                      epoch_error.std(), epoch_accuracy.mean()))

        return [epoch_error.mean(), epoch_accuracy.mean()]
#Train model
param['num_epochs'] = 50

train_error = []
test_error = []

for epoch in range(param['num_epochs']):
    train_error.append(train_model(model, train_loader, epoch))
    test_error.append(test_model(model, test_loader, epoch))
    print('-----------------------------------------------------------')
#Plot loss function profile
plt.plot(list(range(param['num_epochs'])), [x[0] for x in train_error])
plt.plot(list(range(param['num_epochs'])), [x[0] for x in test_error])
plt.legend(['Train', 'Test'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
#Plot accuracy profile
plt.plot(list(range(param['num_epochs'])), [x[1] for x in train_error])
plt.plot(list(range(param['num_epochs'])), [x[1] for x in test_error])
plt.legend(['Train', 'Test'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#Load test data
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
#Adjust test data
test_data_id = test_data['id']

scaled_test_data = scale_data(test_data.drop(columns = 'id'), my_scaler, 1)
scaled_test_data = np.asarray([x.reshape(1, 28, 28) for x in scaled_test_data])
pred = np.argmax(model.forward(torch.from_numpy(scaled_test_data.astype(np.float32))).detach().cpu().numpy(), 1)
submission_data = pd.DataFrame(data = test_data_id.values, columns = ['id'])
submission_data['label'] = pred
print(submission_data.head())
submission_data.to_csv('submission.csv', index = False)