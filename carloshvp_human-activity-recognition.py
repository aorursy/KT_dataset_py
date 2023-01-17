# Installing Jovian dependency
!pip install jovian --upgrade --quiet
# Importing required dependencies
import jovian
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Initial project configuration
result = []
project_name = 'Human Activity Recognition'
arch = "Convolution + pooling + convolution + pooling + dense + dense + dense + output"
batch_size = 64
epochs = 50
lr = 0.01
momentum = 0.9
torch.manual_seed(29)
# Defining data pre-processing functions and Data_loader class
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print(x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print(X.shape)
    return X

def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(6)[data]
    return YY

def load_data():
    str_folder = '../input/uci-human-activity-recognition/' + 'UCI HAR Dataset/'
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]
    str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                       INPUT_SIGNAL_TYPES]
    str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
                      item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
    str_train_y = str_folder + 'train/y_train.txt'
    str_test_y = str_folder + 'test/y_test.txt'

    X_train = format_data_x(str_train_files)
    X_test = format_data_x(str_test_files)
    Y_train = format_data_y(str_train_y)
    Y_test = format_data_y(str_test_y)

    return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)

def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]

class Data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)    

def normalize(x):
    x_min = x.min(axis=(0, 2, 3), keepdims=True)
    x_max = x.max(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

def load(batch_size=64):
    x_train, y_train, x_test, y_test = load_data()
    x_train, x_test = x_train.reshape(
        (-1, 9, 1, 128)), x_test.reshape((-1, 9, 1, 128))
    transform = None
    train_set = Data_loader(x_train, y_train, transform)
    test_set = Data_loader(x_test, y_test, transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
# Defining network architecture
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 26, out_features=1000),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=6)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(-1, 64 * 26)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out
# Train and plot functions
def train(model, optimizer, train_loader, test_loader):
    n_batch = len(train_loader.dataset) // batch_size
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        model.train()
        correct, total_loss = 0, 0
        total = 0
        for index, (sample, target) in enumerate(train_loader):
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()
            sample = sample.view(-1, 9, 1, 128)
            output = model(sample)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

        acc_train = float(correct) * 100.0 / (batch_size * n_batch)
        print(f'Epoch: [{e+1}/{epochs}], loss: {total_loss}, train acc: {acc_train}%')

        # We proceed now to use the test data to evaluate intermediate results without modifying the model (no training)
        model.train(False)
        with torch.no_grad():
            correct, total = 0, 0
            for sample, target in test_loader:
                sample, target = sample.to(
                    DEVICE).float(), target.to(DEVICE).long()
                sample = sample.view(-1, 9, 1, 128)
                output = model(sample)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
        acc_test = float(correct) * 100 / total
        print(f'Epoch: [{e+1}/{epochs}], test acc: {float(correct) * 100 / total}%')
        result.append([acc_train, acc_test])
        result_np = np.array(result, dtype=float)
        np.savetxt('result.csv', result_np, fmt='%.2f', delimiter=',')


def plot():
    data = np.loadtxt('result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Test Accuracy', fontsize=20)
    plt.show()
# Get GPU if available
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
DEVICE = get_default_device()
DEVICE
# Loading the data
train_loader, test_loader = load(
        batch_size=batch_size)
# Load to selected device and traing model
model = Network().to(DEVICE)
optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
train(model, optimizer, train_loader, test_loader)
result = np.array(result, dtype=float)

# Saving results to a csv file which will be used by the plot function
np.savetxt('result.csv', result, fmt='%.2f', delimiter=',')
# Plotting the accuracy for the train and test data
plot()
# We save the current model
torch.save(model.state_dict(), 'human-activity-recognition.pth')
# Save hyperparameters, commit this notebook to Jovian.ml
jovian.log_hyperparams(arch=arch, 
                       lrs=lr, 
                       epochs=epochs)
train_acc = result[len(result)-1,0]
test_acc = result[len(result)-1,1]
jovian.log_metrics(train_acc=train_acc, test_acc=test_acc)
jovian.commit(project=project_name, outputs=['human-activity-recognition.pth'], environment=None)