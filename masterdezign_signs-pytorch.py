import numpy as np # linear algebra

import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.nn as nn
from torch.utils import data
print('pytorch', torch.__version__)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import h5py
        
def load_dataset(cuda=torch.cuda.is_available()):            
    """
    Returns train set features, train set labels, test set features, labels, classes
    """
    train_dataset = h5py.File('../input/train_signs.h5', "r")
    train_set_x_orig = torch.tensor(train_dataset["train_set_x"][:])
    train_set_y_orig = torch.tensor(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('../input/test_signs.h5', "r")
    test_set_x_orig = torch.tensor(test_dataset["test_set_x"][:])
    test_set_y_orig = torch.tensor(test_dataset["test_set_y"][:])

    classes = torch.tensor(test_dataset["list_classes"][:])
    
    return train_set_x_orig.type(torch.FloatTensor), \
           train_set_y_orig.type(torch.LongTensor), \
           test_set_x_orig.type(torch.FloatTensor),  \
           test_set_y_orig.type(torch.LongTensor), \
           classes
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Example of a picture
index = 0
im = X_train_orig[index].cpu().numpy() / 255.
print("y =", str(Y_train_orig[index].item()))
plt.imshow(im)
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1)
print(X_train_flatten.shape)
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1)
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

# No need to convert training and test labels to one hot matrices.
# Using CrossEntropy criterium expects labels in range 0..N-1
Y_train = Y_train_orig
Y_test = Y_test_orig

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

train_set = data.dataset.TensorDataset(X_train, Y_train)
test_set = data.dataset.TensorDataset(X_test, Y_test)
# Sanity check
x0, y0 = train_set[0]
print(x0.shape, y0)
class FC_NET(nn.Module):
    def __init__(self, num_classes=6, input_chan=1):
        super(FC_NET, self).__init__()

        self.classifier = nn.Sequential(            
            nn.Linear(12288, 25),
            nn.ReLU(),
            
            nn.Linear(25, 12),
            nn.ReLU(),
            
            nn.Linear(12, 6)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(1)
net = FC_NET()
net.to(device)
batch_size = 32

train_loader = data.DataLoader(train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=1  # 1 for CUDA
                               )

test_loader = data.DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=1  # 1 for CUDA
                              )
# Train
epochs = 1500

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

losses = []  # To track training losses
for epoch in range(epochs):
    cur_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward and backward passes
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        cur_loss += loss.item()
        
    losses.append((epoch, cur_loss))
    
    if epoch % 20 == 0:
        print('Epoch %d, loss: %.3f' % (epoch, cur_loss))
losses = np.array(losses)
print(losses.shape)
plt.plot(losses[:, 0], losses[:, 1])
plt.xlabel('Epoch')
plt.ylabel('Loss')
def validate(net, loader, device=None):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            xs, labels = data
            xs, labels = xs.to(device), labels.to(device)
            outputs = net(xs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print("Correct:", predicted == labels)
            correct += (predicted == labels).sum().item()
    return (correct, total)
(correct_learning, total_learning) = validate(net, train_loader, device=device)
(correct_validat, total_validat) = validate(net, test_loader, device=device)

print('Learning accuracy: %d %% (%d correct of %d)' % (
    100 * correct_learning / total_learning, correct_learning, total_learning))

print('Validation accuracy: %d %% (%d correct of %d)' % (
    100 * correct_validat / total_validat, correct_validat, total_validat))
