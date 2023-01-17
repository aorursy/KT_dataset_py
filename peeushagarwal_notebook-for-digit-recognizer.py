import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.style.use('dark_background')
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print('Raw data:')
print(f'- Train data: {train_df.shape}')
print(f'- Test data: {test_df.shape}')

# Train Data has 
# 1. label <= Column 0
# 2. Pixels <= Column 1 to Column 784

# Test Data has 
# 1. Pixels <= Column 0 to Column 783

train_labels = train_df['label']
train_df.drop('label', axis=1, inplace=True)

print('After fetchin label from train data:')
print(f'- Label data: {train_labels.shape}')
print(f'- Train data: {train_df.shape}')
print(f'- Test data: {test_df.shape}')
# Split train into Train and Validation dataset
X_train, X_validation, y_train, y_validation = train_test_split(train_df, train_labels, test_size = 0.2, random_state = 42)

print(f'X_train:{X_train.shape}')
print(f'y_train:{y_train.shape}')
print(f'X_validation:{X_validation.shape}')
print(f'y_validation:{y_validation.shape}')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
print(X_train.values[0].shape)
item = torch.tensor(X_train.values[0], dtype=torch.float)
print(item.size())
item_1 = item.view(1, 28, 28)
print(item_1.size())
class MNIST_Dataset(Dataset):
    def __init__(self, features, labels):
        self.inputs = torch.tensor(features.values, dtype=torch.float)
        self.labels = torch.tensor(labels.values)
        
    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        input_item = self.inputs[idx]
        return input_item.view(1, 28, 28), self.labels[idx]
batch_size = 256
trainset = MNIST_Dataset(X_train, y_train)
validset = MNIST_Dataset(X_validation, y_validation)

trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size, shuffle=False, num_workers=2)
def imshow(images, labels):
  plt.figure(figsize=(images.size()[0]*4, 4))
  plt.imshow(images.numpy().transpose((1, 2, 0)))
  plt.axis('off')
  plt.title(f'{[label.item() for label in labels]}')
  plt.show()
inputs, labels = next(iter(trainloader))

inputs = torchvision.utils.make_grid(inputs[:4])
labels = labels[:4]

imshow(inputs, labels)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
class MNIST_CNN(nn.Module):
  def __init__(self):
    super(MNIST_CNN, self).__init__()

    self.features = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1),      # (1, 28, 28) => (32, 28, 28)
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 3, 1, 1),     # (32, 28, 28) => (32, 28, 28)
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        #nn.MaxPool2d(2, stride=2),      # (32, 28, 28) => (32, 14, 14)
        nn.Conv2d(32, 32, 5, 2, 2),        # (32, 28, 28) => (32, 14, 14)
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, 1, 1),     # (32, 14, 14) => (64, 14, 14)
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, 1, 1),     # (64, 14, 14) => (64, 14, 14)
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        #nn.MaxPool2d(2, stride=2)      # (64, 14, 14) => (64, 7, 7)
        nn.Conv2d(64, 64, 5, 2, 2),        # (64, 14, 14) => (64, 7, 7)
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )
    self.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(64*7*7, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 10)
    )

  def forward(self, x):
    out = self.features(x)
    out = out.view(out.size()[0], -1)
    out = self.classifier(out)
    return out
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
%%time
num_epochs = 80

model.train()

epoch_losses = []
accuracies = []
for epoch in range(num_epochs):
  epoch_loss = 0

  total = 0
  correct = 0

  for batch_id, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()

    total += len(labels)
    correct += (preds == labels).sum()

    if batch_id != 0 and batch_id % 128 == 0:
      print(f'Epoch:{epoch}, batch:{batch_id}, loss:{loss}, avg_loss:{epoch_loss/batch_id}, accuracy={100.0*correct/total}')
  
  epoch_loss = epoch_loss/len(trainloader)
  accuracy = 100.0*correct/total
  print(f'Epoch loss: {epoch_loss}, accuracy={accuracy}')
  epoch_losses.append(epoch_loss)
  accuracies.append(accuracy)

print('Training completed')
plt.subplot(121)
plt.plot(range(num_epochs), epoch_losses, 'r--')
plt.title('Loss at each epoch')
plt.subplot(122)
plt.plot(range(num_epochs), accuracies, 'g-')
plt.ylim(bottom=97)
plt.title('Accuracy at each epoch')
plt.show()
model.eval()

with torch.no_grad():
  total = 0
  correct = 0
  conf_mat = np.zeros((10, 10))
  for batch_id, data in enumerate(validloader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)

    for x, y in zip(preds.cpu().numpy(), labels.cpu().numpy()):
      conf_mat[x][y] += 1

    total += len(labels)
    correct += (preds == labels).sum()
  
  validation_accuracy = 100.0*correct/total

print(f'Validation accuracy: {validation_accuracy}')
def plot_conf_matrix(confusion_matrix):
  classes = np.arange(10)
  fig, ax = plt.subplots()
  im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
  ax.figure.colorbar(im, ax=ax)
  ax.set(xticks=np.arange(confusion_matrix.shape[1]),
          yticks=np.arange(confusion_matrix.shape[0]),
          xticklabels=classes, yticklabels=classes,
          ylabel='True label',
          xlabel='Predicted label',
          title='Epoch %d' % epoch)
  thresh = confusion_matrix.max() / 2.
  for i in range(confusion_matrix.shape[0]):
      for j in range(confusion_matrix.shape[1]):
          ax.text(j, i, int(confusion_matrix[i, j]),
                  ha="center", va="center",
                  color="white" if confusion_matrix[i, j] > thresh else "black")
    
  fig.tight_layout()
plot_conf_matrix(conf_mat)
test_df.shape
class MNIST_Test_Dataset(Dataset):
    def __init__(self, features):
        self.inputs = torch.tensor(features.values, dtype=torch.float)
        
    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        input_item = self.inputs[idx]
        return input_item.view(1, 28, 28)
batch_size = 1
testset = MNIST_Test_Dataset(test_df)
testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)
inputs = next(iter(testloader))
inputs = torchvision.utils.make_grid(inputs[:4])

plt.figure(figsize=(16, 4))
plt.imshow(inputs.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.show()
model.eval()

predictions = {}
with torch.no_grad():
    for batch_id, inputs in enumerate(testloader, 1):
        inputs = inputs.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        
        predictions[batch_id] = preds.item()


import csv

with open('predictions_4.csv', mode='w') as csv_file:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for k,v in predictions.items():
        writer.writerow({'ImageId': k, 'Label': v})
predictions[1]
