import pandas as pd
df = pd.read_csv("../input/devadata24/data.csv")
df.head()
#shuffling the dataframe
df = df.sample(frac = 1) 
df.head()
df.isnull().sum().sum()
labels = df['character']
labels
labels.describe()
labels.unique()
import matplotlib.pyplot as plt
images = df.drop('character', 1)
images = images.values.reshape(-1,32,32,1)

plt.imshow(images[0][:,:,0], interpolation='nearest')
plt.show()
import torch
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
targets = le.fit_transform(labels)
# targets: array([0, 1, 2, 3])

targets = torch.as_tensor(targets)
targets
le.inverse_transform(targets)
import numpy as np
import torchvision.transforms as transforms
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
images = images.astype('float')
images = torch.tensor(images)
images = torch.true_divide(images, 255.0)
type(images), type(targets)
train_size = int(92000*0.8)
train_images = images[:train_size]
train_target = targets[:train_size]
test_images = images[train_size:]
test_target = targets[train_size:]
train_data = torch.utils.data.TensorDataset(train_images, train_target)
test_data = torch.utils.data.TensorDataset(test_images, test_target)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
import matplotlib.pyplot as plt
%matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
print(labels)
truel = le.inverse_transform(labels)

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(truel[idx]))
img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 768
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(32 * 32, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (hidden_2 -> 46)
        self.fc3 = nn.Linear(hidden_2, 46)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 32 * 32)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

# initialize the NN
model = Net()
print(model)
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model
n_epochs = 60

model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        data = data.float()
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
             
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))
test_loss = 0.0
class_correct = list(0. for i in range(46))
class_total = list(0. for i in range(46))

model.eval() # prep model for evaluation

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    data = data.float()
    output = model(data)
    # calculate the loss
    target = target.squeeze_()
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(46):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)')

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
dataiter = iter(test_loader)
images, labels = dataiter.next()
labels = le.inverse_transform(labels)

# get sample outputs
images = images.float()
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
preds = le.inverse_transform(preds)
# prep images for display
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(30, 8))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} \n({})".format(str(preds[idx]), str(labels[idx]), color=("green" if preds[idx]==labels[idx] else "red")))
PATH = "state_dict_model.pt"

# Save
torch.save(model.state_dict(), PATH)

# Load
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
images, labels = dataiter.next()
labels = le.inverse_transform(labels)

# get sample outputs
images = images.float()
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
preds = le.inverse_transform(preds)
# prep images for display
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(30, 8))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} \n({})".format(str(preds[idx]), str(labels[idx]), color=("green" if preds[idx]==labels[idx] else "red")))
