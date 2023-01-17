# Bread and butter of Machine learning

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Importing torch libraries

import torch.nn as nn

import torch.nn.functional as F

import torch

from torch.utils.data import Dataset, DataLoader



# Bread and butter of Machine learning

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# To make waiting look fancy

from tqdm import tqdm
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train.head(), test.head()
# train.shape[0] is the number of images in the training data, and test.shape[0] holds the number of images in the test data

train.shape[0], test.shape[0]
def to_tensor(data):

    return [torch.FloatTensor(point) for point in data]



class MNISTData(Dataset):

    def __init__(self, df, X_col, Y_col=None):

        """

        We're divding the values by 255 to normalize the dataset. 

        It speeds up training. Why 255? because that's the maximum value for a pixel

        """

        self.features = df[X_col].values/255

        self.features = self.features.reshape(len(self.features), 1, 28, 28)

        self.targets = df[Y_col].values.reshape((-1, 1)) 

        # -1 indicates that the first dimension could be anything

        

    """

    To return the length of the dataset

    """

    def __len__(self):

        return len(self.targets)

    

    """

    This method will get data from the dataframe, based on the index values(idx)

    """

    def __getitem__(self, idx):

        return to_tensor([self.features[idx], self.targets[idx]])

        

        
# We'll split our data into 90% training and 10% test data 

split = int(0.9 * len(train))

valid_data = train[split:].reset_index(drop=True)

train_data = train[:split].reset_index(drop=True)

valid_data.head()
# Getting features of the image (pixel 0-783, 784 pixels in total for a 28*28 image)

X_col = list(train.columns[1:])

y_col = "label"



train_set = MNISTData(train_data, X_col, y_col)

valid_set = MNISTData(valid_data, X_col, y_col)



train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

valid_loader = DataLoader(valid_set, batch_size=64, shuffle=True)



for data in train_loader:

    X_train, y_train = data

    fig = plt.figure()

    plt.imshow(X_train[0].reshape(28, 28), cmap='gray')

    break
class dummyModel(nn.Module):

    def __init__(self):

        super(dummyModel, self).__init__()

        

        """

        In layer 1, which is a convolutional layer, input channels will be 1, 

        since our data consists of graycale images.

        

        out_channels is the number of filters, of size 3(kernel_size). 

        out_channels can be whatever we want, so I'm gonna select 4.

        

        stride is the number of steps that a filter jumps, from it's previous position.

        

        padding is the number of pixels added to both sides of the input data(image), for each dimension.

        Padding mode is the value of the padding pixels.

        """

        self.layer1 = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        

        #Layer 2

        self.layer2 = nn.Sequential(

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()

        

        #Layer 3

        # FC layer

        

        

    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        return x

        

        
network = dummyModel()

print(network)

num_epochs = 5

for epoch in range(num_epochs):

    for train_batch in train_loader:

        train_X, train_y = train_batch

        # train_X = train_X.reshape(len(train_X), 1, 28, 28)

        outputs = network.forward(train_X)

        # To check dimensions of our tensor

        op = outputs.flatten(start_dim=1, end_dim=-1)

        break

print(outputs.shape)

print(op.shape)

        

        

        
class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.BatchNorm2d(4))



        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.BatchNorm2d(8))



        self.fc1 = nn.Sequential(

            nn.Linear(392, 128),

            nn.Dropout(0.5),

            nn.ReLU(),

            nn.BatchNorm1d(128))



        self.fc2 = nn.Sequential(

            nn.Linear(128, 10))

            

    def forward(self, inputs):

        x = self.conv1(inputs)

        # print(f"After conv1, dimensions are: {x.shape}")

        x = self.conv2(x)

        # print(f"After conv2, dimensions are: {x.shape}")

        # Flattening the image

        x = x.flatten(start_dim=1, end_dim=-1)

        x = self.fc1(x)

        # print(f"After fc1, dimensions are: {x.shape}")

        x = self.fc2(x)

        x = nn.LogSoftmax(dim=1)(x)

        # print(f"After fc2, dimensions are: {x.shape}")

        return x
# Before we start training our neural network, we'll define our accuracy function

def acc(y_true, y_pred):

    y_true = y_true.long().squeeze()

    y_pred = torch.argmax(y_pred, axis=1)

    return (y_true==y_pred).float().sum()/len(y_true)
network = model()

# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(network.parameters())

loss = []

accuracy = []

num_epochs = 5

for i in range(1, num_epochs+1):

    for data in train_loader:

        batch_X, batch_Y = data

        batch_prediction = network.forward(batch_X)

        #print(batch_prediction[0])

        

        batch_Y = batch_Y.long().squeeze()

        #print(f"Y_Pred Dimensions: {batch_Y.shape}\tPrediction Dimensions:{batch_prediction.shape}")

        batch_loss = criterion(batch_prediction, batch_Y)

        batch_acc = acc(batch_Y, batch_prediction)

    

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

        

        loss.append(batch_loss)

        accuracy.append(batch_acc)

    with torch.no_grad():

        for valid_batch in valid_loader:

            valid_X, valid_Y = valid_batch

            valid_Y = valid_Y.long().squeeze()

            valid_prediction = network.forward(valid_X)

            valid_loss = criterion(valid_prediction, valid_Y)

            valid_acc = acc(valid_Y, valid_prediction)

    print("----------------------------------------------------------------------------------------------")  

    print(f"Yo, this is epoch number {i}")

    print(f"Training Accuracy: {batch_acc:.4f}\tTraining Loss:{batch_loss:.4f}\nValidation Accuracy: {valid_acc:.4f}\tValidation Loss: {valid_loss:.4f}")
# Gotta prepare the test set first

test[y_col] = [-1]*len(test)



test_set = MNISTData(test, X_col, y_col)

test_loader = tqdm(DataLoader(test_set, batch_size=1024, shuffle=False))

test_pred = []

with torch.no_grad():

    for test_X, _ in test_loader:

        pred = network.forward(test_X)

        test_pred.extend(np.argmax(pred, axis=1))
test_X, _ = next(iter(test_loader))

test_X = test_X[:36]

fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(15, 15))



for i, image in enumerate(test_X):

    image = image.reshape(28, 28)

    ax[i//6][i%6].axis('off')

    ax[i//6][i%6].imshow(image, cmap='gray')

    ax[i//6][i%6].set_title(test_pred[i].item(), fontsize=20, color='blue')
submissions = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

labels = [x.item() for x in test_pred]

submissions["Label"] = labels

submissions.head()
submissions.to_csv("submission.csv", index=False)