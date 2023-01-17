import os

import cv2

import numpy as np

from tqdm import tqdm

import torch 

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
# False skips data pre-processing and training data can be loaded in directly

REBUILD_DATA = True



class CatsVSDogs():

    

    # Size of the image

    IMG_SIZE = 50

    

    # Directory location

    CATS = '../input/kaggle-cat-vs-dog-dataset/kagglecatsanddogs_3367a/PetImages/Cat/'

    DOGS = '../input/kaggle-cat-vs-dog-dataset/kagglecatsanddogs_3367a/PetImages/Dog/'

    

    # Labels for cats and dogs

    LABELS = {CATS:0, DOGS:1}

    

    # Initializing variables

    training_data = []

    catcount = 0

    dogcount = 0

    

    def make_training_data(self):

        for label in self.LABELS:

            

            # Looping through each pictures

            for f in tqdm(os.listdir(label)):

                

                try:

                    path = os.path.join(label, f)



                    # Reading images and converting to grayscale

                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)



                    # Resizing images

                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))



                    # Getting the training data

                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])



                    # Checking distribution of data

                    if label == self.CATS:

                        self.catcount += 1

                    elif label == self.DOGS:

                        self.dogcount += 1

                        

                except Exception as e:

                    pass



            np.random.shuffle(self.training_data)

            np.save("training_data.npy", self.training_data)

            print("Cates: ", self.catcount)

            print("Dogs: ", self.dogcount)

            

if REBUILD_DATA:

    catsvdogs = CatsVSDogs()

    catsvdogs.make_training_data()
training_data = np.load("training_data.npy", allow_pickle = True)
import matplotlib.pyplot as plt

%matplotlib inline



plt.imshow(training_data[1][0], cmap = "gray")
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)

        self.conv2 = nn.Conv2d(32, 64, 5)

        self.conv3 = nn.Conv2d(64, 128, 5)

        

        # Getting the output size for the fully connected layer

        x = torch.randn(50,50).view(-1, 1, 50, 50)

        self._to_linear = None

        self.convs(x)

        

        self.fc1 = nn.Linear(self._to_linear, 512)

        self.fc2 = nn.Linear(512, 2)



    def convs(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        

        if self._to_linear is None:

            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

            

        return x

    

    def forward(self, x):

        x = self.convs(x)

        x = x.view(-1, self._to_linear)

        x = F.relu(self.fc1(x))

        x = F.softmax(self.fc2(x), dim=1)

        return x



net = Net()
# Using Adam optimizer to optimize weights of the Neural Network

optimizer = optim.Adam(net.parameters(), lr = 0.001)



# Initializing loss function

loss_function = nn.MSELoss()



# Getting the features

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)



# Scaling the features

X = X/255.0



# Getting the target

y = torch.Tensor([i[1] for i in training_data])



# Statistics to create testing set

VAL_PCT = 0.1

val_size = int(len(X)*VAL_PCT)

print(val_size)

# Train test split



train_X = X[:-val_size]

train_y = y[:-val_size]



test_X = X[-val_size:]

test_y = y[-val_size:]
# Training the model in batches of 100

BATCH_SIZE = 100



# Number of Epochs

EPOCHS = 1



# Model Training

for epoch in range(EPOCHS):

    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):

        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)

        batch_y = train_y[i:i+BATCH_SIZE]

        

        net.zero_grad()

        outputs = net(batch_X)

        loss = loss_function(outputs, batch_y)

        loss.backward()

        optimizer.step()

        

print(loss)     
# Calculating Accuracy



correct = 0

total = 0



with torch.no_grad():

     for i in tqdm(range(len(test_X))):

            real_class = torch.argmax(test_y[i])

            net_out = net(test_X[i].view(-1,1,50,50))[0]

            predicted_class = torch.argmax(net_out)

            

            if predicted_class == real_class:

                correct += 1

            total +=1

                

print("Accuracy: ", round(correct/total, 3))