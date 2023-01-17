import pandas as pd

import numpy as np

from skimage.io import imread

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm import tqdm



# PyTorch libraries and modules

import torch

from torch.autograd import Variable

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, BCELoss

from torch.optim import Adam, SGD
train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

test  = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

test_path = '../input/siim-isic-melanoma-classification/jpeg/test/'

train_path = '../input/siim-isic-melanoma-classification/jpeg/train/'

sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
# loading training images



#train_img = []

#for img_name in tqdm(train['image_name']):

    # defining the image path

   # image_path = train_path + str(img_name) + '.jpg'

    # reading the image

  #  img = imread(image_path, as_gray=True)

    # normalizing the pixel values

  #  img /= 255.0

    # converting the type of pixel to float 32

  #  img = img.astype('float32')

    # appending the image into the list

 #   train_img.append(img)



# converting the list to numpy array

#train_x = np.array(train_img)



# defining the target

train_y = train['target'].values

train_x = np.load(("../input/x-train/x_train_32.npy"))

train_x.shape
# visualizing images



i = 0



plt.figure(figsize=(10,10))

plt.subplot(221), plt.imshow(train_x[i])

plt.subplot(222), plt.imshow(train_x[i+25])

plt.subplot(223), plt.imshow(train_x[i+50])

plt.subplot(224), plt.imshow(train_x[i+75])
# create validation set



train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)

(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
train_x = train_x.reshape(26500, 3, 32, 32)

train_x  = torch.from_numpy(train_x)



# converting the target into torch format

train_y = train_y.astype(int);

train_y = torch.from_numpy(train_y)



# shape of training data

train_x.shape, train_y.shape
val_x = val_x.reshape(6626, 3, 32, 32)



val_x  = torch.from_numpy(val_x)



# converting the target into torch format

val_y = val_y.astype(int);

val_y = torch.from_numpy(val_y)



# shape of validation data

val_x.shape, val_y.shape
## Setting the seed



np.random.seed(42)

torch.manual_seed(42)
import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

 

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 4)

        self.conv2 = nn.Conv2d(6, 16, 3)

        self.adapt = nn.AdaptiveMaxPool2d((5,7))

        self.fc1 = nn.Linear(16*5*7, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Sequential(nn.Linear(84, 2))

                



    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x.float())), (2, 2))

        x = self.adapt(F.relu(self.conv2(x.float())))

        x = x.view(-1, 16*5*7)

        x = F.relu(self.fc1(x.float()))

        x = F.relu(self.fc2(x.float()))

        x = self.fc3(x.float())

        return x
model = Net()

# defining the optimizer

optimizer = Adam(model.parameters(), lr=0.07)

# defining the loss function

criterion = CrossEntropyLoss() # nn.BCEWithLogitsLoss()

# checking if GPU is available

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()

    

print(model)
def train(epoch):

    model.train()

    tr_loss = 0

    

    # getting the training set

    x_train, y_train = Variable(train_x), Variable(train_y)

    

    # getting the validation set

    x_val, y_val = Variable(val_x), Variable(val_y)

    

    # converting the data into GPU format

    if torch.cuda.is_available():

        x_train = x_train.cuda()

        y_train = y_train.cuda()

        x_val = x_val.cuda()

        y_val = y_val.cuda()



    # clearing the Gradients of the model parameters

    optimizer.zero_grad()

    

    output_train = model(x_train)

    output_val = model(x_val)



    # computing the training and validation loss

    

    loss_train = criterion(output_train, y_train)

    loss_val = criterion(output_val, y_val)

    train_losses.append(loss_train)

    val_losses.append(loss_val)



    # computing the updated weights of all the model parameters

    

    loss_train.backward()

    optimizer.step()

    tr_loss = loss_train.item()

    if epoch%2 == 0:

        # printing the validation loss

        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
# defining the number of epochs



n_epochs = 11

# empty list to store training losses

train_losses = []

# empty list to store validation losses

val_losses = []

# training the model

#for epoch in range(n_epochs):

 #   train(epoch)

    

# plotting the training and validation loss



#plt.plot(train_losses, label='Training loss')

#plt.plot(val_losses, label='Validation loss')

#plt.legend()

#plt.show()
#with torch.no_grad():

output = model(train_x)

    

softmax = torch.exp(output).cpu()

prob = list(softmax.detach().numpy())

predictions = np.argmax(prob, axis=1)



# accuracy on training set

print(accuracy_score(train_y, predictions))

print(roc_auc_score(train_y, predictions))
output = model(val_x)



softmax = torch.exp(output).cpu()

prob = list(softmax.detach().numpy())

predictions = np.argmax(prob, axis=1)



# accuracy on validation set

print(accuracy_score(val_y, predictions))

print(roc_auc_score(val_y, predictions))
# loading test images



test_img = []



#for img_name in tqdm(test['image_name']):

    # defining the image path

 #   image_path = test_path + str(img_name) + '.jpg'

    # reading the image

  #  img = imread(image_path, as_gray=True)

    # normalizing the pixel values

   # img /= 255.0

    # converting the type of pixel to float 32

    #img = img.astype('float32')

    # appending the image into the list

    #test_img.append(img)



# converting the list to numpy array

#test_x = np.array(test_img)



test_x = np.load(("../input/x-test-32/x_test_32.npy"))

test_x.shape
# converting test images into torch format



test_x = test_x.reshape(10982, 3, 32, 32)

test_x  = torch.from_numpy(test_x)

test_x.shape
# generating predictions for test set



output = model(test_x)



#softmax = torch.exp(output).cpu()

#prob = list(softmax.detach().numpy())

#predictions = np.argmax(prob, axis=1)



preds = F.softmax(output)

preds = preds[:, 0]

preds = preds.detach().numpy()

sample_submission['target'] = preds

sample_submission.to_csv('sub_05.csv', index=False)