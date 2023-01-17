import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from scipy.ndimage.interpolation import shift

from skimage.transform import rotate, AffineTransform, warp

import matplotlib.pyplot as plt



import random

import time

import io

import os



deviceCount = torch.cuda.device_count()

print(deviceCount)



cuda0 = None

if deviceCount > 0:

  print(torch.cuda.get_device_name(0))

  cuda0 = torch.device('cuda:0')
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



print(df.shape)

dataset_size = df.shape[0]



df.head()
x = df.drop('label', axis=1)

x = x.values.reshape(dataset_size, 1, 28, 28)



y = df['label'].values.reshape(dataset_size)



df = None
cross_validation_ratio = 0.05

# cross_validation_ratio = 0.2



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cross_validation_ratio, random_state=93)



x = None

y = None
start_time = time.time()



def shift_image(image, dx, dy):

    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")

    return shifted_image



augmented_x = []

augmented_y = []



random.seed(103)



percentage_processed = 0

for i in range(len(x_train)):

    

    augmented_x.append(x_train[i].astype(int))

    augmented_y.append(y_train[i].astype(int))

    

    for j in range(5):

        image = x_train[i].reshape((28, 28))

        

        x_shift = random.randint(-1, 1)

        y_shift = random.randint(-1, 1)

        rotation_deg = float(random.randint(-15, 15))

        

        image = rotate(image, angle=rotation_deg, cval=0, mode="constant", preserve_range=True)

        image = np.rint(image)

        image = image.astype(int)

        

        image = shift_image(image, x_shift, y_shift)

        

#         magnification = random.uniform(0.98, 1.02)

#         shear = random.uniform(-0.1, 0.1)

        

#         transformation = AffineTransform(shear=shear, scale=(magnification, magnification))

#         image = warp(image, transformation.inverse, preserve_range=True)

#         image = np.rint(image)

#         image = image.astype(int)



        image = x_train[i].reshape((1, 28, 28))



        augmented_x.append(image)

        augmented_y.append(y_train[i].astype(int))

        

    if (i+1)/len(x_train)*100 >= percentage_processed+10:

        print( f'{int((i+1)/len(x_train)*100):3}% images processed')

        percentage_processed += 10

        

train_dataset_size = len(augmented_x)

test_dataset_size = len(x_test)



x_train = None

y_train = None



print(f'\nDuration: {time.time() - start_time:.0f} seconds')
random.seed(93)



for i in range(len(augmented_x)):

    

    index = random.randint(0, len(augmented_x)-1)

    

    tempx = augmented_x[i]

    tempy = augmented_y[i]

    

    augmented_x[i] = augmented_x[index]

    augmented_y[i] = augmented_y[index]

    

    augmented_x[index] = tempx

    augmented_y[index] = tempy
augmented_x = np.array(augmented_x)

augmented_y = np.array(augmented_y)



x_train = torch.FloatTensor(augmented_x)

x_test = torch.FloatTensor(x_test)



y_train = torch.LongTensor(augmented_y)

y_test = torch.LongTensor(y_test)



if cuda0 != None:

  x_train = x_train.cuda()

  x_test = x_test.cuda()

  y_train = y_train.cuda()

  y_test = y_test.cuda()
trainingDataset = TensorDataset(x_train, y_train)

testDataset = TensorDataset(x_test, y_test)



trainloader = DataLoader(trainingDataset, batch_size=512, shuffle=True)



testloader = DataLoader(testDataset, batch_size=512, shuffle=False)
class ConvolutionalNetwork(nn.Module):

    def __init__(self):

        super().__init__()                          

        self.conv1 = nn.Conv2d(1, 16, 3, 1)

        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        self.fc1 = nn.Linear(5*5*32, 140)

        self.fc2 = nn.Linear(140, 80)

        self.fc3 = nn.Linear(80,10)  

        self.dropout1 = nn.Dropout(p=0.5)

        self.dropout2 = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(140)

        self.bn2 = nn.BatchNorm1d(80)

        self.conv1_bn = nn.BatchNorm2d(16)

        self.conv2_bn = nn.BatchNorm2d(32)



    def forward(self, X):

        X = self.conv1(X)

        X = self.conv1_bn(X)

        X = F.relu(X)

        X = F.max_pool2d(X, 2, 2)

        X = self.conv2(X)

        X = self.conv2_bn(X)

        X = F.relu(X)

        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, 5*5*32)

        X = self.fc1(X)

        X = self.bn1(X)

        X = F.relu(X)

        X = self.dropout1(X)

        X = self.fc2(X)

        X = self.bn2(X)

        X = F.relu(X)

        X = self.dropout2(X)

        X = self.fc3(X)

        return X
torch.manual_seed(103)

torch.cuda.manual_seed(103)



model = ConvolutionalNetwork()



if cuda0 != None:

  model.to(cuda0)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



print(model)
import time

start_time = time.time()



epochs = 45

train_losses = []

test_losses = []

train_correct = []

test_correct = []



for i in range(epochs):



    model.train()



    epoch_start_time = time.time()

    trn_corr = 0

    tst_corr = 0

    total = 0

    currentLoss = 0

    

    for currentBatch in enumerate(trainloader):

        bno = currentBatch[0] + 1

        x = currentBatch[1][0]

        y = currentBatch[1][1]

        

        y_pred = model(x)

        loss = criterion(y_pred, y)

        

        lambdaParam = torch.tensor(0.05)

        l2_reg = torch.tensor(0.)

        if cuda0 != None:

          lambdaParam = lambdaParam.cuda()

          l2_reg = l2_reg.cuda() 



        for param in model.parameters():

          if cuda0 != None:

            l2_reg += torch.norm(param).cuda()

          else:

            l2_reg += torch.norm(param)



        loss += lambdaParam * l2_reg

        

        y_pred = F.log_softmax(y_pred, dim=1)

        predicted = torch.max(y_pred.data, 1)[1]

        batch_corr = (predicted == y).sum()

        trn_corr += batch_corr

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        currentLoss += loss.item()

   

        total += len(currentBatch[1][1])



        if bno%100 == 0 or bno==1:

            printStr = f'epoch: {i+1} batch: {bno:3} loss: {loss.item():10.8f} accuracy: {trn_corr.item()/total*100:6.3f}%'

            print(printStr)

            



    train_losses.append(currentLoss/bno)

    train_correct.append(trn_corr.item())

    

    currentLoss = 0

    

    model.eval()

    with torch.no_grad():

        for currentBatch in enumerate(testloader):

            bno = currentBatch[0] + 1

            x = currentBatch[1][0]

            y = currentBatch[1][1]



            y_pred = model(x)



            predicted = torch.max(y_pred.data, 1)[1] 

            tst_corr += (predicted == y).sum()

            

            loss = criterion(y_pred, y)

            currentLoss += loss.item()

            

    test_losses.append(currentLoss/bno)

    test_correct.append(tst_corr.item())



    print('Summary of Epoch {}:'.format(i+1))

    print(f'Train Loss: {train_losses[i]:10.8f}  Train Accuracy: {train_correct[i]/train_dataset_size*100:6.3f}%')

    print(f'Test Loss: {test_losses[i]:10.8f}  Test Accuracy: {test_correct[i]/test_dataset_size*100:6.3f}%')

    print(f'Epoch Duration: {time.time() - epoch_start_time:.0f} seconds')

    print('')



    scheduler.step()

    

print(f'\nDuration: {time.time() - start_time:.0f} seconds')
x = [i+1 for i in range(len(train_losses))]

plt.plot(x, train_losses, label='training loss')

plt.plot(x, test_losses, label='validation loss')

plt.title('Loss for each epoch')

plt.legend();

plt.show()



train_accuracy = [i/train_dataset_size for i in train_correct]

test_accuracy = [i/test_dataset_size for i in test_correct]



plt.plot(x, train_accuracy, label='training accuracy')

plt.plot(x, test_accuracy, label='validation accuracy')

plt.title('Accuracy for each epoch')

plt.legend();

plt.show()
offset = 1



plt.plot(x[offset-1:], train_losses[offset-1:], label='training loss')

plt.plot(x[offset-1:], test_losses[offset-1:], label='validation loss')

plt.title('Loss for each epoch')

plt.legend();

plt.show();



train_accuracy = [i/train_dataset_size for i in train_correct]

test_accuracy = [i/test_dataset_size for i in test_correct]



plt.plot(x[offset-1:], train_accuracy[offset-1:], label='training accuracy')

plt.plot(x[offset-1:], test_accuracy[offset-1:], label='validation accuracy')

plt.title('Accuracy for each epoch')

plt.legend();

plt.show()
x_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

x_test = x_test.values.reshape(28000, 1, 28, 28)

x_test = torch.FloatTensor(x_test)



if cuda0 != None:

    x_test = x_test.cuda()
with torch.no_grad():

    model.eval()

    y_pred = model(x_test)

    predictions = torch.max(y_pred.data, 1)[1] 

    

predictions = predictions.cpu().detach().numpy()

ids = [id+1 for id in range(len(predictions))]

output = pd.DataFrame({'ImageId': ids, 'Label': predictions})
output.to_csv('/kaggle/working/my_submission.csv', index=False)