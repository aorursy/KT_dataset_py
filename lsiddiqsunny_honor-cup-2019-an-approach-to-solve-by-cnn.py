# Import necessary packages

%matplotlib inline

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm



import torch

import torchvision

import torchvision.transforms as transforms

torch.manual_seed(42)
train_image = []

for i in range(1200,1800):

    img = image.load_img('../input/shuffled-data-to-train-honor-cup-2019/data_train/64/'+str(i).zfill(4)+'.png', target_size=(512,512,1), grayscale=False)

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

x = np.array(train_image)

X = torch.from_numpy(x)

print(X.shape)

plt.imshow(X[0])
# Open file    

fileHandler = open ("../input/shuffled-data-to-train-honor-cup-2019/data_train/data_train_64_answers.txt", "r")

 

# Get list of all lines in file

listOfLines = fileHandler.readlines()

 

# Close file 

fileHandler.close()
for line in listOfLines[:6]:

    print(line.strip()) 
i=0

train_answer=[]

for line in listOfLines:

    i=i+1

    if i%2==1:

        continue

    string_line=line.strip().split() 

    answer_line=[]

    for num in string_line:

        answer_line.append(int(num))

    train_answer.append(answer_line)

y=np.array(train_answer)

Y=torch.from_numpy(y)

print(Y.shape)
def restore_image(p_images,permutation):

    images=p_images.clone()

    for i in range(8):

        for j in range(8):

            sr=i*64

            sc=j*64

            #print('{:d} {:d}'.format(sr,sc))

            tr=permutation[i*8+j]//8

            tc=permutation[i*8+j]%8

            tc=tc*64

            tr=tr*64

            #print('{:d} {:d}'.format(tr,tc))

            #sr, sc = perm_inds[j]

            #tr, tc = perm_inds[perms[i, j]]

            images[ sr:sr+64, sc:sc+64,:] = p_images[ tr:tr+64, tc:tc+64,:]

    return images
r_image=restore_image(X[0],Y[0])

plt.imshow(r_image)
def perm2vecmat2x2(perms):

    """

    Converts permutation vectors to vectorized assignment matrices.

    """

    n = perms.size()[0]

    mat = torch.zeros(n, 64, 64)

    # m[i][j] : i is assigned to j

    for i in range(n):

        for k in range(64):

            mat[i, k, perms[i, k]] = 1.

    return mat.view(n, -1)

def vecmat2perm2x2(x):

    """

    Converts vectorized assignment matrices back to permutation vectors.

    Note: this function is compatible with GPU tensors.

    """

    n = x.size()[0]

    x = x.view(n, 64, 64)

    _, ind = x.max(2)

    return ind

    
train_set = torch.utils.data.TensorDataset(X, Y)

sample_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)

dataiter = iter(sample_loader)

images, labels = next(dataiter)

z=perm2vecmat2x2(labels)

vecmat2perm2x2(z)

#r_image=restore_image(images[0],labels[0])

#plt.imshow(r_image)
# Prepare training, validation, and test samples.



validation_ratio = 0.1

total = len(train_set)

ind = list(range(total))

n_train = int(np.floor((1. - validation_ratio) * total))

train_ind, validation_ind = ind[:n_train], ind[n_train:]

train_subsampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)

validation_subsampler = torch.utils.data.sampler.SubsetRandomSampler(validation_ind)



train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,

                                           sampler=train_subsampler, num_workers=0)

validation_loader = torch.utils.data.DataLoader(train_set, batch_size=8,

                                                sampler=validation_subsampler, num_workers=0)



print('Number of training batches: {}'.format(len(train_loader)))

print('Number of validation batches: {}'.format(len(validation_loader)))

from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



def sinkhorn(A, n_iter=4):

    """

    Sinkhorn iterations.

    """

    for i in range(n_iter):

        A /= A.sum(dim=1, keepdim=True)

        A /= A.sum(dim=2, keepdim=True)

    return A



class SimpleConvNet(nn.Module):

    """

    A simple convolutional neural network shared among all pieces.

    """

    def __init__(self):

        super().__init__()

         # 3 x 64 x 64 input

        self.conv1 = nn.Conv2d(3, 8, 3)

        # 8 x 62 x 62

        self.conv2 = nn.Conv2d(8, 8, 3)

        self.conv2_bn = nn.BatchNorm2d(8)

        # 8 x 60 x 60

        self.pool1 = nn.MaxPool2d(2, 2)

        # 8 x 30 x 30

        self.conv3 = nn.Conv2d(8, 16, 3)

        self.conv3_bn = nn.BatchNorm2d(16)

        # 16 x 28 x 28

        self.fc1 = nn.Linear(16*28*28, 512)

        self.fc1_bn = nn.BatchNorm1d(512)

        # 128-d features

        self.fc2 = nn.Linear(512, 512)

        self.fc2_bn = nn.BatchNorm1d(512)

    

    def forward(self, x):

        #print(x.shape)

        x = F.relu(self.conv1(x))

        #print(x.shape)

        x = F.relu(self.conv2_bn(self.conv2(x)))

        #print(x.shape)

        x = self.pool1(x)

        #print(x.shape)

        x = F.relu(self.conv3_bn(self.conv3(x)))

        #print(x.shape)

        x = x.reshape(-1, 16*28*28)

        #print(x.shape)

        x = F.relu(self.fc1_bn(self.fc1(x)))

        #print(x.shape)

        x = F.relu(self.fc2_bn(self.fc2(x)))

        #print(x.shape)

        return x



class JigsawNet(nn.Module):

    """

    A neural network that solves 64x64 jigsaw puzzles.

    """

    def __init__(self, sinkhorn_iter=0):

        super().__init__()

        self.conv_net = SimpleConvNet()

        self.fc1 = nn.Linear(512*64, 8192)

        self.fc1_bn = nn.BatchNorm1d(8192)

        # 4 x 4 assigment matrix

        self.fc2 = nn.Linear(8192, 4096)

        self.sinkhorn_iter = sinkhorn_iter

    

    def forward(self, x):

        # Split the input into four pieces and pass them into the

        # same convolutional neural network.

        piece=[]

        for i in range (8):

            for j in range(8):

                one=self.conv_net(x[:,:, i*64:i*64+64, i*64:i*64+64])

                piece.append(one)

        # Cat

        x = torch.cat(piece, dim=1)

        #print(x.shape)

        # Dense layer

        x = F.dropout(x, p=0.1, training=self.training)

        #print(x.shape)

        x = F.relu(self.fc1_bn(self.fc1(x)))

        #print(x.shape)

        x = self.fc2(x)

        #print(x.shape)

        if self.sinkhorn_iter > 0:

            x = x.view(-1, 64, 64)

            x = sinkhorn(x, self.sinkhorn_iter)

            x = x.view(-1, 4096)

        return x
# Test helper

def compute_acc(p_pred, p_true, average=True):

    """

    We require that the location of all four pieces are correctly predicted.

    Note: this function is compatible with GPU tensors.

    """

    # Remember to cast to float.

    n = torch.sum((torch.sum(p_pred == p_true, 1) == 64).float())

    if average:

        return n / p_pred.size()[0]

    else:

        return n



# Training process

def train_model(model, criterion, optimizer, train_loader, validation_loader,

                n_epochs=40, save_file_name=None):

    loss_history = []

    val_loss_history = []

    acc_history = []

    val_acc_history = []

    for epoch in range(n_epochs):

        with tqdm(total=len(train_loader), desc="Epoch {}".format(epoch + 1), unit='b', leave=False) as pbar:

            # Training phase

            model.train()

            running_loss = 0.

            n_correct_pred = 0

            n_samples = 0

            for i, data in enumerate(train_loader, 0):

                inputs, perms = data

                x_in=inputs

                x_in=x_in.permute(0,3,1,2)

                

                #print(x_in.shape)

                y_in = perm2vecmat2x2(perms)

                

                n_samples += inputs.size()[0]

                if is_cuda_available:

                    x_in, y_in = Variable(x_in.cuda()), Variable(y_in.cuda())

                    perms = Variable(perms.cuda())

                else:

                    x_in, y_in = Variable(x_in), Variable(y_in)

                    perms = Variable(perms)

                optimizer.zero_grad()

                outputs = model(x_in)

                n_correct_pred += compute_acc(vecmat2perm2x2(outputs), perms, False)

                loss = criterion(outputs, y_in)

                print(loss)

                loss.backward()

                optimizer.step()

                running_loss += loss * x_in.size()[0]

                pbar.update(1)

            loss_history.append(running_loss / n_samples)

            acc_history.append(n_correct_pred / n_samples)

            

            # Validation phase

            model.eval()

            running_loss = 0.

            n_correct_pred = 0

            n_samples = 0

            for i, data in enumerate(validation_loader, 0):

                inputs, perms = data

                x_in=inputs

                x_in=x_in.permute(0,3,1,2)

                

                #print(x_in.shape)

                y_in = perm2vecmat2x2(perms)

                n_samples += inputs.size()[0]

                if is_cuda_available:

                    x_in, y_in = Variable(x_in.cuda()), Variable(y_in.cuda())

                    perms = Variable(perms.cuda())

                else:

                    x_in, y_in = Variable(x_in), Variable(y_in)

                    perms = Variable(perms)

                outputs = model(x_in)

                n_correct_pred += compute_acc(vecmat2perm2x2(outputs), perms, False)

                loss = criterion(outputs, y_in)

                running_loss += loss * x_in.size()[0]

            val_loss_history.append(running_loss / n_samples)

            val_acc_history.append(n_correct_pred / n_samples)

            

            # Update the progress bar.

            print("Epoch {0:03d}: loss={1:.4f}, val_loss={2:.4f}, acc={3:.2%}, val_acc={4:.2%}".format(

                epoch + 1, loss_history[-1], val_loss_history[-1], acc_history[-1], val_acc_history[-1]))

    print('Training completed')

    history = {

        'loss': loss_history,

        'val_loss': val_loss_history,

        'acc': acc_history,

        'val_acc': val_acc_history

    }

    # Save the model when requested.

    if save_file_name is not None:

        torch.save({

            'history': history,

            'model': model.state_dict(),

            'optimizer': optimizer.state_dict()

        }, save_file_name)

    return history



# Test process

# Compute the accuracy

def test_model(model, test_loader):

    running_acc = 0.

    n = 0

    model.eval()

    for i, data in enumerate(test_loader, 0):

        inputs, perms = data

        x_in=inputs

        x_in=x_in.permute(0,3,1,2)

                

        #print(x_in.shape)

        y_in = perm2vecmat2x2(perms)

        if is_cuda_available:

            x_in, y_in = Variable(x_in.cuda()), Variable(y_in.cuda())

        else:

            x_in, y_in = Variable(x_in), Variable(y_in)

        pred = model(x_in)

        perms_pred = vecmat2perm2x2(pred.cpu().data)

        running_acc += compute_acc(perms_pred, perms, False)

        n += x_in.size()[0]

    acc = running_acc / n

    return acc

n_epochs = 2

sinkhorn_iter = 4



# Create the neural network.

model = JigsawNet(sinkhorn_iter=sinkhorn_iter)

is_cuda_available = torch.cuda.is_available();

if is_cuda_available:

    model.cuda()



n_params = 0

for p in model.parameters():

    n_params += np.prod(p.size())

print('# of parameters: {}'.format(n_params))



# We use binary cross-entropy loss here.

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters())



# Train



history = train_model(model, criterion, optimizer, train_loader, validation_loader,

                      n_epochs=n_epochs)

model_save_name = 'classifier.pt'

torch.save(model.state_dict(), model_save_name)
plt.figure()

plt.plot(history['loss'])

plt.plot(history['val_loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

plt.show()

plt.figure()

plt.plot(history['acc'])

plt.plot(history['val_acc'])

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()
# Calculate accuracy

print('Training accuracy: {}'.format(test_model(model, train_loader)))

print('Validation accuracy: {}'.format(test_model(model, validation_loader)))

# Here training accuracy will be higher because dropout is disabled
