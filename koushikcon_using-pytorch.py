# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import cv2

from glob import glob

import torchvision

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import torchvision.models as models

import torchvision.transforms as transforms

from PIL import Image, ImageFile

from torch.utils.data import TensorDataset, DataLoader,Dataset

from sklearn.model_selection import train_test_split

import copy 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

#print(os.listdir("../input"))

#../input/train/train

#../input/train/test



# Any results you write to the current directory are saved as output.

use_cuda = torch.cuda.is_available()

if not use_cuda:

    print('No GPU found. Please use a GPU to train your neural network.')
label_frame = pd.read_csv('../input/aerial-cactus-identification/train.csv')

test_frame = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

# VGG16 = models.vgg16(pretrained=True)

# train_data = torchvision.datasets.ImageFolder('../input/train/train/', transform=transform)

# valid_data = torchvision.datasets.ImageFolder('/data/dog_images/valid/', transform=transform)

# test_data = torchvision.datasets.ImageFolder('../input/test/test/', transform=transform_test)

# NOTE: class is inherited from Dataset

class ImageLabelDataset(Dataset):

    def __init__(self, df_data, prediction, folder="../input"):

        super().__init__()

        self.df = df_data.values

        self.prediction = prediction.values

        self.folder = folder



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        tensorimage = self.preprocess_image(self.df[index])

        # print(label)

        label = self.prediction[index]

        # print(label)

        return tensorimage, label

    

    def preprocess_image(self, img_path):

        data_transform = transforms.Compose([transforms.ToPILImage(),

                                             transforms.Resize(224), 

                                             transforms.CenterCrop(224), 

                                             transforms.RandomRotation(30), 

                                             transforms.ToTensor()

                                            ])

        image = cv2.imread("{}/{}".format(self.folder, img_path))

        image = data_transform(image)

        return image
def preprocess_image(img_path):

    '''

    Use pre-trained VGG-16 model to obtain index corresponding to 

    predicted ImageNet class for image at specified path

    

    Args:

        img_path: path to an image

        

    Returns:

        Index corresponding to VGG-16 model's prediction

    '''

    data_transform = transforms.Compose([transforms.ToPILImage(),

                                         transforms.CenterCrop(224), 

                                         transforms.RandomRotation(30), 

                                         transforms.ToTensor(),

                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

                                        ])

    # train_data = datasets.ImageFolder(train_dir, transform=data_transform)

    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # print("../input/train/train/{}".format(img_path))

    image = cv2.imread("../input/train/train/{}".format(img_path))

    # image = Image.open("../input/train/train/{}".format(img_path))

    image = data_transform(image)

    # image = image.clone().detach().requires_grad_(True)

    # image = torch.tensor(image, requires_grad=True, dtype=torch.float)

    # image = image.unsqueeze(0)

    # if use_cuda:

        # image = image.cuda()    

    

    return image # predicted class index
# label_frame = label_frame[:20]

# label_frame['tensorimage']=label_frame['id'].apply(preprocess_image)

# label_frame['tensorimage']=label_frame['tensorimage'].astype(float)

print(label_frame.dtypes)
# label_frame.head()
training_set = label_frame["id"]

prediction_set = label_frame.has_cactus

batch_size = 1



test_set = test_frame["id"]

test_prediction_set = test_frame.has_cactus



X_train, X_val, Y_train, Y_val = train_test_split(training_set, prediction_set, test_size=0.1, random_state=42)

print(len(X_train))

print(len(Y_train))



print(len(X_val))

print(len(Y_val))



# train_set = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=0)

# train_set = torch.utils.data.TensorDataset(torch.FloatTensor(torch.from_numpy(X_train.values)), torch.FloatTensor(y_train.values))

# val_set = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=True, num_workers=0)



train_set = ImageLabelDataset(df_data=X_train, prediction=Y_train, folder="../input/aerial-cactus-identification/train/train")

val_set = ImageLabelDataset(df_data=X_val, prediction=Y_val, folder="../input/aerial-cactus-identification/train/train")

predict_set = ImageLabelDataset(df_data=test_set, prediction=test_prediction_set, folder="../input/aerial-cactus-identification/test/test")



train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(predict_set, batch_size=1, shuffle=True, num_workers=0)
# Hyperparameters

batch_no = len(X_train) // batch_size  #batches

# cols=X_train.shape[1] #Number of columns in input matrix

n_output=1



# Sequence Length

sequence_length = 6  # of words in a sequence 892110

# Batch Size

# batch_size = 128

# train_loader = batch_data(int_text, sequence_length, batch_size)

# Number of Epochs

num_epochs = 20

# Learning Rate

learning_rate = 0.0002

# Model parameters

# Input size

# input_size = cols

# Output size

output_size = 1

# Embedding Dimension

embedding_dim = 128

# Hidden Dimension

hidden_dim = 256

# Number of RNN Layers

n_layers = 2



# Show stats for every n number of batches

show_every_n_batches = 1
def train_rnn(model, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):

    train_loss = 0

    valid_loss = 0

    batch_losses = []

    val_batch_losses = []

    valid_loss_min = np.Inf

    

    model.train()

    

    previousLoss = np.Inf

    minLoss = np.Inf



    print("Training for %d epoch(s)..." % n_epochs)

    for epoch_i in range(1, n_epochs + 1):

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables

            # print(np.squeeze(target))

            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the batch loss      

            loss = criterion(output, target)

            # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # perform a single optimization step (parameter update)

            optimizer.step()

            # update training loss

            ## find the loss and update the model parameters accordingly

            ## record the average training loss, using something like

            #batch_losses.append(loss.item())

            train_loss += ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

            

        model.eval()

        for batch_idx, (data, target) in enumerate(val_loader):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## update the average validation loss

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the batch loss

            loss = criterion(output, target)

            # update average validation loss 

            #val_batch_losses.append(loss.item())

            valid_loss += ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))

        

        valid_loss = valid_loss/len(val_loader.dataset)

        train_loss = train_loss/len(train_loader.dataset)

        

        #train_loss = np.average(batch_losses)

        #valid_loss = np.average(val_batch_losses)

        

        # print training/validation statistics 

        if epoch_i%show_every_n_batches == 0:

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch_i, train_loss, valid_loss))

        

            ## TODO: save the model if validation loss has decreased

            # save model if validation loss has decreased

            if valid_loss < valid_loss_min:

                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                valid_loss_min,

                valid_loss))

                with open('trained_rnn_new', 'wb') as pickle_file:

                    torch.save(model.state_dict(), 'trained_rnn_new')

                valid_loss_min = valid_loss

                train_loss = 0

                valid_loss = 0

                batch_losses = []

                val_batch_losses = []

  

    return model
model_transfer = models.vgg16(pretrained=True)

print(model_transfer)
# os.mkdir("../save")
# Freeze training for all "features" layers

for param in model_transfer.features.parameters():

    param.requires_grad = False

    

custom_model = nn.Sequential(nn.Linear(25088, 1024), 

                  nn.ReLU(),

                  nn.Dropout(p=0.5), 

                  nn.Linear(1024, 512), 

                  nn.ReLU(),

                  nn.Dropout(p=0.5),

                  nn.Linear(512, 2)

                 )



model_transfer.classifier = custom_model



if use_cuda:

    model_transfer = model_transfer.cuda()



# print(model_transfer)



# specify loss function

criterion_scratch = nn.CrossEntropyLoss()

#criterion_scratch = nn.BCELoss()



# specify optimizer

optimizer_scratch = optim.SGD(model_transfer.classifier.parameters(), lr=learning_rate, momentum=0.9)

#optimizer_scratch = optim.Adam(model_transfer.classifier.parameters(),lr=0.0001)



#trained_rnn = train_rnn(model_transfer, batch_size, optimizer_scratch, criterion_scratch, num_epochs, show_every_n_batches)
#optimizer_scratch = optim.SGD(model_transfer.classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

#train_rnn(model_transfer, batch_size, optimizer_scratch, criterion_scratch, num_epochs, show_every_n_batches)
model_transfer.load_state_dict(torch.load('trained_rnn_new'))

# test_files = np.array(glob("test/*"))

# print(test_files[:3])
def predict(file):

    test_transform = transforms.Compose([transforms.ToPILImage(),

                                             transforms.Resize(224), 

                                             transforms.CenterCrop(224), 

                                             transforms.ToTensor()

                                            ])

    image = cv2.imread(file)

    image = test_transform(image)

    image = image.unsqueeze(0)

    if use_cuda:

        image = image.cuda()

    

    output = model_transfer(image)

    print(output)

    pr = output[:,1].detach().cpu().numpy()

    print(pr)

    # print(os.path.basename(file))

    print(output)

    values, indices = output.max(1)

    #print(values)

    #print(indices)

    return os.path.basename(file), indices.item()
model_transfer.eval()

submission = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

preds = []

for batch_idx, (data, target) in enumerate(test_loader):

    # move to GPU

    if use_cuda:

        data, target = data.cuda(), target.cuda()

    ## update the average validation loss

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model_transfer(data)

    values, indices = output.max(1)

    pr = output[:,1].detach().cpu().numpy()

    preds.append(indices.item())

    #for i in pr:

        #preds.append(i)

    #print(output)

    #print(pr)

    #print(values)

    #print(indices)

    # calculate the batch loss

submission["has_cactus"] = preds

print(submission.head())
submission.to_csv('submission_61.csv', index=False)
predictoutput = pd.read_csv("../input/mydatasetforaerialcactus/submission_output.csv")

predictoutput.to_csv('submission_6.csv', index=False)
# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename="submission_1.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(submission)