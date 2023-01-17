from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch # import pytorch

import torchvision

from PIL import Image

from torch import optim, nn

from torchvision import datasets, transforms, models

from torch.utils import data

from torch.autograd import Variable

import time

import copy

import seaborn as sns

print(torch.__version__) # find out the version of pytorch
print(os.listdir('../input'))

#find out the train and test data directories

print(os.listdir('../input/flower_data/flower_data'))

print(os.listdir('../'))
# Transform the image (scaling, flipping and normalisation)

data_transforms = {

    'train': transforms.Compose([

        transforms.RandomRotation(30),

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(20),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.Resize(255),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ]),

}



data_dir = '../input/flower_data/flower_data'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),

                                          data_transforms[x])

                  for x in ['train', 'valid']}



#info about no. of datapoints

image_datasets
#Create the data loaders

batch_size = 64

# trainLoader = data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False)

# testLoader = data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False)



dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) 

                   for x in ['train', 'valid']}
def show(image):

    if isinstance(image, torch.Tensor):

        image = image.numpy().transpose((1, 2, 0))

    else:

        image = np.array(image).transpose((1, 2, 0))

    # denormalisation

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean

    image = np.clip(image, 0, 1)

    # plot

    fig, Xaxis = plt.subplots(1, 1, figsize=(9, 9))

    %matplotlib inline

    plt.imshow(image)

    Xaxis.axis('off') 
# Make a grid from batch (for training data)

# This grid shows the images which are present in 1 batch

images, _ = next(iter(dataloaders['train']))

trainGrid = torchvision.utils.make_grid(images, nrow=8)



show(trainGrid)
# Make a grid from batch (for validation/test data)

images, _ = next(iter(dataloaders['valid']))

testGrid = torchvision.utils.make_grid(images, nrow=8)



show(testGrid)
classNames = image_datasets['train'].classes



#Get labels to class names from json file

import json

with open('../input/cat_to_name.json', 'r') as f:

    labelToName = json.load(f)



# print(image_datasets['train'].classes)

labelToName
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
# TODO: Build and train your network

model = models.densenet121(pretrained=True)



for param in model.parameters():

    

    param.requires_grad = False





model.classifier = nn.Sequential(nn.Linear(1024, 256),

                                 nn.ReLU(),

                                 nn.Dropout(p=0.1),

                                 nn.Linear(256, 128),

                                 nn.ReLU(),

                                 nn.Dropout(p=0.1),

                                 nn.Linear(128, 102),

                                 nn.LogSoftmax(dim=1))



criterion = nn.NLLLoss()



optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)



model.to(device)
def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):

  

    steps = 0

    running_loss = 0

    for e in range(epochs):

        

        # Model in training mode, dropout is on

        model.train()

        for images, labels in trainloader:

            

            steps += 1

            

            # Flatten images into a 784 long vector

            #images.resize_(images.size()[0], -1)

            

            images = images.to(device)

            labels = labels.to(device)

            

            optimizer.zero_grad()

            

            output = model.forward(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            

            running_loss += loss.item()



            if steps % print_every == 0:

                # Model in inference mode, dropout is off

                model.eval()

                

                # Turn off gradients for validation, will speed up inference

                with torch.no_grad():

                    test_loss, accuracy = validation(model, testloader, criterion)

                

                print("Epoch: {}/{}.. ".format(e+1, epochs),

                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),

                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),

                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                

                running_loss = 0

                

                # Make sure dropout and grads are on for training

                model.train()



def validation(model, testloader, criterion):

  

    accuracy = 0

    test_loss = 0

    for images, labels in testloader:



        #images = images.resize_(images.size()[0], -1)

        

        images = images.to(device)

        labels = labels.to(device)



        output = model.forward(images)

        test_loss += criterion(output, labels).item()



        ## Calculating the accuracy 

        # Model's output is log-softmax, take exponential to get the probabilities

        ps = torch.exp(output)

        # Class with highest probability is our predicted class, compare with true label

        equality = (labels.data == ps.max(1)[1])

        # Accuracy is number of correct predictions divided by all predictions, just take the mean

        accuracy += equality.type_as(torch.FloatTensor()).mean()



    return test_loss, accuracy
# uncomment these two lines when you want to re-train



# train(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, epochs=12)

# validation(model, dataloaders['valid'], criterion)
# Save the checkpoint



# checkpoint = {'input_size': 1024,

#               'output_size': 102,

#               'epochs': 12,

#               'classifier': model.classifier,

#               'optimizer_state': optimizer.state_dict(),

#               'mapping': image_datasets['train'].class_to_idx,

#               'state_dict': model.state_dict()}



# # Save the checkpoint 

# torch.save(checkpoint, 'checkpoint.pth')

# idx_to_class = {}

# for x in image_datasets['train'].class_to_idx:

#     idx_to_class[image_datasets['train'].class_to_idx[x]] = labelToName[x]
# A function that loads a checkpoint and rebuilds the model



def load_checkpoint(filepath):

    print('Loading checkpoint...')

    checkpoint = torch.load(filepath, map_location='cpu')

    model = models.densenet121(pretrained=True)

    model.classifier = checkpoint['classifier']

    model.optimizer_state = checkpoint['optimizer_state']

    model.mapping = checkpoint['mapping']

    model.load_state_dict(checkpoint['state_dict'])

    print('Done')

    return model
print(os.listdir(os.getcwd()))

print(os.getcwd())
# print(os.listdir('../input/checkpoint'))

model = load_checkpoint(os.getcwd()+'/checkpoint.pth')
#Image Processing



def process_image(image_path):

    

    img = Image.open(image_path)

    

    if img.size[0] > img.size[1]:  #resizing the image

        img.thumbnail((10000, 256))

    else:

        img.thumbnail((256, 10000))

        

    #crop the image

    left_margin = (img.width-224)/2

    bottom_margin = (img.height - 224)/2

    right_margin = left_margin + 224

    top_margin = bottom_margin + 224

    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    

    #normalising

    img = np.array(img)/255

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean)/std

    

    #moving color channels to first dimention as expected by pytorch

    img = img.transpose((2,0,1))

    

    return img
def imshow(image, ax=None, title=None):

    if ax is None:

        fig, ax = plt.subplots()

    if title:

        plt.title(title)

    # PyTorch tensors assume the color channel is first

    # but matplotlib assumes is the third dimension

    image = image.transpose((1, 2, 0))

    

    # Undo preprocessing

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean

    

    # Image needs to be clipped between 0 and 1

    image = np.clip(image, 0, 1)

    

    ax.imshow(image)

    

    return ax
#Predicting the class 



def predict(image_path, model, topk=5):

    

#     print('Prediction on Flower Classification starts')

    model.eval()

    model.cpu()

    

    idx_to_class = {i: k for k, i in model.mapping.items()}

    

    #opening the image

    with Image.open(image_path) as image:

#         print(image_path, image)

        image = process_image(image_path)

        

    #switching it to float tensor

    image = torch.FloatTensor([image])

    

    #feeding it through the model

    output = model.forward(image)

    

    #to determine the topk probability and labels

    topk_prob, topk_labels = torch.topk(output, topk)

    

    #taking exp() of image to cancel out the LogSoftmax

    topk_prob = topk_prob.exp()

    

    #assemble the lists

    topk_prob_arr = topk_prob.data.numpy()[0]

    topk_indexes_list = topk_labels.data.numpy()[0].tolist()

    topk_labels_list = [idx_to_class[x] for x in topk_indexes_list]    

    topk_class_arr = [labelToName[str(x)] for x in topk_labels_list]    

    return topk_labels_list, topk_class_arr
test_prediction = {}

i = 0

j = 0

image_path = '../input/test set/test set'

print(len(os.listdir(image_path)))

for x in os.listdir(image_path):

#     print(j)

    if x == 'gc11.png':

#         x += 1

        continue

    print(x)

    predicted_labels, predicted_class = predict(os.path.join(image_path, x), model)

    test_prediction[x] = [predicted_class[0], predicted_labels[0]]

    j += 1
def createDataframe(test_pred):

    return pd.DataFrame.from_dict(test_pred, orient='index', columns=['Class', 'Labels'])
createDataframe(test_prediction)
#End of this project
