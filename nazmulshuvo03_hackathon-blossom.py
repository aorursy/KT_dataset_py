# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as matImage

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision.datasets as datasets

import torchvision.transforms as transforms

import torchvision.models as models
import json

from pprint import pprint



train_dir = "../input/flower_data/flower_data/train"

valid_dir = "../input/flower_data/flower_data/valid"

test_dir = "../input/test set"

num_file_train_dir = len(os.listdir(train_dir)) #102

num_file_valid_dir = len(os.listdir(valid_dir)) #102



classes = []

with open('../input/cat_to_name.json') as f:

    data = json.load(f)

    for i in range(len(data)):

        classes.append((i+1, data['{}'.format(i+1)]))



#pprint(data)

classes[0][1]
import PIL.Image as pil_image



images = []

fig=plt.figure(figsize=(32, 32))

columns = 5

rows = 21



for folder in os.listdir(train_dir):

    img_path = os.listdir(train_dir+"/"+folder)[0]

    img = pil_image.open(train_dir+"/"+folder+"/"+img_path)

    img = img.resize((64, 64))

    label = classes[int(folder)-1][1]

    images.append([img, label])



for i, (image, label) in enumerate(images):

    image = image

    fig.add_subplot(rows, columns, i+1)

    plt.title(label)

    plt.imshow(image)

plt.show()
im_size = 256

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),

                                    transforms.ColorJitter(),

                                    transforms.RandomHorizontalFlip(),

                                    transforms.RandomRotation(30),

                                    transforms.Resize((im_size, im_size)),

                                    transforms.ToTensor(),

                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                  ])



valid_transform = transforms.Compose([transforms.Resize((im_size, im_size)),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                     ])



test_transform = transforms.Compose([transforms.Resize((im_size, im_size)),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                     ])



train_data = datasets.ImageFolder(train_dir, transform=train_transform)

valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)

test_data = datasets.ImageFolder(test_dir, transform=test_transform)



trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

validloader = DataLoader(valid_data, batch_size=64)

testloader = DataLoader(test_data)
model = models.resnet50(pretrained=True)

epochs = 30



for param in model.parameters():

    param.requires_grad = False

    

model.fc = nn.Sequential(nn.Linear(2048, 512),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(512, 102),

                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



model.to(device)
# Print model's state_dict

print("Model's state_dict:")

for param_tensor in model.state_dict():

    print(param_tensor, "\t", model.state_dict()[param_tensor].size())



# Print optimizer's state_dict

print("Optimizer's state_dict:")

for var_name in optimizer.state_dict():

    print(var_name, "\t", optimizer.state_dict()[var_name])
train_losses, valid_losses, accuracy_list = [], [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

    else:

        valid_loss = 0

        accuracy = 0

        

        with torch.no_grad():

            model.eval()

            for images, labels in validloader:

                images, labels = images.to(device), labels.to(device)

                

                log_ps = model(images)

                loss_ps = criterion(log_ps, labels)

                valid_loss += loss_ps.item()

                

                ps = torch.exp(log_ps)

                top_p, top_class = log_ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            

        

        train_losses.append(running_loss/len(trainloader))

        valid_losses.append(valid_loss/len(validloader))

        accuracy_list.append(accuracy/len(validloader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),

              "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),

              "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

        running_loss = 0

        model.train()
checkpoint = {'model': model,

              'state_dict': model.state_dict(),

              'optimizer' : optimizer.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    model = checkpoint['model']

    model.load_state_dict(checkpoint['state_dict'])

    for parameter in model.parameters():

        parameter.requires_grad = False

    

    model.eval()

    

    return model



model = load_checkpoint('checkpoint.pth')

print(model)
fig, axs = plt.subplots(nrows=1, ncols=2)



axs[0].plot(train_losses, label='training loss')

axs[0].plot(valid_losses, label='validation loss')

axs[0].legend(frameon=False)



axs[1].plot(accuracy_list, label='accuracy')

axs[1].legend(frameon=False)



plt.show()
files = os.listdir(os.path.join(test_dir+"/test set"))

predicted_flower = []

predicted_class = []



for image, label in testloader:

    image = image.to(device)

    log_ps = model(image)

    output = torch.exp(log_ps)

    probs,top_class = output.topk(1, dim=1)

    class_name = classes[(top_class.item() - 1)]

    #print(class_name)

    predicted_flower.append(class_name[1])

    predicted_class.append(class_name[0])



# df = pd.DataFrame(prediction_list, columns=['Image', 'Flower Name', 'Class Number'])

df = pd.DataFrame({'Image': files, 'Flower Name': predicted_flower, 'Class Name': predicted_class})

pd.set_option('display.max_colwidth', -1)

df
# df.to_csv(r'../input/result.csv')
print(df.to_string())
# from PIL import Image



# files = os.listdir(str(test_dir) + "/test set/")



# prediction_list = []



# for file in files:

#     fullpath = str(test_dir) + "/test set/" + str(file)

#     with Image.open(fullpath) as f:

#         try:

#             img = test_transform(f)

#             img = img.unsqueeze(0)

#             with torch.no_grad():

#                 img = img.to(device)

#                 out = model(img)

#                 output = torch.exp(out)

#                 probs,top_class = output.topk(1, dim=1)

#                 class_name = classes[(top_class.item() - 1)]

#                 prediction_list.append([file, class_name[1], class_name[0]])

                

#         except:

#             None

# df = pd.DataFrame(prediction_list, columns=['Image', 'Flower Name', 'Class Number'])

# pd.set_option('display.max_colwidth', -1)

# print(len(os.listdir(test_dir+'/test set')))

# #print(df['Image'])

# predicted_images_list = []

# for image in df['Image']:

#     predicted_images_list.append(image)

    

# print(len(predicted_images_list))

# #set(predicted_images_list) & set(os.listdir(test_dir+'/test set'))

# len([i for i, j in zip(predicted_images_list, os.listdir(test_dir+'/test set')) if i == j])
import requests

from io import BytesIO



response = requests.get("https://images.homedepot-static.com/productImages/80198476-3136-423d-a608-b71602afb64c/svn/encore-azalea-shrubs-80691-64_1000.jpg")

single_image = Image.open(BytesIO(response.content))

single_img = test_transform(single_image).unsqueeze(0)

with torch.no_grad():

    out = model(single_img.to(device))

    output = torch.exp(out)

    probs,top_class = output.topk(1, dim=1)

    class_name = classes[(top_class.item() - 1)]

    print("File Name: test-azalea", "\tPredicted Label: ", 

          class_name, "\tAnd Label in Number: ",top_class.item())



single_image = single_image.resize((256, 256))

single_image