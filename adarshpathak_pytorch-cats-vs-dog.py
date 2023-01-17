!pip install torch-summary
import numpy as np

import pandas as pd

from imageio import imread

import os

import random

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from torchsummary import summary

import torch

import torchvision

import torch.nn as nn

from torchvision import transforms

import warnings

warnings.filterwarnings('ignore')



from PIL import Image
train_dir = '/kaggle/input/cat-and-dog/training_set/training_set'

test_dir = '/kaggle/input/cat-and-dog/test_set/test_set'
cats_train = os.path.join(train_dir,'cats')

dogs_train = os.path.join(train_dir,'dogs')

cats_test =  os.path.join(test_dir,'cats')

dogs_test =  os.path.join(test_dir,'dogs')
def visualize_image():

    available_image_folders = [cats_train,dogs_train,cats_test,dogs_test]

    plt.figure(figsize=(15,7))

    for i in range(6):

        folder = random.choice(available_image_folders)

        plt.subplot(2,3,i+1)

        imgs_choice = random.choice([i for i in os.listdir(folder)])

        images_path = os.path.join(folder,imgs_choice)

        images = imread(images_path)

        plt.imshow(images)

    plt.show()
visualize_image()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])



train_transform = torchvision.transforms.Compose([

    transforms.Resize((224,224)),

        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        normalize

])



test_transform = torchvision.transforms.Compose([

    transforms.Resize((224,224)),

        transforms.ToTensor(),

        normalize

])
train_dataloader = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(train_dir,

                                                                                       transform=train_transform),

                                              shuffle=True,batch_size=32,num_workers=0)
test_dataloader = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(test_dir,

                                                                                     transform=test_transform),

                                             shuffle=True,batch_size=32,num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
model = torchvision.models.resnet50(pretrained=True).to(device)
for params in model.parameters():

    params.requires_grads = False
model.fc = nn.Sequential(

    nn.Linear(2048,128),

    nn.ReLU(inplace=True),

    nn.Linear(128,2)

).to(device)
criterian = nn.CrossEntropyLoss()

optimizers = torch.optim.Adam(model.fc.parameters())
model_summary = summary(model,(3,224,224))
n_epochs = 4

for epoch in tqdm_notebook(range(n_epochs)):

    model.train()

    for batch_idx,(data,labels) in tqdm_notebook(enumerate(train_dataloader)):

        data = data.to(device)

        labels = labels.to(device)

        optimizers.zero_grad()

        output = model(data)

        loss = criterian(output,labels)

        loss.backward()

        optimizers.step()

    print(f'epochs: {epoch} loss: {loss.item()}')
torch.save(model.state_dict(),'model1.h5')
test_imgs = ["/cats/cat.4001.jpg",

                        "/cats/cat.4003.jpg",

                        "/dogs/dog.4004.jpg",

                       "/dogs/dog.4006.jpg"]

img_list = [Image.open(test_dir + img_path) for img_path in test_imgs]
test_batch = torch.stack([test_transform(img).to(device)

                                for img in img_list])
pred_logits_tensor = model(test_batch)

pred_logits_tensor
pred_probs = nn.functional.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

pred_probs
fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))

for i, img in enumerate(img_list):

    ax = axs[i]

    ax.axis('off')

    ax.set_title("{:.0f}% Cat, {:.0f}% Dog".format(100*pred_probs[i,0],

                                                            100*pred_probs[i,1]))

    ax.imshow(img)