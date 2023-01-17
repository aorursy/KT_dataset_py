!pip3 install git+https://github.com/JasperHG90/malaria-convnet
!pip3 install torchsummary
import shutil

import os

import logging

import random

from glob import glob

from pathlib import Path

from convNet.model import convNet

from torchsummary import summary

from torchvision import datasets, transforms

import torch

import numpy as np

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)



# Ensure reproducibility

torch.manual_seed(767365)

random.seed(5423)

torch.backends.cudnn.deterministic = True # Runs on CPU.

torch.backends.cudnn.benchmark = False
img_path = "../input/cell-images-for-detecting-malaria/cell_images/"

image_files = glob(img_path + "/*/*.png")
for data_part in ["train", "val", "test"]:

    Path(f"data/{data_part}/Parasitized").mkdir(parents=True, exist_ok=True)

    Path(f"data/{data_part}/Uninfected").mkdir(parents=True, exist_ok=True)
# SHuffle images

np.random.seed(25254)

np.random.shuffle(image_files)
# Split into batches

val_imgs = image_files[:250]

test_imgs = image_files[250:500]

train_imgs = image_files[500:]
base_dir = "data"

for img in val_imgs:

    fn_split = img.split("/")

    img_name = fn_split[-1]

    img_class = fn_split[-2]

    img_out = os.path.join(base_dir, "val", img_class, img_name)

    if not os.path.exists(img_out):

        shutil.copy(img, img_out)

    

for img in test_imgs:

    fn_split = img.split("/")

    img_name = fn_split[-1]

    img_class = fn_split[-2]

    img_out = os.path.join(base_dir, "test", img_class, img_name)

    if not os.path.exists(img_out):

        shutil.copy(img, img_out)

    

for img in train_imgs:

    fn_split = img.split("/")

    img_name = fn_split[-1]

    img_class = fn_split[-2]

    img_out = os.path.join(base_dir, "train", img_class, img_name)

    if not os.path.exists(img_out):

        shutil.copy(img, img_out)
!ls data/train/Uninfected | head -n 5
H = W = 32

# Define transformations

transform = {

    'train': transforms.Compose([

        transforms.Resize([H,W]),

        transforms.RandomHorizontalFlip(),

        transforms.RandomAffine(

            degrees=(-15,15),

            translate=(0,.2),

            scale=(.8, 1.2),

            shear=0.1,

        ),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))

    ]),

    'test': transforms.Compose([

        transforms.Resize([H,W]),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))

    ])

}
train_data_folder = "data/train"

val_data_folder = "data/val"

# Set up data loaders

train_dataset = datasets.ImageFolder(

    root = train_data_folder,

    transform = transform["train"],

)



# Validation data

val_dataset = datasets.ImageFolder(

    root = val_data_folder,

    transform = transform["test"]

)

train_data_loader = torch.utils.data.DataLoader(

    dataset=train_dataset,

    batch_size=64,

    shuffle=True,

    num_workers=4

)



val_data_loader = torch.utils.data.DataLoader(

    dataset=val_dataset,

    batch_size=64,

    shuffle=True,

    num_workers=4

)
net = convNet(dropout=0.4)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)



summary(net, (3, H, W))
for epoch in range(5):

    running_loss = 0.0

    acc = 0

    batches = 0

    for i, data in enumerate(train_data_loader, 0):

        batch_x, batch_y = data

        # Zero gradients

        optimizer.zero_grad()

        # Forward pass, backward pass

        outputs = net(batch_x)

        loss = criterion(outputs.view(-1), batch_y.type(torch.FloatTensor))

        loss.backward()

        # Optimize parameters

        optimizer.step()

        # print statistics

        running_loss += loss.item()

        if i % 100 == 99:    # print every 100 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 100))

            running_loss = 0.0

        batches += 1

        outputs_class = outputs > 0

        acc_current = torch.sum(outputs_class.view(-1) == batch_y).numpy() / batch_y.shape[0]

        acc += acc_current

    acc /= batches

    print("Accuracy on train set is: %s" % acc)

    # On cross-validation set

    with torch.no_grad():

        acc = 0

        batches = 0

        for i, data in enumerate(val_data_loader, 0):

            batch_x, batch_y = data

            outputs = net(batch_x)

            loss = criterion(outputs.view(-1), batch_y.type(torch.FloatTensor)).item()

            # Predict

            outputs_class = outputs > 0

            acc_current = torch.sum(outputs_class.view(-1) == batch_y).numpy() / batch_y.shape[0]

            batches += 1

            acc += acc_current

        acc /= batches

        print("Accuracy on validation set is: %s" % acc)
test_data_folder = "data/test"

# Define the dataset

test_dataset = datasets.ImageFolder(

    root = test_data_folder,

    transform = transform["test"]

)

# Set up the data loader

test_data_loader = torch.utils.data.DataLoader(

    dataset=test_dataset,

    batch_size=64,

    shuffle=True,

    num_workers=4

)



with torch.no_grad():

    acc = 0

    batches = 0

    for i, data in enumerate(test_data_loader, 0):

        batch_x, batch_y = data

        outputs = net(batch_x)

        loss = criterion(outputs.view(-1), batch_y.type(torch.FloatTensor)).item()

        # Predict

        outputs_class = outputs > 0

        acc_current = torch.sum(outputs_class.view(-1) == batch_y).numpy() / batch_y.shape[0]

        batches += 1

        acc += acc_current

    acc /= batches

    print("Accuracy on validation set is: %s" % acc)
torch.save(net.state_dict(), "model.pt")
# Remove the image data from the current directory

shutil.rmtree("data")