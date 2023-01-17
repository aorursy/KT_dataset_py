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
import sys



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from torch import optim



import torchvision.transforms as transforms

import torchvision



from fastprogress import master_bar, progress_bar



from PIL import Image
data = pd.read_csv("../input/vietai-advanced-final-project-00/train.csv")

data.head()
LABELS = data.columns[5:]

LABELS
train_data, val_data = train_test_split(data, test_size=0.1, random_state=2019)
IMAGE_SIZE = 224                              # Image size (224x224)

IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)

IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

BATCH_SIZE = 96                              

LEARNING_RATE = 0.001

LEARNING_RATE_SCHEDULE_FACTOR = 0.1           # Parameter used for reducing learning rate

LEARNING_RATE_SCHEDULE_PATIENCE = 5           # Parameter used for reducing learning rate

MAX_EPOCHS = 100                              # Maximum number of training epochs
class ChestXrayDataset(Dataset):

    

    def __init__(self, folder_dir, dataframe, image_size, normalization):

        """

        Init Dataset

        

        Parameters

        ----------

        folder_dir: str

            folder contains all images

        dataframe: pandas.DataFrame

            dataframe contains all information of images

        image_size: int

            image size to rescale

        normalization: bool

            whether applying normalization with mean and std from ImageNet or not

        """

        self.image_paths = [] # List of image paths

        self.image_labels = [] # List of image labels

        

        # Define list of image transformations

        image_transformation = [

            transforms.Resize((image_size, image_size)),

            transforms.ToTensor()

        ]

        

        if normalization:

            # Normalization with mean and std from ImageNet

            image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

        

        self.image_transformation = transforms.Compose(image_transformation)

        

        # Get all image paths and image labels from dataframe

        for index, row in dataframe.iterrows():

            image_path = os.path.join(folder_dir, row.Path)

            self.image_paths.append(image_path)

            if len(row) < 14:

                labels = [0] * 14

            else:

                labels = []

                for col in row[5:]:

                    if col == 1:

                        labels.append(1)

                    else:

                        labels.append(0)

            self.image_labels.append(labels)

    

    def __len__(self):

        return len(self.image_paths)

    

    def __getitem__(self, index):

        """

        Read image at index and convert to torch Tensor

        """

        

        # Read image

        image_path = self.image_paths[index]

        image_data = Image.open(image_path).convert("RGB") # Convert image to RGB channels

        

        # TODO: Image augmentation code would be placed here

        

        # Resize and convert image to torch tensor 

        image_data = self.image_transformation(image_data)

        

        return image_data, torch.FloatTensor(self.image_labels[index])
train_dataset = ChestXrayDataset("../input/vietai-advanced-final-project-00/train/train", train_data, IMAGE_SIZE, True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
for data, label in train_dataloader:

    print(data.size())

    print(label.size())

    break
val_dataset = ChestXrayDataset("../input/vietai-advanced-final-project-00/train/train", val_data, IMAGE_SIZE, True)

val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
class DenseNet121(nn.Module):

    def __init__(self, num_classes, is_trained=True):

        """

        Init model architecture

        

        Parameters

        ----------

        num_classes: int

            number of classes

        is_trained: bool

            whether using pretrained model from ImageNet or not

        """

        super().__init__()

        

        # Load the DenseNet121 from ImageNet

        self.net = torchvision.models.densenet121(pretrained=is_trained)

        

        # Get the input dimension of last layer

        kernel_count = self.net.classifier.in_features

        

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output

        self.net.classifier = nn.Sequential(nn.Linear(kernel_count, num_classes), nn.Sigmoid())

        

    def forward(self, inputs):

        """

        Forward the netword with the inputs

        """

        return self.net(inputs)
device = "cuda" if torch.cuda.is_available() else "cpu"

device
model = DenseNet121(num_classes=len(LABELS)).to(device)

model
sum(p.numel() for p in model.parameters() if p.requires_grad)
# Loss function

loss_criteria = nn.BCELoss()



# Adam optimizer

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)



# Learning rate will be reduced automatically during training

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = LEARNING_RATE_SCHEDULE_FACTOR, patience = LEARNING_RATE_SCHEDULE_PATIENCE, mode = 'max', verbose=True)
def multi_label_auroc(y_gt, y_pred):

    """ Calculate AUROC for each class



    Parameters

    ----------

    y_gt: torch.Tensor

        groundtruth

    y_pred: torch.Tensor

        prediction



    Returns

    -------

    list

        F1 of each class

    """

    auroc = []

    gt_np = y_gt.to("cpu").numpy()

    pred_np = y_pred.to("cpu").numpy()

    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"

    for i in range(gt_np.shape[1]):

        auroc.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))

    return auroc
def epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb):

    """

    Epoch training



    Paramteters

    -----------

    epoch: int

      epoch number

    model: torch Module

      model to train

    train_dataloader: Dataset

      data loader for training

    device: str

      "cpu" or "cuda"

    loss_criteria: loss function

      loss function used for training

    optimizer: torch optimizer

      optimizer used for training

    mb: master bar of fastprogress

      progress to log



    Returns

    -------

    float

      training loss

    """

    # Switch model to training mode

    model.train()

    training_loss = 0 # Storing sum of training losses

   

    # For each batch

    for batch, (images, labels) in enumerate(progress_bar(train_dataloader, parent=mb)):

        

        # Move X, Y  to device (GPU)

        images = images.to(device)

        labels = labels.to(device)

        

        # Clear previous gradient

        optimizer.zero_grad()



        # Feed forward the model

        pred = model(images)

        loss = loss_criteria(pred, labels)



        # Back propagation

        loss.backward()



        # Update parameters

        optimizer.step()



        # Update training loss after each batch

        training_loss += loss.item()



        mb.child.comment = f'Training loss {training_loss/(batch+1)}'



    del images, labels, loss

    if torch.cuda.is_available(): torch.cuda.empty_cache()



    # return training loss

    return training_loss/len(train_dataloader)
def evaluating(epoch, model, val_loader, device, loss_criteria, mb):

    """

    Validate model on validation dataset

    

    Parameters

    ----------

    epoch: int

        epoch number

    model: torch Module

        model used for validation

    val_loader: Dataset

        data loader of validation set

    device: str

        "cuda" or "cpu"

    loss_criteria: loss function

      loss function used for training

    mb: master bar of fastprogress

      progress to log

  

    Returns

    -------

    float

        loss on validation set

    float

        metric score on validation set

    """



    # Switch model to evaluation mode

    model.eval()



    val_loss = 0                                   # Total loss of model on validation set

    out_pred = torch.FloatTensor().to(device)      # Tensor stores prediction values

    out_gt = torch.FloatTensor().to(device)        # Tensor stores groundtruth values



    with torch.no_grad(): # Turn off gradient

        # For each batch

        for step, (images, labels) in enumerate(progress_bar(val_loader, parent=mb)):

            # Move images, labels to device (GPU)

            images = images.to(device)

            labels = labels.to(device)



            # Update groundtruth values

            out_gt = torch.cat((out_gt,  labels), 0)



            # Feed forward the model

            ps = model(images)

            loss = loss_criteria(ps, labels)



            # Update prediction values

            out_pred = torch.cat((out_pred, ps), 0)



            # Update validation loss after each batch

            val_loss += loss

            mb.child.comment = f'Validation loss {val_loss/(step+1)}'



    # Clear memory

    del images, labels, loss

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # return validation loss, and metric score

    return val_loss/len(val_loader), np.array(multi_label_auroc(out_gt, out_pred)).mean()
import time
# Best AUROC value during training

best_score = 0

model_path = "densenet.pth"

training_losses = []

validation_losses = []

validation_score = []





# Config progress bar

mb = master_bar(range(MAX_EPOCHS))

mb.names = ['Training loss', 'Validation loss', 'Validation AUROC']

x = []



nonimproved_epoch = 0

start_time = time.time()



# Training each epoch

for epoch in mb:

    mb.first_bar.comment = f'Best AUROC score: {best_score}'

    x.append(epoch)



    # Training

    train_loss = epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb)

    mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_loss))

    training_losses.append(train_loss)



    # Evaluating

    val_loss, new_score = evaluating(epoch, model, val_dataloader, device, loss_criteria, mb)

    mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, new_score))

    validation_losses.append(val_loss)

    validation_score.append(new_score)



    # Update learning rate

    lr_scheduler.step(new_score)



    # Update training chart

    mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0,epoch+1], [0,1])



    # Save model

    if best_score < new_score:

        mb.write(f"Improve AUROC from {best_score} to {new_score}")

        best_score = new_score

        nonimproved_epoch = 0

        torch.save({"model": model.state_dict(), 

                    "optimizer": optimizer.state_dict(), 

                    "best_score": best_score, 

                    "epoch": epoch, 

                    "lr_scheduler": lr_scheduler.state_dict()}, model_path)

    else: 

        nonimproved_epoch += 1

    if nonimproved_epoch > 10:

        break

        print("Early stopping")

    if time.time() - start_time > 3600*8:

        break

        print("Out of time")
test_df = pd.read_csv("../input/vietai-advanced-final-project-00/sample_submission.csv")

test_df.head()
test_dataset = ChestXrayDataset("../input/vietai-advanced-final-project-00/test/test", test_df, IMAGE_SIZE, True)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=700, shuffle=False, num_workers=2, pin_memory=True)
state = torch.load(model_path, map_location=device)

model.load_state_dict(state["model"])

model.eval()
from tqdm import tqdm



out_pred = torch.FloatTensor().to(device)

with torch.no_grad(): # Turn off gradient

    # For each batch

    for step, (images, _) in tqdm(enumerate(test_dataloader)):

        # Move images, labels to device (GPU)

        images = images.to(device)



        # Feed forward the model

        ps = model(images)

        out_pred = torch.cat((out_pred, ps), dim=0)
label_list = list(LABELS)

for col in test_df.columns[1:]:

    test_df[col] = out_pred[:, label_list.index(col)].cpu().numpy()
test_df.to_csv("submission.csv", index=False)

test_df.head()