!pip install stegano                     # steganalysis library

!pip install -q efficientnet_pytorch     # Convolutional Neural Net from Google Research
import stegano

from stegano import lsb



# System

import cv2

import os, os.path

from PIL import Image              # from RBG to YCbCr



# Basics

import pandas as pd

import numpy as np

from numpy import pi                # for DCT

from numpy import r_                # for DCT

import scipy                        # for cosine similarity

from scipy import fftpack           # for DCT

import random

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg    # to check images

%matplotlib inline

from tqdm.notebook import tqdm      # beautiful progression bar



# SKlearn

from sklearn.model_selection import KFold

from sklearn import metrics



# PyTorch

import torch

import torch.nn as nn

import torch.optim as optim

from torch import FloatTensor, LongTensor

from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F



# Data Augmentation for Image Preprocessing

from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,

                            RandomBrightness, RandomContrast, HueSaturationValue, Blur, GaussNoise)

from albumentations.pytorch import ToTensorV2, ToTensor

from efficientnet_pytorch import EfficientNet

from torchvision.models import resnet34



import warnings

warnings.filterwarnings("ignore")
print(list(os.listdir("../input/v2-effnet-epoch-6-auc-08023")))
# Create a new image with secret message

msg_to_hide = "Message encoded blue cats from Mars are coming to enslave us all be aware!!!!!!"

secret = lsb.hide("../input/suki-image/capture27.png", 

                    msg_to_hide, 

                    auto_convert_rgb=True)

secret.save("./SukiSecret.png")



# Reveal the hidden message

print(lsb.reveal("./SukiSecret.png"))



# See the 2 images side by side (no apparent difference, but WE KNOW the text is there.)

f, ax = plt.subplots(1, 2, figsize=(14,5))

                           

original = mpimg.imread('../input/suki-image/capture27.png')

original_plot = ax[0].imshow(original)



altered = mpimg.imread('./SukiSecret.png')

altered_plot = ax[1].imshow(altered)
# From image to array 

# (vectorize the matrix to be able to feed it to the cosine function)

original_vector = np.array(original).flatten()

altered_vector = np.array(altered).flatten()



print('Original shape:', original_vector.shape, '\n' +

      'Altered shape:', altered_vector.shape)





# Distance between the original image and itself (should be 0, because they are identical)

dist1 = np.sum(original_vector - original_vector)

print('Dist1:', dist1)



# Distance between the original image and altered image

dist2 = np.sum(original_vector - altered_vector)

print('Dist2:', dist2)
# ---- STATICS ----

base_path = '../input/alaska2-image-steganalysis'



def read_images_path(dir_name='Cover', test = False):

    '''series_name: 0001.jpg, 0002.jpg etc.

    series_paths: is the complete path to a certain image.'''

    

    # Get name of the files

    series_name = pd.Series(os.listdir(base_path + '/' + dir_name))

    if test:

        series_name = pd.Series(os.listdir(base_path + '/' + 'Test'))

    

    # Create the entire path

    series_paths = pd.Series(base_path + '/' + dir_name + '/' + series_name)

    

    return series_paths
# Read in the data

cover_paths = read_images_path('Cover', False)

jmipod_paths = read_images_path('JMiPOD', False)

juniward_paths = read_images_path('JUNIWARD', False)

uerd_paths = read_images_path('UERD', False)

test_paths = read_images_path('Test', True)
def show15(title = "Default"):

    '''Shows n amount of images in the data'''

    plt.figure(figsize=(16,9))

    plt.suptitle(title, fontsize = 16)

    

    for k, path in enumerate(cover_paths[:15]):

        cover = mpimg.imread(path)

        

        plt.subplot(3, 5, k+1)

        plt.imshow(cover)

        plt.axis('off')
show15(title = "15 Original Images")
image_sample = mpimg.imread(cover_paths[0])



print('Image sample shape:', image_sample.shape)

print('Image sample size:', image_sample.size)

print('Image sample data type:', image_sample.dtype)
def show_images_alg(n = 3, title="Default"):

    '''Returns a plot of the original Image and Encoded ones.

    n: number of images to display'''

    

    f, ax = plt.subplots(n, 4, figsize=(16, 7))

    plt.suptitle(title, fontsize = 16)

    



    for index in range(n):

        cover = mpimg.imread(cover_paths[index])

        ipod = mpimg.imread(jmipod_paths[index])

        juni = mpimg.imread(juniward_paths[index])

        uerd = mpimg.imread(uerd_paths[index])



        # Plot

        ax[index, 0].imshow(cover)

        ax[index, 1].imshow(ipod)

        ax[index, 2].imshow(juni)

        ax[index, 3].imshow(uerd)

        

        # Add titles

        if index == 0:

            ax[index, 0].set_title('Original', fontsize=12)

            ax[index, 1].set_title('IPod', fontsize=12)

            ax[index, 2].set_title('Juni', fontsize=12)

            ax[index, 3].set_title('Uerd', fontsize=12)
show_images_alg(n = 3, title = "Algorithm Difference")
def show_ycbcr_images(n = 3, title = "Default"):

    '''Shows n images as: original RGB, YCbCr and Y, Cb, Cr channels split'''

    

    # 4: original image, YCbCr image, Y, Cb, Cr (separate chanels)

    fig, ax = plt.subplots(n, 5, figsize=(16, 7))

    plt.suptitle(title, fontsize = 16)



    for index, path in enumerate(cover_paths[:n]):

        # Read in the original image and convert

        original_image = Image.open(path)

        ycbcr_image = original_image.convert('YCbCr')

        (y, cb, cr) = ycbcr_image.split()



        # Plot

        ax[index, 0].imshow(original_image)

        ax[index, 1].imshow(ycbcr_image)

        ax[index, 2].imshow(y)

        ax[index, 3].imshow(cb)

        ax[index, 4].imshow(cr)



        # Add Title

        if index==0:

            ax[index, 0].set_title('Original', fontsize=12)

            ax[index, 1].set_title('YCbCr', fontsize=12)

            ax[index, 2].set_title('Y', fontsize=12)

            ax[index, 3].set_title('Cb', fontsize=12)

            ax[index, 4].set_title('Cr', fontsize=12)
show_ycbcr_images(n = 3, title = "YCbCr Channels")
# Read in an Image Example

image = mpimg.imread(cover_paths[2])



plt.figure(figsize = (6, 6))

plt.imshow(image)

plt.title('Original Image', fontsize=16)

plt.axis('off');
# Define 2D DCT

def dct2(a):

    # Return the Discrete Cosine Transform of arbitrary type sequence x.

    return fftpack.dct(fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho')



# Perform a blockwise DCT

imsize = image.shape

dct = np.zeros(imsize)



# Do 8x8 DCT on image (in-place)

for i in r_[:imsize[0]:8]:

    for j in r_[:imsize[1]:8]:

        dct[i:(i+8),j:(j+8)] = dct2( image[i:(i+8),j:(j+8)] )
# ---- STATICS ----

pos = 128   # can be changed



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))



# Display original

ax1.imshow(image[pos:pos+8,pos:pos+8],cmap='gray')

ax1.set_title("An 8x8 block : Original Image", fontsize=16)



# Display the dct of that block

ax2.imshow(dct[pos:pos+8,pos:pos+8],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])

ax2.set_title("An 8x8 DCT block", fontsize = 16);
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))



# Original image

ax1.imshow(image);

ax1.set_title("Original Image", fontsize = 16);



# DCT Blocks

ax2.imshow(dct,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)

ax2.set_title("DCT blocks", fontsize = 14);
# Threshold

thresh = 0.02

dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))





plt.figure(figsize=(14, 6))

plt.imshow(dct_thresh, cmap='gray', vmax = np.max(dct)*0.01, vmin = 0)

plt.title("Thresholded 8x8 DCTs of the image", fontsize = 16)



percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)

print("Keeping only {}% of the DCT coefficients".format(round(percent_nonzeros*100.0, 3)))
def set_seed(seed = 1234):

    '''Sets the seed of the entire notebook so results are the same every time we run.

    This is for REPRODUCIBILITY.'''

    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set

    torch.backends.cudnn.deterministic = True

    # Set a fixed value for the hash seed

    os.environ['PYTHONHASHSEED'] = str(seed)

    

set_seed()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device available now:', device)
# ----- STATICS -----

sample_size = 256

num_classes = 4

# -------------------



# Read in Data



# --- 10 classes ---

# train_df = pd.read_csv('../input/alaska2trainvalsplit/alaska2_train_df.csv', 

#                        header=0, names=['Path', 'Label'], dtype = {'Label':np.int32})

# valid_df = pd.read_csv('../input/alaska2trainvalsplit/alaska2_val_df.csv', 

#                        header=0, names=['Path', 'Label'], dtype = {'Label':np.int32})



# --- 4 classes ---

train_df = pd.read_csv('../input/alaska2-trainvalid-4-class-csv/alaska2_train_data_4classes.csv', 

                       header=0, names=['Path', 'Label'], dtype = {'Label':np.int32})

valid_df = pd.read_csv('../input/alaska2-trainvalid-4-class-csv/alaska2_valid_data_4classes.csv', 

                       header=0, names=['Path', 'Label'], dtype = {'Label':np.int32})



# Sample out Data

def sample_data(dataframe, sample_size, num_classes, train=True):

    '''Sample same number of images for each label.'''

    if train:

        size = int(0.75 * sample_size)

    else:

        size = int(0.25 * sample_size)

        

    # Number of images in class

    no = int(np.floor(size/num_classes))

    labels = [i for i in range(num_classes)]

    new_data = pd.DataFrame()

    

    # For each label

    for label in labels:

        # Sample out data

        data = dataframe[dataframe['Label'] == label].sample(no, random_state=123)

        new_data = pd.concat([new_data, data], axis=0)

        

    return new_data
# Sample out data

train_df = sample_data(train_df, num_classes=num_classes, 

                       sample_size=sample_size, train=True).reset_index(drop=True)

valid_df = sample_data(valid_df, num_classes=num_classes, 

                       sample_size=sample_size, train=False).reset_index(drop=True)
print('Train Data Size:', len(train_df), '\n' +

      'Valid Data Size:', len(valid_df), '\n' +

      '----------------------', '\n' +

      'Total:', len(train_df) + len(valid_df))



fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))



sns.countplot(x = train_df['Label'], ax = ax1, palette = sns.color_palette("GnBu_d", 10))

sns.countplot(x = valid_df['Label'], ax = ax2, palette = sns.color_palette("YlOrRd", 10))



ax1.set_title('Train Data', fontsize=16)

ax2.set_title('Valid Data', fontsize=16);
class AlaskaDataset(Dataset):

    '''Alaska2 Dataset.

    If data is test or eval, it skips the transformations applied to training part.'''

            

    def __init__(self, dataframe, is_test, is_val, vertical_flip=0.5, horizontal_flip=0.5):

        self.dataframe, self.is_test, self.is_val = dataframe, is_test, is_val

        self.vertical_flip, self.horizontal_flip = vertical_flip, horizontal_flip

        # Flag to mark Testing and Evaluation Datasets

        flag = is_test or is_val

        

        # If data is NOT Train

        if flag:

            self.transform = Compose([Resize(512, 512), 

                                      Normalize(),

                                      ToFloat(max_value=255),

                                      ToTensor()])

        else:

            # Compose transforms and handle all transformations regarding bounding boxes

            self.transform = Compose([Resize(512, 512), 

                                      VerticalFlip(p = vertical_flip),

                                      HorizontalFlip(p = horizontal_flip),

                                      Normalize(),

                                      ToFloat(max_value=255),

                                      # Convert image and mask to torch.Tensor

                                      ToTensor()])        

        

    

    # So len(data) returns the size of dataset

    def __len__(self):

        return len(self.dataframe)

    

    # Very important function for Data Loader

    def __getitem__(self, index):

        

        if self.is_test: 

            path = self.dataframe.loc[index][0]

        else:

            path, label = self.dataframe.loc[index]

        

        # ::-1 to not overload memory

        image = cv2.imread(path)[:, :, ::-1]

        image = self.transform(image=image)

        image = image['image']

        

        if self.is_test:

            return image

        else:

            return image, label
class EfficientNetwork(nn.Module):

    def __init__(self, output_size, b1=False, b2=False):

        super().__init__()

        self.b1, self.b2 = b1, b2

        

        # Define Feature part

        if b1:

            self.features = EfficientNet.from_pretrained('efficientnet-b1')

        elif b2:

            self.features = EfficientNet.from_pretrained('efficientnet-b2')

        else:

            self.features = EfficientNet.from_pretrained('efficientnet-b0')

        

        # Define Classification part

        if b1:

            self.classification = nn.Linear(1280, output_size)

        elif b2:

            self.classification = nn.Linear(1408, output_size)

        else:

            self.classification = nn.Linear(1280, output_size)

        

        

    def forward(self, image, prints=False):

        if prints: print('Input Image shape:', image.shape)

        

        image = self.features.extract_features(image)

        if prints: print('Features Image shape:', image.shape)

            

        if self.b1:

            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1280)

        elif self.b2:

            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)

        else:

            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1280)

        if prints: print('Image Reshaped shape:', image.shape)

        

        out = self.classification(image)

        if prints: print('Out shape:', out.shape)

        

        return out
# Create an example model (B2)

model_example = EfficientNetwork(output_size=num_classes, b1=False, b2=True)
# Data object and Loader

example_data = AlaskaDataset(train_df, is_test=False, is_val=False)

example_loader = torch.utils.data.DataLoader(example_data, batch_size = 1, shuffle=True)



# Get a sample

for image, labels in example_loader:

    images_example = image

    labels_example = torch.tensor(labels, dtype=torch.long)

    break

print('Images shape:', images_example.shape)

print('Labels:', labels, '\n')



# Outputs

out = model_example(images_example, prints=True)



# Criterion example

criterion_example = nn.CrossEntropyLoss()

loss = criterion_example(out, labels_example)

print('Loss:', loss.item())
class ResNet34Network(nn.Module):

    def __init__(self, output_size):

        super().__init__()

        

        # Define Feature part

        self.features = resnet34(pretrained=True)

        

        # Define Classification part

        self.classification = nn.Linear(1000, output_size)

        

        

    def forward(self, image, prints=False):

        if prints: print('Input Image shape:', image.shape)

        

        image = self.features(image)

        if prints: print('Features Image shape:', image.shape)

        

        out = self.classification(image)

        if prints: print('Out shape:', out.shape)

        

        return out
# Create an example model

model_example = ResNet34Network(output_size=num_classes)
# Data object and Loader

example_data = AlaskaDataset(train_df, is_test=False, is_val=False)

example_loader = torch.utils.data.DataLoader(example_data, batch_size = 1, shuffle=True)



# Get a sample

for image, labels in example_loader:

    images_example = image

    labels_example = torch.tensor(labels, dtype=torch.long)

    break

print('Images shape:', images_example.shape)

print('Labels:', labels, '\n')



# Outputs

out = model_example(images_example, prints=True)



# Criterion example

criterion_example = nn.CrossEntropyLoss()

loss = criterion_example(out, labels_example)

print('Loss:', loss.item())
# ----- STATICS -----

vertical_flip = 0.5

horizontal_flip = 0.5

# -------------------
# Data Objects

train_data = AlaskaDataset(train_df, is_test=False, is_val=False, 

                           vertical_flip=vertical_flip, horizontal_flip=horizontal_flip)

valid_data = AlaskaDataset(valid_df, is_test=False, is_val=True, 

                           vertical_flip=vertical_flip, horizontal_flip=horizontal_flip)
def alaska_weighted_auc(y_true, y_valid):

    tpr_thresholds = [0.0, 0.4, 1.0]

    weights = [2, 1]

    

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    

    # Size of subsets

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

    normalization = np.dot(areas, weights)

    

    competition_metric = 0

    for idx, weight in enumerate(weights):

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        # Normalize such that curve starts at y = 0

        y = y - y_min 

        score = metrics.auc(x, y)

        submetric = score * weight

        best_subscore = (y_max - y_min) * weight

        competition_metric += submetric

        

    return competition_metric / normalization
def train(model, epochs, batch_size, num_workers, learning_rate, weight_decay, 

          version = 'vx', plot_loss=False):

    # Create file to save logs

    f = open(f"logs_{version}.txt", "w+")

    

    # Best AUC value

    best_auc = None    

    

    # Data Loaders

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,

                                              drop_last=True, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers,

                                              drop_last=True, shuffle=True)



    # Criterion

    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max',

                                                           patience=1, verbose=True, factor=0.4)





    train_losses = []

    evaluation_losses = []



    for epoch in range(epochs):



        # Sets the model in training mode

        model.train()



        train_loss = 0



        for images, labels in train_loader:

            # Need to access the images

            images = images.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.long)



            # Clear gradients

            optimizer.zero_grad()



            # Make prediction

            out = model(images)



            # Compute loss and Backpropagate

            loss = criterion(out, labels)

            loss.backward()

            optimizer.step()



            train_loss += loss.item()



        # Compute average epoch loss

        epoch_loss_train = train_loss / batch_size

        train_losses.append(epoch_loss_train)





        # ===== Evaluate =====

        model.eval()



        evaluation_loss = 0

        actuals, predictions = [], []



        # To disable gradients

        with torch.no_grad():

            for images, labels in valid_loader:

                images = images.to(device, dtype=torch.float)

                labels = labels.to(device, dtype=torch.long)



                # Prediction

                out = model(images)

                loss = criterion(out, labels)

                actuals.extend(labels.cpu().numpy().astype(int))

                predictions.extend(F.softmax(out, 1).cpu().numpy())



                evaluation_loss += loss.item()



        # Compute epoch loss

        epoch_loss_eval = evaluation_loss/batch_size

        evaluation_losses.append(epoch_loss_eval)



        # Prepare predictions and actuals

        predictions = np.array(predictions)

        # Choose label (array)

        predicted_labels = predictions.argmax(1)



        # ----- Accuracy -----

        accuracy = (predicted_labels == actuals).mean()



        # Compute AUC

        new_preds = np.zeros(len(predictions))

        temp = predictions[predicted_labels != 0, 1:]



        new_preds[predicted_labels != 0] = temp.sum(1)

        new_preds[predicted_labels == 0] = 1 - predictions[predicted_labels == 0, 0]

        actuals = np.array(actuals)

        actuals[actuals != 0] = 1



        auc_score = alaska_weighted_auc(actuals, new_preds)





        with open(f"logs_{version}.txt", 'a+') as f:

            print('Epoch: {}/{} | Train Loss: {:.3f} | Eval Loss: {:.3f} | AUC: {:3f} | Acc: {:3f}'.\

                     format(epoch+1, epochs, epoch_loss_train, epoch_loss_eval, auc_score, accuracy), file=f)

        

        print('Epoch: {}/{} | Train Loss: {:.3f} | Eval Loss: {:.3f} | AUC: {:3f} | Acc: {:3f}'.\

              format(epoch+1, epochs, epoch_loss_train, epoch_loss_eval, auc_score, accuracy))



        # Update AUC

        # If AUC is improving, then we also save model

        if best_auc == None:

            best_auc = auc_score

            torch.save(model.state_dict(),

                       f"Epoch_{epoch+1}_ValLoss_{epoch_loss_eval:.3f}_AUC_{auc_score:.3f}.pth")

            continue

            

        if auc_score > best_auc:

            best_auc = auc_score

            torch.save(model.state_dict(),

                       f"Epoch_{epoch+1}_ValLoss_{epoch_loss_eval:.3f}_AUC_{auc_score:.3f}.pth")

        

        # Update scheduler (for learning_rate)

        scheduler.step(auc_score)

        

    # Plots the loss of Train and Valid

    if plot_loss:

        plt.figure(figsize=(14,5))

        plt.plot(train_losses, c='#fdc975ff', lw = 3)

        plt.plot(evaluation_losses, c='#29896bff', lw = 3)

        plt.legend(['Train Loss', 'Evaluation Loss'])

        plt.title('Losses over Epochs');
# ----- STATICS -----

version = 'v8'

epochs = 10

batch_size = 32

num_workers = 8

learning_rate = 0.0001

weight_decay = 0.00001

plot_loss = False

# -------------------
# # Efficient Net B0

# eff_net0 = EfficientNetwork(output_size = num_classes, b1=False, b2=False).to(device)



# # Load any pretrained model

# eff_net0.load_state_dict(torch.load('../input/v2-effnet-epoch-6-auc-08023/10Class_epoch_14_val_loss_77.34_auc_0.78_EffNetB0.pth'))



# # Uncomment and train the model

# train(model=eff_net0, epochs=epochs, batch_size=batch_size, num_workers=num_workers, 

#       learning_rate=learning_rate, weight_decay=weight_decay, plot_loss=plot_loss)
# Efficient Net B2

eff_net2 = EfficientNetwork(output_size = num_classes, b1=False, b2=True).to(device)



# # Add previous trained model:

# eff_net2.load_state_dict(torch.load('../input/v2-effnet-epoch-6-auc-08023/Epoch_7_ValLoss_58.146_AUC_0.799.pth'))



# Uncomment and train the model

# train(model=eff_net2, epochs=epochs, batch_size=batch_size, num_workers=num_workers, 

#       learning_rate=learning_rate, weight_decay=weight_decay, plot_loss=plot_loss)
# # ResNet34

# eff_net34 = ResNet34Network(output_size = num_classes).to(device)



# # Uncomment and train the model

# train(model=eff_net34, epochs=epochs, batch_size=batch_size, num_workers=num_workers, 

#       learning_rate=learning_rate, weight_decay=weight_decay, plot_loss=plot_loss)
# Extract a sample of paths

directory = '../input/alaska2-image-steganalysis/'

name = pd.Series(sorted(os.listdir(directory + 'Test/')))

path = pd.Series(directory + 'Test/' + name)



# Create dataframe

test_df = pd.DataFrame(data=path, columns=['Path'])



# Dataset

test_data = AlaskaDataset(test_df, is_test=True, is_val=False,

                          vertical_flip=vertical_flip, horizontal_flip=horizontal_flip)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle=False)
# list(os.listdir('../working/Epoch_17_ValLoss_36.195_AUC_0.794.pth'))



# Import model if necessary:

# Load any pretrained model

# eff_net2.load_state_dict(torch.load('../working/Epoch_17_ValLoss_36.195_AUC_0.794.pth'))
# Evaluation Mode

eff_net2.eval()



predictions = []



with torch.no_grad():

    for k, images in enumerate(test_loader):

        images = images.to(device)

        out0 = eff_net2(images)

        

        # Flip vertical

        images_vertical = images.flip(2)

        out1 = eff_net2(images_vertical)

        

        # Flip again original

        images_flip = images.flip(3)

        out2 = eff_net2(images_flip)

        

        # 50% results from flip + 50% result from normal

        outputs = (0.25*out1 + 0.25*out2)

        outputs = (outputs + 0.5*out0)



        predictions.extend(F.softmax(outputs, 1).cpu().numpy())
# Making the predictions the same manner as in Train Function

predictions = np.array(predictions)

predicted_labels = predictions.argmax(1)

new_preds = np.zeros(len(predictions))

temp = predictions[predicted_labels != 0, 1:]

new_preds[predicted_labels != 0] = temp.sum(1)

new_preds[predicted_labels == 0] = 1 - predictions[predicted_labels == 0, 0]
ss = pd.read_csv('../input/alaska2-image-steganalysis/sample_submission.csv')

ss['Label'] = new_preds



ss.to_csv(f'submission_{version}.csv', index=False)
!pip install -q efficientnet_pytorch > /dev/null
from glob import glob

from sklearn.model_selection import GroupKFold

import cv2

from skimage import io

import torch

from torch import nn

import os

from datetime import datetime

import time

import random

import cv2

import pandas as pd

import numpy as np

import albumentations as A

import matplotlib.pyplot as plt

from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data import Dataset,DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

import sklearn



SEED = 42



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
%%time



dataset = []



for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):

    for path in glob('../input/alaska2-image-steganalysis/Cover/*.jpg'):

        dataset.append({

            'kind': kind,

            'image_name': path.split('/')[-1],

            'label': label

        })



random.shuffle(dataset)

dataset = pd.DataFrame(dataset)



gkf = GroupKFold(n_splits=5)



dataset.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):

    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number
# dataset = pd.read_csv('../input/alaska2-public-baseline/groupkfold_by_shonenkov.csv')
def get_train_transforms():

    return A.Compose([

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)



def get_valid_transforms():

    return A.Compose([

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
DATA_ROOT_PATH = '../input/alaska2-image-steganalysis'



def onehot(size, target):

    vec = torch.zeros(size, dtype=torch.float32)

    vec[target] = 1.

    return vec



class DatasetRetriever(Dataset):



    def __init__(self, kinds, image_names, labels, transforms=None):

        super().__init__()

        self.kinds = kinds

        self.image_names = image_names

        self.labels = labels

        self.transforms = transforms



    def __getitem__(self, index: int):

        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

            

        target = onehot(4, label)

        return image, target



    def __len__(self) -> int:

        return self.image_names.shape[0]



    def get_labels(self):

        return list(self.labels)
fold_number = 0



train_dataset = DatasetRetriever(

    kinds=dataset[dataset['fold'] != fold_number].kind.values,

    image_names=dataset[dataset['fold'] != fold_number].image_name.values,

    labels=dataset[dataset['fold'] != fold_number].label.values,

    transforms=get_train_transforms(),

)



validation_dataset = DatasetRetriever(

    kinds=dataset[dataset['fold'] == fold_number].kind.values,

    image_names=dataset[dataset['fold'] == fold_number].image_name.values,

    labels=dataset[dataset['fold'] == fold_number].label.values,

    transforms=get_valid_transforms(),

)
image, target = train_dataset[0]

numpy_image = image.permute(1,2,0).cpu().numpy()



fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    

ax.set_axis_off()

ax.imshow(numpy_image);
from sklearn import metrics



class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count

        

        

def alaska_weighted_auc(y_true, y_valid):

    """

    https://www.kaggle.com/anokas/weighted-auc-metric-updated

    """

    tpr_thresholds = [0.0, 0.4, 1.0]

    weights = [2, 1]



    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)



    # size of subsets

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])



    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

    normalization = np.dot(areas, weights)



    competition_metric = 0

    for idx, weight in enumerate(weights):

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)

        # pdb.set_trace()



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        y = y - y_min  # normalize such that curve starts at y=0

        score = metrics.auc(x, y)

        submetric = score * weight

        best_subscore = (y_max - y_min) * weight

        competition_metric += submetric



    return competition_metric / normalization

        

class RocAucMeter(object):

    def __init__(self):

        self.reset()



    def reset(self):

        self.y_true = np.array([0,1])

        self.y_pred = np.array([0.5,0.5])

        self.score = 0



    def update(self, y_true, y_pred):

        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)

        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]

        self.y_true = np.hstack((self.y_true, y_true))

        self.y_pred = np.hstack((self.y_pred, y_pred))

        self.score = alaska_weighted_auc(self.y_true, self.y_pred)

    

    @property

    def avg(self):

        return self.score
class LabelSmoothing(nn.Module):

    def __init__(self, smoothing = 0.05):

        super(LabelSmoothing, self).__init__()

        self.confidence = 1.0 - smoothing

        self.smoothing = smoothing



    def forward(self, x, target):

        if self.training:

            x = x.float()

            target = target.float()

            logprobs = torch.nn.functional.log_softmax(x, dim = -1)



            nll_loss = -logprobs * target

            nll_loss = nll_loss.sum(-1)

    

            smooth_loss = -logprobs.mean(dim=-1)



            loss = self.confidence * nll_loss + self.smoothing * smooth_loss



            return loss.mean()

        else:

            return torch.nn.functional.cross_entropy(x, target)
import warnings



warnings.filterwarnings("ignore")



class Fitter:

    

    def __init__(self, model, device, config):

        self.config = config

        self.epoch = 0

        

        self.base_dir = './'

        self.log_path = f'{self.base_dir}/log.txt'

        self.best_summary_loss = 10**5



        self.model = model

        self.device = device



        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [

            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

        ] 



        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.criterion = LabelSmoothing().to(self.device)

        self.log(f'Fitter prepared. Device is {self.device}')



    def fit(self, train_loader, validation_loader):

        for e in range(self.config.n_epochs):

            if self.config.verbose:

                lr = self.optimizer.param_groups[0]['lr']

                timestamp = datetime.utcnow().isoformat()

                self.log(f'\n{timestamp}\nLR: {lr}')



            t = time.time()

            summary_loss, final_scores = self.train_one_epoch(train_loader)



            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            self.save(f'{self.base_dir}/last-checkpoint.bin')



            t = time.time()

            summary_loss, final_scores = self.validation(validation_loader)



            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            if summary_loss.avg < self.best_summary_loss:

                self.best_summary_loss = summary_loss.avg

                self.model.eval()

                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:

                    os.remove(path)



            if self.config.validation_scheduler:

                self.scheduler.step(metrics=summary_loss.avg)



            self.epoch += 1



    def validation(self, val_loader):

        self.model.eval()

        summary_loss = AverageMeter()

        final_scores = RocAucMeter()

        t = time.time()

        for step, (images, targets) in enumerate(val_loader):

            if self.config.verbose:

                if step % self.config.verbose_step == 0:

                    print(

                        f'Val Step {step}/{len(val_loader)}, ' + \

                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \

                        f'time: {(time.time() - t):.5f}', end='\r'

                    )

            with torch.no_grad():

                targets = targets.to(self.device).float()

                batch_size = images.shape[0]

                images = images.to(self.device).float()

                outputs = self.model(images)

                loss = self.criterion(outputs, targets)

                final_scores.update(targets, outputs)

                summary_loss.update(loss.detach().item(), batch_size)



        return summary_loss, final_scores



    def train_one_epoch(self, train_loader):

        self.model.train()

        summary_loss = AverageMeter()

        final_scores = RocAucMeter()

        t = time.time()

        for step, (images, targets) in enumerate(train_loader):

            if self.config.verbose:

                if step % self.config.verbose_step == 0:

                    print(

                        f'Train Step {step}/{len(train_loader)}, ' + \

                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \

                        f'time: {(time.time() - t):.5f}', end='\r'

                    )

            

            targets = targets.to(self.device).float()

            images = images.to(self.device).float()

            batch_size = images.shape[0]



            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, targets)

            loss.backward()

            

            final_scores.update(targets, outputs)

            summary_loss.update(loss.detach().item(), batch_size)



            self.optimizer.step()



            if self.config.step_scheduler:

                self.scheduler.step()



        return summary_loss, final_scores

    

    def save(self, path):

        self.model.eval()

        torch.save({

            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),

            'scheduler_state_dict': self.scheduler.state_dict(),

            'best_summary_loss': self.best_summary_loss,

            'epoch': self.epoch,

        }, path)



    def load(self, path):

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_summary_loss = checkpoint['best_summary_loss']

        self.epoch = checkpoint['epoch'] + 1

        

    def log(self, message):

        if self.config.verbose:

            print(message)

        with open(self.log_path, 'a+') as logger:

            logger.write(f'{message}\n')
from efficientnet_pytorch import EfficientNet



def get_net():

    net = EfficientNet.from_pretrained('efficientnet-b2')

    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)

    return net



net = get_net().cuda()
class TrainGlobalConfig:

    num_workers = 4

    batch_size = 16 

    n_epochs = 100

    lr = 0.0001



    # -------------------

    verbose = True

    verbose_step = 1

    # -------------------



    # --------------------

    step_scheduler = False  # do scheduler.step after optimizer.step

    validation_scheduler = True  # do scheduler.step after validation stage loss



#     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR

#     scheduler_params = dict(

#         max_lr=0.001,

#         epochs=n_epochs,

#         steps_per_epoch=int(len(train_dataset) / batch_size),

#         pct_start=0.1,

#         anneal_strategy='cos', 

#         final_div_factor=10**5

#     )

    

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau

    scheduler_params = dict(

        mode='min',

        factor=0.5,

        patience=1,

        verbose=False, 

        threshold=0.0001,

        threshold_mode='abs',

        cooldown=0, 

        min_lr=1e-8,

        eps=1e-08

    )

    # --------------------
from catalyst.data.sampler import BalanceClassSampler



def run_training():

    device = torch.device('cuda:0')



    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),

        batch_size=TrainGlobalConfig.batch_size,

        pin_memory=False,

        drop_last=True,

        num_workers=TrainGlobalConfig.num_workers,

    )

    val_loader = torch.utils.data.DataLoader(

        validation_dataset, 

        batch_size=TrainGlobalConfig.batch_size,

        num_workers=TrainGlobalConfig.num_workers,

        shuffle=False,

        sampler=SequentialSampler(validation_dataset),

        pin_memory=False,

    )



    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)

#     fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')

    fitter.fit(train_loader, val_loader)
# run_training()
file = open('../input/alaska2-public-baseline/log.txt', 'r')

for line in file.readlines():

    print(line[:-1])

file.close()
checkpoint = torch.load('../input/alaska2-public-baseline/best-checkpoint-033epoch.bin')

net.load_state_dict(checkpoint['model_state_dict']);

net.eval();
checkpoint.keys()
class DatasetSubmissionRetriever(Dataset):



    def __init__(self, image_names, transforms=None):

        super().__init__()

        self.image_names = image_names

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_name = self.image_names[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/Test/{image_name}', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']



        return image_name, image



    def __len__(self) -> int:

        return self.image_names.shape[0]
dataset = DatasetSubmissionRetriever(

    image_names=np.array([path.split('/')[-1] for path in glob('../input/alaska2-image-steganalysis/Test/*.jpg')]),

    transforms=get_valid_transforms(),

)





data_loader = DataLoader(

    dataset,

    batch_size=8,

    shuffle=False,

    num_workers=2,

    drop_last=False,

)
%%time



result = {'Id': [], 'Label': []}

for step, (image_names, images) in enumerate(data_loader):

    print(step, end='\r')

    

    y_pred = net(images.cuda())

    y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]

    

    result['Id'].extend(image_names)

    result['Label'].extend(y_pred)
submission = pd.DataFrame(result)

submission.to_csv('submission.csv', index=False)

submission.head()
submission['Label'].hist(bins=100);