# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from kaggle_datasets import KaggleDatasets

IMAGE_SIZE= [256,256]; BATCH_SIZE = 32
#GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification') # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
AUTO = tf.data.experimental.AUTOTUNE
TRAINING_FILENAMES = tf.io.gfile.glob('/kaggle/input/256images/train*.tfrec')

image_dir = "/kaggle/input/256images"
train_csv = "/kaggle/input/siim-isic-melanoma-classification/train.csv"
train_meta = pd.read_csv(train_csv)
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    image = tf.image.resize(image, [256,256])
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def __len__(self):
    return self.len

def get_training_dataset():
    #Feed each of the files from TRAINING FILENAMES to the load_dataset function and save the output as "dataset"
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048) #Shuffle the dataset to remove any correlation in the training data which may skew the model.
    dataset = dataset.batch(BATCH_SIZE) #Combine consecutive elements of the dataset into chunks of size: BATCH_SIZE
    #dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
print("Training data shapes:")
for image, label in get_training_dataset().take(20):
    print(image.shape, label.shape)
    
from torch.utils.data import DataLoader, Dataset #Create an efficient dataloader set to feed images to the model
# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)
CLASSES = [0,1]

"""def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels
"""
import math
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms

import albumentations as A #Package of transformations
from albumentations.pytorch.transforms import ToTensorV2

train_loader = DataLoader(get_training_dataset(), batch_size = 16, shuffle = True, num_workers = 0)

from torch import nn
from torch.nn import functional as F
import torchvision.models as models

model = models.resnet18(pretrained=True)
model
import torch

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = False

    
from collections import OrderedDict

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(512, 256)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(256, 2)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.fc = classifier
model.fc

for parameter in model.fc.parameters():
    parameter.requires_grad = True

model.to(device)
model.fc.to(device)
# Train the classifier
def train_classifier(model, optimizer, criterion, train_loader, epochs):

    steps = 0
    print_every = 5

    for e in range(epochs):
        #start = time.time()

        model.train()

        running_loss = 0
        
        for images, labels in iter(train_loader):            
            images, labels = images.cuda(), labels.cuda()

            steps += 1
            print("Steps: " + str(steps))

            optimizer.zero_grad()

            output = model.forward(images)
            #print("Output Finished", time.time() - start)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                # Turn off gradients for validation, saves memory and computations
                #with torch.no_grad():
                 #   validation_loss, accuracy = validation(model, valid_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  #    "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_loader)),
                   #   "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader))
                     )

                running_loss = 0
                model.train()
                
    model_path = "/kaggle/working/Resnet181E.pth"
    torch.save(model, model_path)
                
    
from torch import optim

#Loss function
criterion = nn.CrossEntropyLoss()

# Gradient descent optimizer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
train_classifier(model, optimizer, criterion, train_loader, epochs = 1)