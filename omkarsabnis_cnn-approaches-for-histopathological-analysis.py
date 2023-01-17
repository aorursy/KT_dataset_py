# Modules required for simple model

from glob import glob 

import numpy as np

import pandas as pd

import cv2,os

import keras

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers import BatchNormalization

from keras.layers import Activation

from keras.layers import Conv2D

from keras.layers import MaxPool2D

from keras.layers import BatchNormalization

from tqdm import tqdm_notebook,trange

import matplotlib.pyplot as plt

import gc

scale = 70

seed = 7

# Modules required for NASNET Mobile Model

from random import shuffle

from sklearn.model_selection import train_test_split

from imgaug import augmenters as iaa

import imgaug as ia

# Modules required for DenseNet 121 Model

#from fastai.vision import *

#import torchvision
#set paths to training and test data

path = "../input/"

train_path = path + 'train/'

test_path = path + 'test/'

df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))})

df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])

labels = pd.read_csv(path+"train_labels.csv")

# merge labels and filepaths

df = df.merge(labels, on = "id") 

# Loading the images

def load_data(N,df):

    # allocate a numpy array for the images - 3 channels

    X = np.zeros([N,96,96,3],dtype=np.uint8) 

    #convert the labels to an array

    y = np.squeeze(df.as_matrix(columns=['label']))[0:N]

    #read images one by one, tdqm notebook displays a progress bar

    for i, row in tqdm_notebook(df.iterrows(), total=N):

        if i == N:

            break

        X[i] = cv2.imread(row['path'])

    return X,y

N = 10000

X,y = load_data(N=N,df=df)
# Displaying the loaded images

fig = plt.figure(figsize=(10, 4), dpi=150)

np.random.seed(100)

for plotNr,idx in enumerate(np.random.randint(0,N,8)):

    ax = fig.add_subplot(2, 8//2, plotNr+1, xticks=[], yticks=[])

    plt.imshow(X[idx])

    ax.set_title('Label: ' + str(y[idx]))
fig = plt.figure(figsize=(4, 2),dpi=150)

plt.bar([1,0], [(y==0).sum(), (y==1).sum()]); #plot a bar chart of the label frequency

plt.xticks([1,0],["Negative (N={})".format((y==0).sum()),"Positive (N={})".format((y==1).sum())]);

plt.ylabel("# of samples")
#Separating the classes

positive_samples = X[y == 1]

negative_samples = X[y == 0]

#Binning each pixel value for the histogram

nr_of_bins = 256 

fig,axs = plt.subplots(4,2,sharey=True,figsize=(8,8),dpi=150)

#RGB channels

axs[0,0].hist(positive_samples[:,:,:,0].flatten(),bins=nr_of_bins,density=True)

axs[0,1].hist(negative_samples[:,:,:,0].flatten(),bins=nr_of_bins,density=True)

axs[1,0].hist(positive_samples[:,:,:,1].flatten(),bins=nr_of_bins,density=True)

axs[1,1].hist(negative_samples[:,:,:,1].flatten(),bins=nr_of_bins,density=True)

axs[2,0].hist(positive_samples[:,:,:,2].flatten(),bins=nr_of_bins,density=True)

axs[2,1].hist(negative_samples[:,:,:,2].flatten(),bins=nr_of_bins,density=True)

#All channels

axs[3,0].hist(positive_samples.flatten(),bins=nr_of_bins,density=True)

axs[3,1].hist(negative_samples.flatten(),bins=nr_of_bins,density=True)

# Labelling the Plots

axs[0,0].set_title("Positive samples (N =" + str(positive_samples.shape[0]) + ")");

axs[0,1].set_title("Negative samples (N =" + str(negative_samples.shape[0]) + ")");

axs[0,1].set_ylabel("Red",rotation='horizontal',labelpad=35,fontsize=12)

axs[1,1].set_ylabel("Green",rotation='horizontal',labelpad=35,fontsize=12)

axs[2,1].set_ylabel("Blue",rotation='horizontal',labelpad=35,fontsize=12)

axs[3,1].set_ylabel("RGB",rotation='horizontal',labelpad=35,fontsize=12)

for i in range(4):

    axs[i,0].set_ylabel("Relative frequency")

axs[3,0].set_xlabel("Pixel value")

axs[3,1].set_xlabel("Pixel value")

fig.tight_layout()
# We use 64 bins to get smooth graphs

nr_of_bins = 64 

fig,axs = plt.subplots(1,2,sharey=True, sharex = True, figsize=(8,2),dpi=150)

axs[0].hist(np.mean(positive_samples,axis=(1,2,3)),bins=nr_of_bins,density=True);

axs[1].hist(np.mean(negative_samples,axis=(1,2,3)),bins=nr_of_bins,density=True);

axs[0].set_title("Mean brightness, positive samples");

axs[1].set_title("Mean brightness, negative samples");

axs[0].set_xlabel("Image mean brightness")

axs[1].set_xlabel("Image mean brightness")

axs[0].set_ylabel("Relative frequency")

axs[1].set_ylabel("Relative frequency");
# Getting the number of images in the dataset

N = df["path"].size

X,y = load_data(N=N,df=df)



# Collecting garbage

positives_samples = None

negative_samples = None

gc.collect();



# Setting up the training/testing ratio

training_portion = 0.8 

split_idx = int(np.round(training_portion * y.shape[0]))



#Setting seeds to ensure we can repeat this process 

np.random.seed(42) 

idx = np.arange(y.shape[0])

np.random.shuffle(idx)

X = X[idx]

y = y[idx]
# Network Parameters

kernel_size = (3,3)

pool_size= (2,2)

first_filters = 32

second_filters = 64

third_filters = 128



# Setting up dropout parameters for regularization

dropout_conv = 0.3

dropout_dense = 0.5



# Creating model

model = Sequential()



# Convolutional Block 1

model.add(Conv2D(first_filters, kernel_size, input_shape = (96, 96, 3)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(first_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size)) 

model.add(Dropout(dropout_conv))



# Convolutional Block 2

model.add(Conv2D(second_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



# Convolutional Block 3

model.add(Conv2D(third_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



# Dense Layer 

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(dropout_dense))



# Softmax - to convert values to 0 or 1

model.add(Dense(1, activation = "sigmoid"))



model.summary()



batch_size = 50



model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(0.001), 

              metrics=['accuracy'])
epochs = 3 #how many epochs we want to perform

for epoch in range(epochs):

    #compute how many batches we'll need

    iterations = np.floor(split_idx / batch_size).astype(int) #the floor makes us discard a few samples here, I got lazy...

    loss,acc = 0,0 #we will compute running loss and accuracy

    with trange(iterations) as t: #display a progress bar

        for i in t:

            start_idx = i * batch_size #starting index of the current batch

            x_batch = X[start_idx:start_idx+batch_size] #the current batch

            y_batch = y[start_idx:start_idx+batch_size] #the labels for the current batch



            metrics = model.train_on_batch(x_batch, y_batch) #train the model on a batch



            loss = loss + metrics[0] #compute running loss

            acc = acc + metrics[1] #compute running accuracy

            t.set_description('Running training epoch ' + str(epoch)) #set progressbar title

            t.set_postfix(loss="%.2f" % round(loss / (i+1),2),acc="%.2f" % round(acc / (i+1),2)) #display metrics
#compute how many batches we'll need

iterations = np.floor((y.shape[0]-split_idx) / batch_size).astype(int) #as above, not perfect

loss,acc = 0,0 #we will compute running loss and accuracy

with trange(iterations) as t: #display a progress bar

    for i in t:

        start_idx = i * batch_size #starting index of the current batch

        x_batch = X[start_idx:start_idx+batch_size] #the current batch

        y_batch = y[start_idx:start_idx+batch_size] #the labels for the current batch

        

        metrics = model.test_on_batch(x_batch, y_batch) #compute metric results for this batch using the model

        

        loss = loss + metrics[0] #compute running loss

        acc = acc + metrics[1] #compute running accuracy

        t.set_description('Running training') #set progressbar title

        t.set_description('Running validation')

        t.set_postfix(loss="%.2f" % round(loss / (i+1),2),acc="%.2f" % round(acc / (i+1),2))

        

print("Validation loss:",loss / iterations)

print("Validation accuracy:",acc / iterations)
X = None

y = None

gc.collect();

base_test_dir = path + 'test/' #specify test data folder

test_files = glob(os.path.join(base_test_dir,'*.tif')) #find the test file names

submission = pd.DataFrame() #create a dataframe to hold results

file_batch = 5000 #we will predict 5000 images at a time

max_idx = len(test_files) #last index to use

for idx in range(0, max_idx, file_batch): #iterate over test image batches

    print("Indexes: %i - %i"%(idx, idx+file_batch))

    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]}) #add the filenames to the dataframe

    test_df['id'] = test_df.path.map(lambda x: x.split('/')[3].split(".")[0]) #add the ids to the dataframe

    test_df['image'] = test_df['path'].map(cv2.imread) #read the batch

    K_test = np.stack(test_df["image"].values) #convert to numpy array

    predictions = model.predict(K_test,verbose = 1) #predict the labels for the test data

    test_df['label'] = predictions #store them in the dataframe

    submission = pd.concat([submission, test_df[["id", "label"]]])

print(submission.head())

submission.to_csv("submission.csv", index = False, header = True) #create the submission file