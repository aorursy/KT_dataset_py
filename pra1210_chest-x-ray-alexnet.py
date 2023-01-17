# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Define path to the data directory

data_dir = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')



# Path to train directory (Fancy pathlib...no more os.path!!)

tr_dir = data_dir / 'train'



# Path to validation directory

val_dir = data_dir / 'val'



# Path to test directory

te_dir = data_dir / 'test'



# Any results you write to the current directory are saved as output.
NormalCases_dir = tr_dir/'NORMAL'

PneumoniaCases_dir = tr_dir/'PNEUMONIA'



#Get the list of all images

NormalCases = NormalCases_dir.glob('*.jpeg')

PneumoniaCases = PneumoniaCases_dir.glob('*.jpeg')



# An empty list. We will insert the data into this list in (img_path, label) format

train_data = []



for img in NormalCases:

    train_data.append((img, 0))

    

for img in PneumoniaCases:

    train_data.append((img, 1))

    

train_data = pd.DataFrame(train_data, columns = ['image', 'label'], index = None)



# Shuffle the data

train_data = train_data.sample(frac = 1).reset_index(drop = True)

# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows

# The frac keyword argument specifies the fraction of rows to return in the random sample, so frac=1 means return all rows (in random order).

# If you wish to shuffle your dataframe in-place and reset the index, you could do

# Here, specifying drop=True prevents .reset_index from creating a column containing the old index entries.



train_data.head(5)
cases = train_data['label'].value_counts()

print(cases)
import matplotlib.pyplot as plt

import seaborn as sns



# https://seaborn.pydata.org/generated/seaborn.barplot.html

plt.figure(figsize = (10, 8))

sns.barplot(x = cases.index, y = cases.values)

plt.title('Number of Cases', fontsize = 15)

plt.xlabel('Case type', fontsize = 12)

plt.ylabel('Count', fontsize = 12)

plt.xticks(range(len(cases.index)), ['Normal - 0', 'Pneumonia -1 '])

plt.show()

from skimage.io import imread

pneumonia_samples = (train_data[train_data['label'] == 1]['image'].iloc[:4]).tolist()

normal_samples = (train_data[train_data['label'] == 0]['image'].iloc[:4]).tolist()



samples = pneumonia_samples + normal_samples

del pneumonia_samples, normal_samples



f, ax = plt.subplots(2, 4, figsize = (20, 10))

for i in range(8):

    img = imread(samples[i])

    ax[i//4, i%4].imshow(img, cmap='gray')

    if i<4:

        ax[i//4, i%4].set_title("Pneumonia")

    else:

        ax[i//4, i%4].set_title("Normal")

    ax[i//4, i%4].axis('off')

    ax[i//4, i%4].set_aspect('auto')

plt.show()
import cv2

from keras.utils import to_categorical



NormalCasesDir = val_dir/'NORMAL'

PneumoniaCasesDir = val_dir/'PNEUMONIA'



NormalCases = NormalCasesDir.glob('*.jpeg')

PneumoniaCases = PneumoniaCasesDir.glob('*.jpeg')



validation_data = []

validation_labels = []



# Some images are in grayscale while majority of them contains 3 channels. So, if the image is grayscale, we will convert into a image with 3 channels.

# We will normalize the pixel values and resizing all the images to 224x224 



# Normal cases

for img in NormalCases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(0, num_classes=2)

    validation_data.append(img)

    validation_labels.append(label)

                      

# Pneumonia cases        

for img in PneumoniaCases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(1, num_classes=2)

    validation_data.append(img)

    validation_labels.append(label)

    

# Convert the list into numpy arrays

validation_data = np.array(validation_data)

validation_labels = np.array(validation_labels)



print("Total number of validation examples: ", validation_data.shape)

print("Total number of labels:", validation_labels.shape)
from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator()



aug = ImageDataGenerator(

    rotation_range=20,

	zoom_range=0.15,

	width_shift_range=0.2,

	height_shift_range=0.2,

	shear_range=0.15,

	horizontal_flip=True,

	fill_mode="nearest")
def data_gen(data, batch_size):

    # Get total number of samples in the data

    n = len(data)

    steps = n//batch_size

    

    # Define two numpy arrays for containing batch data and labels

    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

    batch_labels = np.zeros((batch_size,2), dtype=np.float32)



    # Get a numpy array of all the indices of the input data

    indices = np.arange(n)

    

    # Initialize a counter

    i =0

    while True:

        np.random.shuffle(indices)

        # Get the next batch 

        count = 0

        next_batch = indices[(i*batch_size):(i+1)*batch_size]

        for j, idx in enumerate(next_batch):

            img_name = data.iloc[idx]['image']

            label = data.iloc[idx]['label']

            

              # one hot encoding

            encoded_label = to_categorical(label, num_classes=2)

            # read the image and resize

            img = cv2.imread(str(img_name))

            img = cv2.resize(img, (224,224))

            

            # check if it's grayscale

            if img.shape[2]==1:

                img = np.dstack([img, img, img])

            

            # cv2 reads in BGR mode by default

            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # normalize the image pixels

            orig_img = img.astype(np.float32)/255.

            

            batch_data[count] = orig_img

            batch_labels[count] = encoded_label

            

            # generating more samples of the undersampled class

            if label==0 and count < batch_size-2:

                aug_img1 = aug.augment_image(img)

                aug_img2 = aug.augment_image(img)

                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)

                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)

                aug_img1 = aug_img1.astype(np.float32)/255.

                aug_img2 = aug_img2.astype(np.float32)/255.

                

                batch_data[count+1] = aug_img1

                batch_labels[count+1] = encoded_label

                batch_data[count+2] = aug_img2

                batch_labels[count+2] = encoded_label

                count +=2

            

            else:

                count+=1

            

            if count==batch_size-1:

                break

            

        i+=1

        yield batch_data, batch_labels

            

        if i>=steps:

            i=0
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

import numpy as np

np.random.seed(1000)

#Instantiate an empty model

model = Sequential()



# 1st Convolutional Layer

model.add(Conv2D(filters = 96, input_shape = (224,224,3), kernel_size=(11,11), strides=(4,4), activation = 'relu'))

# Max Pooling

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



# 2nd Convolutional Layer

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), activation = 'relu'))

# Max Pooling

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



# 3rd Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation = 'relu'))



# 4th Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation = 'relu'))



# 5th Convolutional Layer

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation = 'relu'))

# Max Pooling

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



# Passing it to a Fully Connected layer

model.add(Flatten())

# 1st Fully Connected Layer

model.add(Dense(4096, input_shape=(224*224*3,), activation = 'relu'))

# Add Dropout to prevent overfitting

model.add(Dropout(0.4))



# 2nd Fully Connected Layer

model.add(Dense(4096, activation = 'relu'))



# Add Dropout

model.add(Dropout(0.4))



# 3rd Fully Connected Layer

model.add(Dense(1000, activation = 'relu'))



# Add Dropout

model.add(Dropout(0.4))



# Output Layer

model.add(Dense(2, activation = 'softmax'))



model.summary()



# Compile the model

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
batch_size = 16

epochs = 20



# Get a train data generator

train_data_gen = data_gen(data = train_data, batch_size = batch_size)



# Define the number of training steps

train_steps = train_data.shape[0]//batch_size
import cv2

from keras.utils import to_categorical



NormalCasesDir = te_dir/'NORMAL'

PneumoniaCasesDir = te_dir/'PNEUMONIA'



NormalCases = NormalCasesDir.glob('*.jpeg')

PneumoniaCases = PneumoniaCasesDir.glob('*.jpeg')



test_data = []

test_labels = []



# Some images are in grayscale while majority of them contains 3 channels. So, if the image is grayscale, we will convert into a image with 3 channels.

# We will normalize the pixel values and resizing all the images to 224x224 



# Normal cases

for img in NormalCases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(0, num_classes=2)

    test_data.append(img)

    test_labels.append(label)

                      

# Pneumonia cases        

for img in PneumoniaCases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(1, num_classes=2)

    test_data.append(img)

    test_labels.append(label)

    

# Convert the list into numpy arrays

test_data = np.array(test_data)

test_labels = np.array(test_labels)



print("Total number of test examples: ", test_data.shape)

print("Total number of labels:", test_labels.shape)
# Evaluation on test dataset

test_loss, test_score = model.evaluate(test_data, test_labels, batch_size = 16)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
# Get predictions

preds = model.predict(test_data, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(test_labels, axis=-1)



print(orig_test_labels.shape)

print(preds.shape)
# Calculate Precision and Recall

tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



print("Recall of the model is {:.2f}".format(recall))

print("Precision of the model is {:.2f}".format(precision))