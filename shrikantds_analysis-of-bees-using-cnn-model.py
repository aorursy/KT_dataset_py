
import pandas as pd
import numpy as np
import skimage
import skimage.io
import skimage.transform
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import random
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

np.random.seed(42)
# Global variables
img_folder='../input/bee_imgs/bee_imgs/'
img_width=100
img_height=100
img_channels=3

# Any results you write to the current directory are saved as output.
bees=pd.read_csv('../input/bee_data.csv')
bees.head()
bees.shape
bees=pd.read_csv('../input/bee_data.csv', 
                index_col=False,  
                parse_dates={'datetime':[1,2]},
                dtype={'subspecies':'category', 'health':'category','caste':'category'})

def read_or_skip(file):
    """This function is to supress imageio exception if file doesn't exist"""
    try:
        img = skimage.io.imread(img_folder + file)
        img = skimage.transform.resize(img, (img_width, img_height), mode='reflect')
        return img[:,:,:img_channels]
    except:
        #print('Skipping %s. %s' %(file, sys.exc_info()[1]))
        return None

bees['img'] = bees['file'].apply(read_or_skip)
bees.dropna(inplace=True)

# Print sample data without img array
bees.drop('img',axis=1).head()
bees.head()
bees.shape
f, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
bees['subspecies'].value_counts().plot(kind='bar',ax=ax[0,0])
ax[0,0].set_ylabel('Count')
ax[0,0].set_title('Subspecies')


bees['location'].value_counts().plot(kind='bar',ax = ax[0,1])
ax[0,1].set_title('Location')
ax[0,1].set_ylabel('Count')

bees['caste'].value_counts().plot(kind='bar',ax = ax[1,0])
ax[1,0].set_title('Caste')
ax[1,0].set_ylabel('Count')

bees['health'].value_counts().plot(kind='bar',ax = ax[1,1])
ax[1,1].set_title('Health')
ax[1,1].set_ylabel('Count')

f.subplots_adjust(hspace=0.7)
f.tight_layout()
plt.show()

import imageio
# Select first X subspecies titles 
columns=7
subspecies = bees['subspecies'].unique()[:7]
f, ax = plt.subplots(nrows=1,ncols=7, figsize=(12,3))
i=0

# Draw the first found bee of given subpecies
for s in subspecies:
    if s == 'healthy': continue
    file=img_folder + bees[bees['subspecies']==s].iloc[0]['file']
    im=imageio.imread(file)
    ax[i].imshow(im, resample=True)
    ax[i].set_title(s, fontsize=8)
    i+=1
    
plt.suptitle("Subspecies of Bee")
plt.tight_layout()
plt.show()


healthy = bees[bees['health'] == 'healthy'].iloc[:5]

f, ax = plt.subplots(nrows=1,ncols=5, figsize=(12,3))
# Read image of original size from disk, because bees['img'] contains resized numpy array
for i in range(0,5): 
    file = img_folder + healthy.iloc[i]['file']
    ax[i].imshow(imageio.imread(file))

plt.suptitle("Healthy Bees")
plt.tight_layout()
plt.show()
healths_cat = bees['health'].cat.categories
healths_cat
healths_cat.size
f, ax = plt.subplots(1, healths_cat.size-1, figsize=(12,3))
i=0

for c in healths_cat:
    if c == 'healthy': continue
    bee = bees[bees['health'] == c].iloc[0]
    f = bee['file']
    f_path= img_folder + f
    ax[i].imshow(imageio.imread(f_path))
    ax[i].set_title(bee['health'], fontsize=8)
    i += 1
plt.suptitle("Sick Bees")    
plt.tight_layout()
plt.show()
# Prepare train and test data
labels = pd.get_dummies(bees.subspecies, drop_first=True)
X = np.stack(bees.img)
train_data, test_data, train_labels, test_labels = train_test_split(X, labels)
# Build and train CNN model
model1=Sequential()
model1.add(Conv2D(5, kernel_size=3, input_shape=(img_width, img_height,3), activation='relu'))
model1.add(MaxPool2D(2))
model1.add(Conv2D(10, kernel_size=3, activation='relu'))
model1.add(Flatten())
model1.add(Dense(labels.columns.size, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

training = model1.fit(train_data, train_labels, validation_split=0.2, epochs=20, batch_size=10)
## Trained model analysis and evaluation
f, ax = plt.subplots(2,1, figsize=(5,5))
ax[0].plot(training.history['loss'])
ax[0].set_title('Detect kind of Bee: loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')

# Accuracy
ax[1].plot(training.history['acc'])
ax[1].set_title('Detect kind of Bee: accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
plt.tight_layout()
plt.show()

# Accuracy by subspecies
test_pred = model1.predict(test_data)
acc_by_subspecies = np.logical_and((test_pred > 0.5), test_labels).sum()/test_labels.sum()
acc_by_subspecies.plot(kind='bar', title='Subspecies prediction accuracy')
plt.ylabel('Accuracy')
plt.show()

# Loss function and accuracy
test_res = model1.evaluate(test_data, test_labels)
print('Evaluation: loss function: %s, accuracy:' % test_res[0], test_res[1])
# Prepare train and test data
labels = pd.get_dummies(bees.health)
X = np.stack(bees.img)
train_data, test_data, train_labels, test_labels = train_test_split(X, labels)

# Data augmentation - a little bit rotate, zoom and shift input images.
generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
generator.fit(train_data)

# Split train data to train and validation
train_data, train_data_val, train_labels, train_labels_val = train_test_split(train_data, 
                                                                              train_labels,
                                                                              test_size=0.1)  
# Build and train CNN model
model2 = Sequential()
model2.add(Conv2D(6, kernel_size=3, input_shape=(img_width, img_height,3), activation='relu'))
model2.add(MaxPool2D(2))
model2.add(Conv2D(12, kernel_size=3, activation='relu'))
model2.add(Flatten())
model2.add(Dense(labels.columns.size, activation='softmax'))
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# We'll stop training if no improvement after some epochs
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train
training = model2.fit_generator(generator.flow(train_data,train_labels, batch_size=20),
                               epochs = 20,
                               validation_data=(train_data_val, train_labels_val),
                               steps_per_epoch=20,  # batch_size
                               callbacks=[earlystopper])
f, ax = plt.subplots(2,1, figsize=(5,5))

# Loss function
ax[0].plot(training.history['loss'])
ax[0].set_title('Detect Bee health: loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')

# Accuracy
ax[1].plot(training.history['acc'])
ax[1].set_title('Detect Bee health: accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
plt.tight_layout()
plt.show()

# Prediction accuracy by health status
test_pred = model2.predict(test_data)
acc_by_health = np.logical_and((test_pred > 0.5), test_labels).sum()/test_labels.sum()
acc_by_health.plot(kind='bar', title='Health prediction accuracy')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

test_res = model2.evaluate(test_data, test_labels)
print('Evaluation: loss function: %s, accuracy:' % test_res[0], test_res[1])
# Common function for visualization of kernels
def visualize_layer_kernels(img, conv_layer, title):
    """
    Displays how input sample image looks after convolution by each kernel
    :param img: Sample image array
    :param conv_layer: Layer of Conv2D type
    :param title: Text to display on the top 
    """
    # Extract kernels from given layer
    weights1 = conv_layer.get_weights()
    kernels = weights1[0]
    kernels_num = kernels.shape[3]
    
    # Each row contains 3 images: kernel, input image, output image
    f, ax = plt.subplots(kernels_num, 3, figsize=(7, kernels_num*2))

    for i in range(0, kernels_num):
        # Get kernel from the layer and draw it
        kernel=kernels[:,:,:3,i]
        ax[i][0].imshow((kernel * 255).astype(np.uint8), vmin=0, vmax=255)
        ax[i][0].set_title("Kernel %d" % i, fontsize = 9)
        
        # Get and draw sample image from test data
        ax[i][1].imshow((img * 255).astype(np.uint8), vmin=0, vmax=255)
        ax[i][1].set_title("Before", fontsize=8)
        
        # Filtered image - apply convolution
        img_filt = scipy.ndimage.filters.convolve(img, kernel)
        ax[i][2].imshow((img_filt * 255).astype(np.uint8), vmin=0, vmax=255)
        ax[i][2].set_title("After", fontsize=8)
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()   
# Take sample image to visualize convolutoin
idx = random.randint(0,len(test_data)-1)
img = test_data[idx,:,:,:]
# Take 1st convolutional layer and look at it's filters
conv1 = model1.layers[0]
img = visualize_layer_kernels(img, conv1, "Subspecies CNN. Layer 0")

# Take sample image to visualize convolutoin
idx = random.randint(0,len(test_data)-1)
img = test_data[idx,:,:,:]
# Take another convolutional layer and look at it's filters
conv2 = model1.layers[2]
res = visualize_layer_kernels(img, conv2, "Subspecies CNN. Layer 2")
# Take sample image to visualize convolutoin
idx = random.randint(0,len(test_data)-1)
img = test_data[idx,:,:,:]
# Take 1st convolutional layer and look at it's filters
conv1 = model2.layers[0]
visualize_layer_kernels(img, conv1, "Health CNN layer 0")

# Take sample image to visualize convolutoin
idx = random.randint(0,len(test_data)-1)
img = test_data[idx,:,:,:]
# Take another convolutional layer and look at it's filters
conv2 = model2.layers[2]
visualize_layer_kernels(img, conv2, "Health CNN layer 2")
