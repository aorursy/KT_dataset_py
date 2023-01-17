! cp ../input/my-python/* ../working/
! ls ../working 
# Generic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from tqdm import tqdm_notebook
%matplotlib inline
sns.set_style('whitegrid')

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Keras imports
from keras.applications import VGG16, InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing.image import load_img,img_to_array

# cyclical learning rates
from clr import LRFinder
from clr_callback import CyclicLR
# Constants
DATA_DIR = '../input/dog-breed-identification/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
BATCH_SIZE = 32
INPUT_SIZE = 224
NUM_CLASSES = 120
SEED = 42
# Let's check what's in the data directory
! ls $DATA_DIR
# Read the train data set, which has the ids of the images and their labels (breeds)
# (adding the extension .jpg to the id becomes the file name of the image) 
train = pd.read_csv(DATA_DIR + 'labels.csv')
train.head()
# The submission file contains one column for the image id, and then one column 
# each breed in alphabetical order, with the probability of the dog in the image beeing of that breed
submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
submission.head()
# Create a map of breeds to labels in the same order as the columns of the submission file
# and create a new column 'label' in the train data frame with the breeds mapped to this labels.
# This will make easier build the submission file from the predicted probabilities of the trained model
breed_labels = {breed:label for label,breed in enumerate(submission.columns[1:].values)}
train['label'] = train['breed'].map(breed_labels)
train.head()
# Frequency of each breed in the train set. We can see that the most frequent breed
# has just above 120 images and the less frequent just above 60 images.
counts = train.breed.value_counts()
plt.figure(figsize=(10,40))
plt.xticks(np.arange(0, 130, 5))
sns.barplot(x=counts.values, y=counts.index);
# Let's plot some random images
fig, axs = plt.subplots(5,5, figsize=(20,20), squeeze=True)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
axs = axs.reshape(-1)
indices = np.random.choice(train.shape[0], 25, replace=False)
for ax, i in zip(axs, indices):
    img = cv2.imread(os.path.join(DATA_DIR, 'train', train.iloc[i].id + '.jpg'))
    h, w, c = img.shape
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(train.iloc[i].breed, fontsize=12)
    ax.imshow(img)
    
base = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
base.summary()
# Make the batchsize of the data generator a divisor of the number of images, as we have
# to make just one pass for feature extraction
batch_size = 269           # 10222 = 2 * 19 * 269

# No data augmentation, just rescaling the image
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(dataframe=train, 
                                              directory= TRAIN_DIR,
                                              x_col='id',
                                              y_col='label',
                                              class_mode='categorical',
                                              has_ext=False,
                                              batch_size=batch_size,   
                                              shuffle=False,
                                              seed=42,
                                              target_size=(224,224)                                              
                                             )
%%time

# Let's read all the images and labels into arrays in memory

train_generator.reset()
train_size = train.shape[0]
features = np.zeros(shape=(train_size, 7,7,512))
labels = np.zeros(shape=(train_size, NUM_CLASSES))
i = 0
for inputs_batch, labels_batch in tqdm_notebook(train_generator):
    features_batch = base.predict(inputs_batch)
    features[i * batch_size:(i+1) * batch_size] = features_batch
    labels[i * batch_size:(i+1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= train_size:
        break;
   
# Flatten the output of the VGG16 base model
features = features.reshape(train_size, -1)
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.20, random_state=SEED)
# Build the classifier

def create_model(dropout=None):
    model = Sequential()
    model.add(Dense(1024, activation='relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'],  )
    return model

# Run a lr range test to find good learning rate margins

model = create_model()

BATCH_SIZE = 32
STEP_SIZE_TRAIN = X_train.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = X_val.shape[0] // BATCH_SIZE

EPOCHS = 1
base_lr=0.0001
max_lr=100
step_size = EPOCHS * STEP_SIZE_TRAIN 
lrf = LRFinder(X_train.shape[0], BATCH_SIZE,
                       base_lr, max_lr,
                       # validation_data=(X_val, Yb_val),
                       lr_scale='exp', save_dir='./lr_find/', verbose=False)

history = model.fit(X_train, y_train, epochs=EPOCHS, steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data=[X_val, y_val], validation_steps = STEP_SIZE_VALID,
                   callbacks=[lrf])
# Training
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=[X_val, y_val])
fig = plt.figure(figsize=(15,7))
lrf.plot_schedule(clip_beginning=95, clip_endding=60)
10**(-1.7)
model = create_model()
EPOCHS=10
BATCH_SIZE=32
clr = CyclicLR(base_lr=0.005, max_lr=0.02, step_size=2*STEP_SIZE_TRAIN)

history = model.fit(X_train, y_train, epochs=EPOCHS, steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=[X_val, y_val], validation_steps=STEP_SIZE_VALID,
                    callbacks=[clr])

def plt_history(history, metric, title, ax, val=True):
    ax.plot(history[metric])
    if val:
        ax.plot(history['val_' + metric])
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    
hist = history.history
fig, ax = plt.subplots(1,2, figsize=(15,6))
plt_history(hist, 'loss', 'LOSS', ax[0])
plt_history(hist, 'acc', 'ACCURACY', ax[1])
base = InceptionV3(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3), pooling='avg')
base.summary()
# Make the batchsize of the data generator a divisor of the number of images, as we have
# to make just one pass for feature extraction
batch_size = 269           # 10222 = 2 * 19 * 269

# No data augmentation, just rescaling the image
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(dataframe=train, 
                                              directory= TRAIN_DIR,
                                              x_col='id',
                                              y_col='label',
                                              class_mode='categorical',
                                              has_ext=False,
                                              batch_size=batch_size,   
                                              shuffle=False,
                                              seed=42,
                                              target_size=(224,224)                                              
                                             )
%%time

# Let's read all the images and labels into arrays in memory. We can use the same generator,
# but this time the features array will have shape (train_size, 512) instead of (train_size, 7*7*512)


train_generator.reset()
train_size = train.shape[0]
features = np.zeros(shape=(train_size, 2048))
labels = np.zeros(shape=(train_size, NUM_CLASSES))
i = 0
for inputs_batch, labels_batch in tqdm_notebook(train_generator):
    features_batch = base.predict(inputs_batch)
    features[i * batch_size:(i+1) * batch_size] = features_batch
    labels[i * batch_size:(i+1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= train_size:
        break;
   
# This time the features array doesn't need flattening

# Build the classifier. This time, the classifier has a lot fewer parameters
def create_model_pool(lr=0.001):
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    sgd = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'],  )
    return model
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.25, random_state=SEED)
# Run a lr range test to find good learning rate margins

model = create_model_pool()

BATCH_SIZE = 32
STEP_SIZE_TRAIN = X_train.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = X_val.shape[0] // BATCH_SIZE

EPOCHS = 1
base_lr=0.001
max_lr=1
step_size = EPOCHS * STEP_SIZE_TRAIN 
lrf = LRFinder(X_train.shape[0], BATCH_SIZE,
                       base_lr, max_lr,
                       validation_data=(X_val, y_val),
                       lr_scale='exp', save_dir='./lr_find/', verbose=False)

history = model.fit(X_train, y_train, epochs=EPOCHS, steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data=[X_val, y_val], validation_steps = STEP_SIZE_VALID,
                   callbacks=[lrf])
fig = plt.figure(figsize=(15,7))
lrf.plot_schedule(clip_beginning=50)
10**(-1.5)
# Training
model = create_model_pool()
EPOCHS=20
clr = CyclicLR(base_lr=0.01, max_lr=0.03, step_size=2*STEP_SIZE_TRAIN)

history = model.fit(X_train, y_train, epochs=EPOCHS, steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=[X_val, y_val], validation_steps=STEP_SIZE_VALID,
                    callbacks=[clr])

hist = history.history
fig, ax = plt.subplots(1,2, figsize=(15,6))
plt_history(hist, 'loss', 'LOSS', ax[0])
plt_history(hist, 'acc', 'ACCURACY', ax[1])