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
CLASS = {
    'Black-grass': 0,
    'Charlock': 1,
    'Cleavers': 2,
    'Common Chickweed': 3,
    'Common wheat': 4,
    'Fat Hen': 5,
    'Loose Silky-bent': 6,
    'Maize': 7,
    'Scentless Mayweed': 8,
    'Shepherds Purse': 9,
    'Small-flowered Cranesbill': 10,
    'Sugar beet': 11
}

dim = 64
sample_sub = pd.read_csv("../input/sample_submission.csv")
sample_sub.head(10)
import imageio
from skimage.transform import resize as imresize
from tqdm import tqdm

# fill train and test dict
def fill_dict(paths, some_dict):
    text = ''
    if 'train' in paths[0]:
        text = 'Start fill train_dict'
    elif 'test' in paths[0]:
        text = 'Start fill test_dict'

    for p in tqdm(paths, ascii=True, ncols=85, desc=text):
        img = imageio.imread(p)
        img = img_reshape(img)
        some_dict['image'].append(img)
        some_dict['label'].append(img_label(p))
        if 'train' in paths[0]:
            some_dict['class'].append(img_class(p))

    return some_dict


# Resize all image to 51x51 
def img_reshape(img):
    img = imresize(img, (dim, dim, 3)) # already normalizes? /255?
    return img

# get image tag
def img_label(path):
    return str(str(path.split('/')[-1]))

# get plant class on image
def img_class(path):
    return str(path.split('/')[-2])

file_ext = []
train_path = []
test_path = []

for root, dirs, files in os.walk('../input'):
    if dirs != []:
        print('Root: ' + str(root))
        print('Dirs: ' + str(dirs))
    else:
        for f in files:
            ext = os.path.splitext(str(f))[1][1:]

            if ext not in file_ext:
                file_ext.append(ext)

            if 'train' in root:
                path = os.path.join(root, f)
                train_path.append(path)
                
            elif 'test' in root:
                path = os.path.join(root, f)
                test_path.append(path)
train_dict = {
    'image': [],
    'label': [],
    'class': []
}
test_dict = {
    'image': [],
    'label': []
}

train_dict = fill_dict(train_path, train_dict)
test_dict = fill_dict(test_path, test_dict)
train_dict['image'][:5]
file_ext
train_path[:10]
from keras.utils import to_categorical

xtrain = np.array(train_dict['image'])
_ytrain = np.array([CLASS[l] for l in train_dict['class']])
ytrain = to_categorical(np.array([CLASS[l] for l in train_dict['class']]))
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

# plot how many images there are in each class
sns.countplot(_ytrain)

print(_ytrain.shape)
print(type(_ytrain))
__ytrain = pd.Series(_ytrain)

# array with each class and its number of images
vals_class = __ytrain.value_counts()
print(vals_class)

# mean and std
cls_mean = np.mean(vals_class)
cls_std = np.std(vals_class,ddof=1)

print("The mean amount of elements per class is", cls_mean)
print("The standard deviation in the element per class distribution is", cls_std)

# 68% - 95% - 99% rule, the 68% of the data should be cls_std away from the mean and so on
# https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
if cls_std > cls_mean * (0.6827 / 2):
    print("The standard deviation is high")
    
# if the data is skewed then we won't be able to use accurace as its results will be misleading and we may use F-beta score instead.
xtest = np.array(test_dict['image'])
label = test_dict['label']
xtrain[:5]
xtrain.shape # 4750, 64, 64, 3
xtrain[:5]
ytrain.shape # (4750, 12)
nclasses = 12

from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# percentage of xtrain which will be xval
split_pct = 0.05

# Split the train and the validation set
xtrain, xval, ytrain, yval = train_test_split(xtrain,
                                              ytrain, 
                                              test_size=split_pct,
                                              random_state=seed,
                                              stratify=ytrain
                                             )

print(xtrain.shape)
print(xval.shape)
print(ytrain.shape)
print(yval.shape)
from keras import backend as K

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D
model = Sequential()

ksize = 5

model.add(Conv2D(filters=32, kernel_size=(ksize,ksize), padding='same', activation='relu', input_shape=(dim,dim,3)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(ksize,ksize), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(ksize,ksize), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nclasses, activation='softmax'))
model.summary()
# Compile the model
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(xtrain)
epochs = 35
batch_size = 64
# Fit the model
history = model.fit_generator(datagen.flow(xtrain,ytrain, batch_size=batch_size),
                              epochs=epochs, validation_data=(xval,yval),
                              verbose=1, steps_per_epoch=xtrain.shape[0]//batch_size, 
                              callbacks=[learning_rate_reduction])
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

from sklearn.metrics import confusion_matrix
import itertools

# Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
ypred = model.predict(xval)
# Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2
ypred_classes = np.argmax(ypred,axis=1)
# Convert validation observations from one hot vectors to labels
ytrue = np.argmax(yval,axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(ytrue, ypred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(nclasses))
INV_CLASS = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet'
}
predictions = model.predict_classes(xtest, verbose=1)
sub = pd.DataFrame({"file": label,
                    "species": [INV_CLASS[p] for p in predictions]})
sub.head(10)
sub.to_csv("plant0708.csv", index=False, header=True)
