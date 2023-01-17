import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
from scipy.misc import imresize, imread
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
%matplotlib inline
print("don")
# Create a new directory
base_dir = 'data'
os.mkdir(base_dir)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]


# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'test')
os.mkdir(val_dir)


# [CREATE FOLDERS INSIDE THE TRAIN, VALIDATION AND TEST FOLDERS]
# Inside each folder we create seperate folders for each class

# create new folders inside train_dir
one = os.path.join(train_dir, '1')
os.mkdir(one)

zero = os.path.join(train_dir, '0')
os.mkdir(zero)
# create new folders inside val_dir
one = os.path.join(val_dir, '1')
os.mkdir(one)

zero = os.path.join(val_dir, '0')
os.mkdir(zero)
from glob import glob
imagePatches = glob('/kaggle/input/IDC_regular_ps50_idx5/**/*_class0.png', recursive=True)
len(imagePatches)
from glob import glob
imagePatches = glob('/kaggle/input/IDC_regular_ps50_idx5/**/*_class1.png', recursive=True)
len(imagePatches)
os.listdir('/kaggle/working/data/train/1')
from glob import glob
import os
import cv2
from pathlib import Path    
imagePatches = glob('/kaggle/input/IDC_regular_ps50_idx5/**/*_class1.png', recursive=True)
for filename in imagePatches[0:60000]:
    #print(filename)
    cv_img = cv2.imread(filename)
    name='/kaggle/working/data/train/1/'+Path(filename).name
   # print(x)
    cv2.imwrite(name, cv_img)
    
     
count=0
for filename in imagePatches[60000:78786]:
    cv_img = cv2.imread(filename)
    name='/kaggle/working/data/test/1/'+Path(filename).name
    count+=1
    print(count)
    cv2.imwrite(name, cv_img)
    
print('don')
from glob import glob
import os
import cv2
from pathlib import Path    
imagePatches = glob('/kaggle/input/IDC_regular_ps50_idx5/**/*_class0.png', recursive=True)
for filename in imagePatches[0:180000]:
    #print(filename)
    cv_img = cv2.imread(filename)
    name='/kaggle/working/data/train/0/'+Path(filename).name
   # print(x)
    cv2.imwrite(name, cv_img)
    
     
count=0
for filename in imagePatches[190000:198738]:
    cv_img = cv2.imread(filename)
    name='/kaggle/working/data/test/0/'+Path(filename).name
    count+=1
    print(count)
    cv2.imwrite(name, cv_img)
    
print('don')
image_name = "/kaggle/input/IDC_regular_ps50_idx5/9135/1/9135_idx5_x1701_y1851_class1.png" #Image to be used as query
def plotImage(image_location):
    image = cv2.imread(image_name)
    image = cv2.resize(image, (50,50))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis('off')
    return
plotImage(image_name)
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


# dimensions of our images
img_width, img_height = 50, 50

# load the model we saved
model = load_model('model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# predicting images
img = image.load_img('/kaggle/input/IDC_regular_ps50_idx5/12935/1/12935_idx5_x2001_y1201_class1.png', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print(classes)

