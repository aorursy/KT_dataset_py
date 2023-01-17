# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd

train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape)
ntrain = train.shape[0]
print(test.shape)
ntest = train.shape[0]
train.head()
print(train.dtypes[:5])
print(test.dtypes[:5])
ytrain = train['label']
print('Shape of ytrain', ytrain.shape)
xtrain = train.drop('label',axis=1)
from math import sqrt
dim = int(sqrt(xtrain.shape[1]))
print('The images are {}x{} squares.'.format(dim, dim))
print('shape of xtrain:',xtrain.shape)
ytrain.head()
import seaborn as sns
sns.set(style='white',context='notebook',palette='deep')
sns.countplot(ytrain)
print(ytrain.shape)
print(type(ytrain))
vals_class = ytrain.value_counts()
print(vals_class)
cls_mean = np.mean(vals_class)
cls_std = np.std(vals_class, ddof=1)
print('The mean amount of elements per class is', cls_mean)
print('The Standard deviation in the element per class distribution is', cls_std)

if cls_std > cls_mean * (0.687/2):
    print('The standard deviation is high')
    

def check_nan(df):
    print(df.isnull().any().describe())
    print('There are missing values' if df.isnull().any().any() else "There are no missing values")
    
    if df.isnull().any().any():
        print(df.isnull().sum(axis=0))
    
    print()
    
check_nan(xtrain)
check_nan(test)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
xtrain_vis = xtrain.values.reshape(ntrain, dim, dim)

for i in range(0,9):
    plt.subplot(330 + (i+1))
    plt.imshow(xtrain_vis[1], cmap = plt.get_cmap('gray'))
    plt.title(ytrain[i]);
xtrain = xtrain/255.0
test = test/255
def df_reshape(df):
    print('Previous shape, pixels are in 1D vector', df.shape)
    df = df.values.reshape(-1, dim , dim , 1)
    print('After reshape, pixels are a 28*28*1 3D matrix', df.shape)
    return df

xtrain = df_reshape(xtrain)
test = df_reshape(test)
from keras.utils.np_utils import to_categorical

print(type(ytrain))

nclasses = ytrain.max() - ytrain.min() + 1
print('Shape of ytrain before:', ytrain.shape)

ytrain = to_categorical(ytrain, num_classes = nclasses)
print('Shape pf ytrain after:', ytrain.shape)

print(type(ytrain))
from sklearn.model_selection import train_test_split

seed = 2
np.random.seed(seed)

split_pct = 0.1

xtrain, xval, ytrain, yval = train_test_split(xtrain,
                                              ytrain,
                                              test_size=split_pct,
                                              random_state=seed,
                                              shuffle = True,
                                              stratify = ytrain
                                              )
print(xtrain.shape, ytrain.shape, xval.shape, yval.shape)
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import AveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
nets = 5
model = [0]*nets
for i in range(nets):
    model[i] = Sequential()
    model[i].add(Conv2D(32, kernel_size=3, activation = 'relu',input_shape=(28,28,1)))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(32, kernel_size=3, activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(32, kernel_size=5, strides=2, padding='same',activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.4))
    
    model[i].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(64, kernel_size=5, strides = 2, padding='same', activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.4))
    
    model[i].add(Conv2D(128, kernel_size=4, activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Flatten())
    model[i].add(Dropout(0.4))
    model[i].add(Dense(10, activation='softmax'))
    model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = [0] * nets
epochs = 45
for j in range(nets):
    print('CNN', j+1)
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(xtrain , ytrain , test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2, Y_train2, batch_size=64),
                                       epochs = epochs, 
                                       steps_per_epoch = X_train2.shape[0]//64,
                                        validation_data = (X_val2, Y_val2), callbacks = [annealer], verbose=1)
    print('CNN {0:d}: Epochs = {1:d}, Train_accuracy={2:.5f}, Validation accuracy = {3:.5f}'.format(
                                        j+1, epochs, max(history[j].history['acc']), max(history[j].history['val_acc']) ))
results = np.zeros( (test.shape[0],10) ) 
for j in range(nets):
    results = results + model[j].predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("predictions.csv",index=False)
