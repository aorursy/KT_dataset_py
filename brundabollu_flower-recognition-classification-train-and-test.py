
import os
from glob import glob

#For basic operations
import pandas as pd
import numpy as np
from random import randint

#Used for training
import tensorflow as tf
import math
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, recall_score, precision_score, f1_score, auc, accuracy_score

#used for visual representation
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
DATASET_PATH="/kaggle/input/flowers-recognition/flowers/" #Path where dataset stored
classes = ['daisy', 'rose', 'dandelion', 'sunflower', 'tulip'] 
classes
data_x=[]
data_y=[]
w, h = 80, 80

center = (w / 2, h / 2)
 
angle90 = 90
angle180 = 180
angle270 = 270
 
scale = 1.0

#Basic Preprocession
for c in tqdm(classes):
    _list = glob(DATASET_PATH+c+"/*.jpg")
    for name in _list:
        img = cv2.resize(plt.imread(name), (w, h))/255
        data_x.append(img)
        data_y.append(c)
        
        M = cv2.getRotationMatrix2D(center, angle90, scale)
        rotated90 = cv2.warpAffine(img, M, (h, w))
        data_x.append(img)
        data_y.append(c)
        
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated180 = cv2.warpAffine(img, M, (w, h))
        data_x.append(img)
        data_y.append(c)
        
        M = cv2.getRotationMatrix2D(center, angle270, scale)
        rotated270 = cv2.warpAffine(img, M, (w, h))
        data_x.append(img)
        data_y.append(c)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=[w, h, 3]))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(64, (3,3)))
# model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())

model.add(tf.keras.layers.Conv2D(128, (3,3)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())

model.add(tf.keras.layers.Conv2D(128, (3,3)))
# model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())

model.add(tf.keras.layers.Conv2D(64, (3,3)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())

model.add(tf.keras.layers.Conv2D(32, (3,3)))
# model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(len(classes), activation=None))
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
def step(real_x, real_y):
    with tf.GradientTape() as tape:
        pred_y = model(np.reshape(real_x, newshape=[-1, w, h, 3]))
        loss = tf.nn.softmax_cross_entropy_with_logits(real_y, pred_y)
    model_grad = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(model_grad, model.trainable_variables))
    return loss

def calc_metrics(pred, test_y):
    _test_y, _pred = np.argmax(test_y.values, axis=1), tf.math.argmax(pred, axis=1).numpy()
    _f1 = f1_score(_test_y,_pred ,  average="weighted")
    _recall = recall_score(_test_y,_pred ,  average="weighted")
    _precision = precision_score(_test_y,_pred ,  average="weighted")
    _acc = accuracy_score(_test_y,_pred)
    return _f1, _recall, _precision, _acc

def show_metrics(test_x, test_y):
    batch_size=32
    bat_per_epoch = math.floor(len(train_x) / batch_size)
    f1_list = []
    pre_list = []
    rec_list = []
    acc_list = []
    for i in range(bat_per_epoch):
        pred = tf.nn.softmax(model(np.array(test_x)[i:i+batch_size], training=False))
        f1,rec,pre,acc = calc_metrics(pred, test_y.iloc[i:i+batch_size])
        f1_list.append(f1)
        rec_list.append(rec)
        pre_list.append(pre)
        acc_list.append(acc)
    return round(np.mean(f1_list), 4), round(np.mean(pre_list), 4), round(np.mean(rec_list), 4), round(np.mean(acc_list), 4)

def return_probabs(test_x, test_y):
    batch_size=32
    bat_per_epoch = math.floor(len(test_x) / batch_size)
    pred_list = []
    target_list = []
    for i in range(bat_per_epoch):
        pred = tf.nn.softmax(model(np.array(test_x)[i:i+batch_size], training=False))
        target_list+=list(test_y.values[i:i+batch_size])
        pred_list+=list(pred)
    return np.array(target_list), np.array(pred_list)
train_x, train_y = shuffle(data_x, data_y)
train_x, train_y, test_x, test_y = (train_x[0:int(len(train_x)*0.9)], pd.get_dummies(train_y[0:int(len(train_x)*0.9)]), 
                                    train_x[int(len(train_x)*0.9):], pd.get_dummies(train_y[int(len(train_x)*0.9):]))
print("Train Data len: ", len(train_x))
print("Test Data len: ", len(test_x))
batch_size=32
bat_per_epoch = math.floor(len(train_x) / batch_size)
# _train_y = pd.get_dummies(np.array(train_y))
epochs = 6
for epoch in range(epochs):
    train_x, train_y = shuffle(train_x, train_y)
    _loss_list = []
    for i in range(bat_per_epoch):
        n = i*batch_size
        _loss = step(np.array(train_x[n:n+batch_size]), np.array(train_y[n:n+batch_size]))
        _loss_list.append(_loss.numpy().mean())
        
    print("Epoch: ", epoch+1, " loss: ", np.mean(_loss_list))
    train_x_data, train_y_data = shuffle(train_x, train_y)
    train_f1,train_pre,train_rec,train_acc = show_metrics(train_x_data[:1000], train_y_data.iloc[:1000])
    print(" \t Train: f1_score:", train_f1, " precision:", train_pre, " recall:",train_rec, " Accuracy:", train_acc)
    
    test_f1,test_pre,test_rec,test_acc = show_metrics(test_x, test_y)
    print(" \t Test: f1_score:", test_f1, " precision:", test_pre, " recall:",test_rec, " Accuracy:", test_acc)
    
    print()
plt.figure(figsize=[15,15])
for i in range(16):
    idx=randint(0, len(test_x))
    pred = tf.nn.softmax(model(np.array(test_x)[idx:idx+1], training=False))
    test_label, pred_label = test_y.columns[np.argmax(test_y.values[idx])], test_y.columns[tf.math.argmax(pred[0]).numpy()]
    plt.subplot(4,4,i+1)
    plt.imshow(test_x[idx])
    _=plt.title("True: "+test_label+" | pred:"+pred_label)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import DenseNet121
flower_types=['dandelion','daisy','tulip','rose','sunflower']
data_dir = '../input/flowers-recognition/flowers/'
train_dir = os.path.join(data_dir)
train_data = []
for flower_id, sp in enumerate(flower_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), flower_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'FlowerId','Flower Type'])
train.tail()
# Randomize the order of training set
SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()
# Plot a histogram
plt.hist(train['FlowerId'])
plt.title('Frequency Histogram of Flower Types')
plt.figure(figsize=(12, 12))
plt.show()
def plot_defects(flower_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(12, 12))
    flower_files = train['File'][train['Flower Type'] == flower_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, flower_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1
# Displays first n images of class from training set
plot_defects('tulip', 5, 5)
IMAGE_SIZE = 64

def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)
X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X_Train = X_train / 255
print('Train Shape: {}'.format(X_Train.shape))
Y_train = train['FlowerId'].values
Y_train = to_categorical(Y_train, num_classes=5)
BATCH_SIZE = 64

# Split the train and validation sets 
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_train, test_size=0.2, random_state=SEED)
fig, ax = plt.subplots(1, 4, figsize=(15, 15))
for i in range(4):
    ax[i].set_axis_off()
    ax[i].imshow(X_train[i])
    ax[i].set_title(flower_types[np.argmax(Y_train[i])])