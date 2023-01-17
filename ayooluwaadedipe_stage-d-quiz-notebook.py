#Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import fbeta_score
from tqdm import tqdm
import cv2
from PIL import Image
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import fbeta_score
import time
%matplotlib inline

pal = sns.color_palette()

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#Load train and test CSVs
df_train = pd.read_csv('../input/planets-dataset/planet/planet/train_classes.csv')
df_test = pd.read_csv('../input/planets-dataset/planet/planet/sample_submission.csv')
#Explore train labels distribution
labels = df_train['tags'].apply(lambda x: x.split(' '))
from collections import Counter, defaultdict
counts = defaultdict(int) #dictionary containing each individual label
for l in labels:
    for l2 in l:
        counts[l2] += 1

# data=[go.Bar(x=list(counts.keys()), y=list(counts.values()))]
# layout=dict(height=800, width=800, title='Distribution of training labels')
# fig=dict(data=data, layout=layout)
# py.iplot(data, filename='train-label-dist')
# plt.show()
tag_list=list(counts.keys()) 
y=list(counts.values())
sns.barplot(x=tag_list, y=y);
plt.xlabel('labels');
plt.xticks(rotation = 90);
plt.title('Tag count for train set');
#Explore test labels distribution
labels_test = df_test['tags'].apply(lambda x: x.split(' '))
from collections import Counter, defaultdict
counts_test = defaultdict(int)
for l in labels_test:
    for l2 in l:
        counts_test[l2] += 1

tag_list_test=list(counts_test.keys()) 
test_count=list(counts_test.values())
sns.barplot(x=tag_list_test, y=test_count);
plt.xlabel('labels');
plt.xticks(rotation = 90);
plt.title('Tag counts for test set');

#These are not actual labels, just placeholders
#View some of the train images

new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20, 20))
i = 0
for f, l in df_train[:9].values:
    img = cv2.imread('../input/planets-dataset/planet/planet/train-jpg/{}.jpg'.format(f))
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))
    #ax[i // 4, i % 4].show()
    i += 1
    
plt.show()
#Create a dictionary assigning a numerical value to each label
label_map = {i:j for j, i in enumerate(tag_list)}
label_map
# One hot encode the training labels. Convert the images into pixels and resize them
X_train, Y_train = [], []
for img, label in tqdm(df_train.values, miniters = 1000):
  target = np.zeros(17)
  for tag in label.split(' '):
    target[label_map[tag]]=1
  X_train.append(cv2.resize(cv2.imread('../input/planets-dataset/planet/planet/train-jpg/{}.jpg'.format(img)), (64,64)))
  Y_train.append(target)
#convert the test images to pixels and resize them as well
X_test=[]
for img, label in tqdm(df_test[:40669].values, miniters = 1000):
  X_test.append(cv2.resize(cv2.imread('../input/planets-dataset/planet/planet/test-jpg/{}.jpg'.format(img)), (64,64)))
for img, label in tqdm(df_test[40669:].values, miniters = 1000):
  X_test.append(cv2.resize(cv2.imread('../input/planets-dataset/test-jpg-additional/test-jpg-additional/{}.jpg'.format(img)), (64,64)))
#Confirm the dimensions
len(X_test), len(X_train), len(Y_train)
import sys
sys.getsizeof(X_train)
#Change lists to numpy arrays and normalize
x_train = np.array(X_train, np.float16)/255
y_train = np.array(Y_train, np.uint8)
#x_test = np.array(X_test, np.float16)/255

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 1)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
del(X_test, X_train, Y_train)
#Split the data into 5 folds and train on 4 folds while validating on 1 fold
# nfolds = 5
# num_fold = 0
# sum_score = 0
yfull_test = []
yfull_train = []
X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)
# kf = KFold(n_splits = nfolds, shuffle = True, random_state = 1)
# print(kf)

#Train a convolutional neural network using KFold cross validation to prevent overfitting
# for train_index, test_index in kf.split(x_train):
#     start_time_model_fitting = time.time()
        
# X_train_ = x_train[train_index]
# Y_train_ = y_train[train_index]
# X_val_ = x_train[test_index]
# Y_val_ = y_train[test_index]

#     #Track the progress through the K folds
#     num_fold += 1
#     print('Start KFold number {} from {}'.format(num_fold, nfolds))
#     print('Split train: ', len(X_train_), len(Y_train_))
#     print('Split valid: ', len(X_val_), len(Y_val_))
    

        
#Build a five layer CNN model

#Create a path to save the weights
kfold_weights_path = os.path.join('', 'weights_kfold_' + '.h5')
model = Sequential()
model.add(BatchNormalization(input_shape=(64, 64,3)))
model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
        
model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
        
model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
        
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

    #Try a combination of epoch lengths and learning rates
epochs = 20
learn_rate = 0.0001
opt  = optimizers.Adam(lr=learn_rate)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]
#    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)] #save the weights of the best performing model

model.fit(x = X_train_, y= Y_train_, validation_data=(X_val_, Y_val_),batch_size=128,verbose=2, epochs=epochs,callbacks=callbacks,shuffle=True)
        
p_val = model.predict(X_val_, batch_size = 32, verbose=2)
print(fbeta_score(Y_val_, np.array(p_val) > 0.2, beta=2, average='samples')) #Check the model performance on the validation set

p_train = model.predict(x_train, batch_size =128, verbose=2) #save the training predictions
yfull_train.append(p_train)
        
p_test = model.predict(x_test, batch_size = 128, verbose=2) #save the test predictions
yfull_test.append(p_test)

result = np.array(yfull_test[0])
# for i in range(1, nfolds):
#     result += np.array(yfull_test[i])
# result /= nfolds
result = pd.DataFrame(result, columns = labels)
result
# base model. Feel free to try out other architectures and ideas to improve fbeta score

import numpy as np
from keras import backend as K


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())



from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, RMSprop


model = keras.Sequential()
model.add(Conv2D(64, 5, 2, activation = "relu", input_shape = (64, 64, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, 2, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dense(17, activation = "sigmoid"))

model.compile(loss = "binary_crossentropy", optimizer = Adam(), metrics = [fbeta])
model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 20, batch_size = 128)
#convert the test images to pixels and resize them as well
X_test=[]
for img, label in tqdm(df_test[:40669].values, miniters = 1000):
  X_test.append(cv2.resize(cv2.imread('../input/planets-dataset/planet/planet/test-jpg/{}.jpg'.format(img)), (64,64)))
for img, label in tqdm(df_test[40669:].values, miniters = 1000):
  X_test.append(cv2.resize(cv2.imread('../input/planets-dataset/test-jpg-additional/test-jpg-additional/{}.jpg'.format(img)), (64,64)))

x_test = np.array(X_test, np.float16)/255
predictions = model.predict(x_test, batch_size = 128)
predictions
pred = pd.DataFrame(predictions, columns =  tag_list)
pred
tag_list
