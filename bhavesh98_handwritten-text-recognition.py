import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import cv2
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import TensorBoard
%matplotlib inline
#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 

sns.set()

dataset = pd.read_csv("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv").astype('float32')
dataset.rename(columns={'0':'label'}, inplace=True)
train_d = pd.read_csv('../input/digit-recognizer/train.csv')
test_d = pd.read_csv('../input/digit-recognizer/test.csv')
# Splite data the X - Our data , and y - the prdict label
X_l = dataset.drop('label',axis = 1)
y_l = dataset['label']
print("shape:",X_l.shape)
print("culoms count:",len(X_l.iloc[1]))
print("784 = 28X28")

X_l.head()
training_d = np.array(train_d, dtype = 'float32')
testing_d = np.array(test_d, dtype = 'float32')

training_d = np.array(train_d, dtype = 'float32')
testing_d = np.array(test_d, dtype = 'float32')
i = random.randint(0, len(training_d))
plt.imshow(d_image, cmap = 'Greys')
#plt.axis('off')
label = training_d[i, 0]
x = []
w_grid = 5
l_grid = 2
fig, axes = plt.subplots(l_grid, w_grid, figsize = (20, 8))
axes = axes.ravel()
i = 0
while len(x) < 10:
    index = np.random.randint(0, len(training_d))
    if training_d[index, 0] not in x:
        axes[i].imshow(training_d[index, 1:].reshape((28, 28)), cmap = 'Greys') 
        x.append(training_d[index, 0])
        i += 1
from sklearn.utils import shuffle

X_shuffle = shuffle(X_l)

plt.figure(figsize = (12,10))
row, colums = 4, 4
for i in range(16):  
    plt.subplot(colums, row, i+1)
    plt.imshow(X_shuffle.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')
    #plt.imshow(test_input,interpolation='nearest', cmap='Greys')
plt.show()
print("Amount of each labels")

# Change label to alphabets
alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
dataset_alphabets = dataset.copy()
dataset['label'] = dataset['label'].map(alphabets_mapper)

label_size = dataset.groupby('label').size()
label_size.plot.barh(figsize=(10,10))
plt.show()

X_train_d = training_d[:, 1:]
y_train_d = training_d[:, 0]

X_train_d, X_validate_d, y_train_d, y_validate_d = train_test_split(X_train_d, y_train_d, test_size = 0.2, random_state = 0)

X_train_d = X_train_d.reshape(X_train_d.shape[0], *(28, 28, 1))
X_validate_d = X_validate_d.reshape(X_validate_d.shape[0], *(28, 28, 1))
# splite the data
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l,y_l)

# scale data
standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train_l)

X_train_l = standard_scaler.transform(X_train_l)
X_test_l = standard_scaler.transform(X_test_l)
print("Data after scaler")
X_shuffle = shuffle(X_train_l)

plt.figure(figsize = (12,10))
row, colums = 4, 4
for i in range(16):  
    plt.subplot(colums, row, i+1)
    plt.imshow(X_shuffle[i].reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()
X_train_l = X_train_l.reshape(X_train_l.shape[0], 28, 28, 1).astype('float32')
X_test_l = X_test_l.reshape(X_test_l.shape[0], 28, 28, 1).astype('float32')

y_train_l = np_utils.to_categorical(y_train_l)
y_test_l = np_utils.to_categorical(y_test_l)
#For Digits 
cnn = Sequential()
cnn.add(Conv2D(128, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(output_dim = 32, activation = 'relu'))
cnn.add(Dense(output_dim = 10, activation = 'sigmoid'))
cnn.compile(optimizer= Adam(lr = 0.001), loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])
epochs = 20
cnn.fit(X_train_d, y_train_d, batch_size = 512, epochs = epochs, validation_data = (X_validate_d, y_validate_d), verbose = 1,)

scores = cnn.evaluate(X_validate_d, y_validate_d, verbose=0)

print("CNN Score for digits:",scores[1])
#For Letters
cls = Sequential()
cls.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
cls.add(MaxPooling2D(pool_size=(2, 2)))
cls.add(Dropout(0.3))
cls.add(Flatten())
cls.add(Dense(128, activation='relu'))
cls.add(Dense(len(y.unique()), activation='softmax'))

cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = cls.fit(X_train_l, y_train_l, validation_data=(X_test_l, y_test_l), epochs=18, batch_size=200, verbose=2)

scores = cls.evaluate(X_test_l,y_test_l, verbose=0)
print("CNN Score for letters:",scores[1])
cm=confusion_matrix(y_test_l.argmax(axis=1),cls.predict(X_test_l).argmax(axis=1))
df_cm = pd.DataFrame(cm, range(26),range(26))
plt.figure(figsize = (20,15))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
check = int(input('Enter the index number to check : '))
digit_check = X_train_d[check].reshape(1, 28, 28, 1).astype('float32')
digit_out = cnn.predict(digit_check).argmax(axis=1)
print(digit_out)
alpha_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
toPre = X_train_l[check].reshape(1, 28, 28, 1).astype('float32')
toOut = cls.predict(toPre).argmax(axis=1)
print(alpha_mapper[toOut[0]])
plt.figure(figsize = (12, 10))
row, colums = 4, 4
plt.imshow(X_train_d[check].reshape(28, 28),cmap='Greys')
#plt.axis('off')
plt.show

plt.figure(figsize = (12,10))
row, colums = 4, 4
plt.imshow(X_train_l[check].reshape(28,28), cmap='Greys')
#plt.axis('off')
plt.show
d_image = cv2.imread('../input/handwriten-digit-test/4test.png')
gray_d = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
gray_d = cv2.resize(gray_d, (28, 28))
test_input_d = cv2.bitwise_not(gray_d)
test_input_d.shape
digit_out = cnn.predict(test_input_d.reshape(1, 28, 28, 1)).argmax(axis=1)
print(digit_out)
  
image = cv2.imread('../input/handwritten-testing/MTest.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayImage = cv2.resize(grayImage, (28, 28))
test_input = cv2.bitwise_not(grayImage)
standard_scaler.fit(test_input)
alpha_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
toOut = cls.predict(test_input.reshape(1,28,28,1)).argmax(axis=1)
print(alpha_mapper[toOut[0]])