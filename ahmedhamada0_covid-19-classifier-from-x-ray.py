import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

import os

import seaborn as sn

import glob as gb

import cv2

import tensorflow as tf

import keras

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

### for Kaggle

trainpath = '../input/covid19-chest-xray-dataset/training/'

testpath  = '../input/covid19-chest-xray-dataset/test/'

predpath  = '../input/covid19-chest-xray-dataset/pred/'



### for Jupyter

# trainpath = ''

# testpath = ''

# predpath = ''
for folder in  os.listdir(trainpath) : 

    files = gb.glob(pathname= str( trainpath + folder + '/*.jpg'))

    print(f'For training data , found {len(files)} in folder {folder}')
for folder in  os.listdir(testpath) : 

    files = gb.glob(pathname= str( testpath + folder + '/*.jpg'))

    print(f'For testing data , found {len(files)} in folder {folder}')
files = gb.glob(pathname= str(predpath +'rand/*.jpg'))

print(f'For Prediction data , found {len(files)}')
code = {'COVID-19':0, 'SARS':1, 'ARDS':2, 'Pneumocystis':3, 'Streptococcus':4, 'Normal':5}



def getcode(n) : 

    for x , y in code.items() : 

        if n == y : 

            return x    
size = []

for folder in  os.listdir(trainpath) : 

    files = gb.glob(pathname= str( trainpath + folder + '/*.jpg'))

    for file in files: 

        image = plt.imread(file)

        size.append(image.shape)

pd.Series(size).value_counts()
size = []

for folder in  os.listdir(testpath) : 

    files = gb.glob(pathname= str( testpath + folder + '/*.jpg'))

    for file in files: 

        image = plt.imread(file)

        size.append(image.shape)

pd.Series(size).value_counts()
#s = 512

s = 300
X_train = []

y_train = []

for folder in  os.listdir(trainpath) : 

    files = gb.glob(pathname= str( trainpath + folder + '/*.jpg'))

    for file in files: 

        image = cv2.imread(file)

        image_array = cv2.resize(image , (s,s))

        X_train.append(list(image_array))

        y_train.append(code[folder])
print(f'we have {len(X_train)} items in X_train')
plt.figure(figsize=(20,20))

for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 

    plt.subplot(6,6,n+1)

    plt.imshow(X_train[i])   

    plt.axis('off')

    plt.title(getcode(y_train[i]))
X_test = []

y_test = []

for folder in  os.listdir(testpath) : 

    files = gb.glob(pathname= str(testpath + folder + '/*.jpg'))

    for file in files: 

        image = cv2.imread(file)

        image_array = cv2.resize(image , (s,s))

        X_test.append(list(image_array))

        y_test.append(code[folder])

        
print(f'we have {len(X_test)} items in X_test')
plt.figure(figsize=(20,20))

for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 

    plt.subplot(6,6,n+1)

    plt.imshow(X_test[i])    

    plt.axis('off')

    plt.title(getcode(y_test[i]))
X_pred = []

files = gb.glob(pathname= str(predpath + 'rand/*.jpg'))

for file in files: 

    image = cv2.imread(file)

    image_array = cv2.resize(image ,(s,s))

    X_pred.append(list(image_array))       
print(f'we have {len(X_pred)} items in X_pred')
plt.figure(figsize=(20,20))

for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 

    plt.subplot(6,6,n+1)

    plt.imshow(X_pred[i])    

    plt.axis('off')
X_train = np.array(X_train)

X_test = np.array(X_test)

X_pred_array = np.array(X_pred)

y_train = np.array(y_train)

y_test = np.array(y_test)



print(f'X_train shape  is {X_train.shape}')

print(f'X_test shape  is {X_test.shape}')

print(f'X_pred shape  is {X_pred_array.shape}')

print(f'y_train shape  is {y_train.shape}')

print(f'y_test shape  is {y_test.shape}')
KerasModel = keras.models.Sequential([

        keras.layers.Conv2D(32, border_mode='same', kernel_size=(5,5),activation='relu',input_shape=(s,s,3)),

        keras.layers.Conv2D(64, border_mode='same', kernel_size=(5,5),activation='relu'),

        keras.layers.MaxPool2D(4,4),

        keras.layers.Conv2D(128, border_mode='same', kernel_size=(3,3),activation='relu'),    

        keras.layers.Conv2D(256, border_mode='same', kernel_size=(3,3),activation='relu'),    

        keras.layers.MaxPool2D(2,2),

        keras.layers.Flatten(),    

        keras.layers.Dense(300,activation='relu'),

        keras.layers.Dropout(rate=0.5),

        keras.layers.Dense(6,activation='softmax'),

        ])
KerasModel.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print('Model Details are : ')

print(KerasModel.summary())
epochs = 150

ThisModel = KerasModel.fit(X_train, y_train, epochs=epochs, batch_size=4, verbose=1, validation_split=0.2, shuffle=True)
#  "Accuracy"

plt.plot(ThisModel.history['accuracy'])

plt.plot(ThisModel.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()





# "Loss"

plt.plot(ThisModel.history['loss'])

plt.plot(ThisModel.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()





ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)





print('Test Loss is {}'.format(ModelLoss))

print('Test Accuracy is {}'.format(ModelAccuracy))





np.save("CNN_model.npy",KerasModel)

# serialize weights to HDF5

KerasModel.save_weights("CNN_model.h5")

print("Saved model to disk")
y_pred = KerasModel.predict(X_test, verbose=0)

yhat_probs = KerasModel.predict_classes(X_test, verbose=0)

print(y_test)

print(yhat_probs)

print('=========================================')



f1_score(y_test, yhat_probs, average='weighted')

# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_test, yhat_probs)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(y_test, yhat_probs ,average='weighted')

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(y_test, yhat_probs, average='weighted')

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, yhat_probs, average='weighted')

print('F1 score: %f' % f1)



print('=========================================')

print('Prediction Shape is {}'.format(y_pred.shape))

print('test Shape is {}'.format(y_test.shape))



cn = confusion_matrix(y_test, yhat_probs)

sn.heatmap(cn, center = True)

plt.show()
y_result = KerasModel.predict(X_pred_array)



print('Prediction Shape is {}'.format(y_result.shape))
plt.figure(figsize=(20,20))

for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 

    plt.subplot(6,6,n+1)

    plt.imshow(X_pred[i])    

    plt.axis('off')

    plt.title(getcode(np.argmax(y_result[i])))