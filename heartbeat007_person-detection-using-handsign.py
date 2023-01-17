#### load partial images
import cv2
import os
#### this function will import the data from the folder
def Otsus_Binarization(folder,target):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if img is not None:
            images.append([th2,target])
    return images


import json
with open('../input/person-name/person.json') as json_file:
    data = json.load(json_file)
for item in range(1,15):
    
    if item==1:
        one = Otsus_Binarization('../input/hand-data/1/',item)
    elif item==11:
            continue
    else:
        q = '../input/hand-data/{}/'.format(item)
        one.extend(Otsus_Binarization(q,item))
        
training_data = one
feature_matrix = []
target = []
for x,y in training_data:
    feature_matrix.append(x)
    target.append(y)
import matplotlib.pyplot as plt
feature_matrix
plt.imshow(feature_matrix[15],cmap='binary')
print("Preson : ",data[str(target[15])])
plt.imshow(feature_matrix[22],cmap="binary")
print("Person : ",data[str(target[22])])
X=[]
IMG_SIZE= 120
for x in feature_matrix:
    new_array = cv2.resize(x,(IMG_SIZE,IMG_SIZE))
    X.append(new_array)

Xx = []
for x in X:
    tmp = x/255
    Xx.append(tmp)
Xx
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Xx,target)
future_x = x_test
import numpy as np
print(np.array(x_train).shape)
x_train = np.array(x_train)
x_test = np.array(x_test)
img_rows=x_train[0].shape[0]
img_cols=x_train[0].shape[1]

X_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)

X_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
img_cols
X_train[0].shape
X_train.shape
X_test.shape
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(120, 120,1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(15, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
model =  define_model()
history = model.fit(np.array(X_train),np.array(y_train), epochs=50,validation_data=(X_test, np.array(y_test)))
from keras.utils.vis_utils import plot_model
plot_model(model)


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


predicted = model.predict(np.array(X_test))
model.evaluate(X_test,np.array(y_test))
result = []
for item in predicted:
    result.append(np.argmax(item))
result
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(result,y_test))
import json
predicted = []
actual = []
with open('../input/person-name/person.json') as json_file:
    data = json.load(json_file)
    for item in result:
        predicted.append(data[str(item)])
    for item in y_test:
        actual.append(data[str(item)])
        
import pandas as pd
df = pd.DataFrame(list(zip(predicted, actual)), 
               columns =['predieted', 'actual'])
df

for x,y,z in zip(future_x,actual,predicted):
    print("Actual : {}".format(y))
    print("Predicted : {}".format(z))
    print ("IMAGE :")
    plt.imshow(x,cmap="binary")
    plt.show()
    
    
