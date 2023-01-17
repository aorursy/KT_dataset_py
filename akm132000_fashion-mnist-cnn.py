import numpy as np 
import pandas as pd 
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from collections import Counter
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
common_path='/kaggle/input/fashionmnist'
train_df=pd.read_csv(os.path.join(common_path,'fashion-mnist_train.csv'))
test_df=pd.read_csv(os.path.join(common_path,'fashion-mnist_test.csv'))
print('train dataframe shape:',train_df.shape)
print('test dataframe shape:',test_df.shape)
y_train=np.array(train_df['label'])
train_df.drop('label',axis=1,inplace=True)
y_test=np.array(test_df['label'])
test_df.drop('label',axis=1,inplace=True)
def getLabels(num):
    if (num==0):
        return 'T-shirt/top'
    elif (num==1):
        return 'Trouser'
    elif(num==2):
        return 'Pullover'
    elif(num==3):
        return 'Dress'
    elif(num==4):
        return 'Coat'
    elif(num==5):
        return 'Sandal'
    elif(num==6):
        return 'Shirt'
    elif(num==7):
        return 'Sneaker'
    elif(num==8):
        return 'Bag'
    else:
        return 'Ankle boot'

train_labels=[]
test_labels=[]
for elem in y_train:
    train_labels.append(getLabels(elem))
    
for elem in y_test:
    test_labels.append(getLabels(elem))
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
class_wise_freq=Counter(train_labels)
sns.barplot(x=list(class_wise_freq.keys()),y=list(class_wise_freq.values()))
plt.xticks(rotation=40)
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Train Data Class Wise Counts')

plt.subplot(2,2,2)
class_wise_freq=Counter(test_labels)
sns.barplot(x=list(class_wise_freq.keys()),y=list(class_wise_freq.values()))
plt.xticks(rotation=40)
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Test Data Class Wise Counts')
plt.show()
x_train=train_df.values
x_test=test_df.values
x_train=x_train/255
x_test=x_test/255
x_train_small,x_validate,y_train_small,y_validate=train_test_split(x_train,y_train)
fig,axis=plt.subplots(4,4)
count=0
for row in range(4):
    for col in range(4):
        axis[row][col].imshow(x_train_small[count].reshape(28,28))
        axis[row][col].set_title(train_labels[count])
        count+=1
plt.show()
num_classes=10
img_width=28
img_height=28
num_channels=1

img_shape=(img_width,img_height,num_channels)

# convert class vectors to binary class matrices
y_train_small_categorical = keras.utils.to_categorical(y_train_small, num_classes)
y_validate_categorical = keras.utils.to_categorical(y_validate, num_classes)
y_test_categorical=keras.utils.to_categorical(y_test,num_classes)
fig,axis=plt.subplots(4,4)
count=0
for row in range(4):
    for col in range(4):
        axis[row][col].imshow(x_train_small[count].reshape(28,28))
        axis[row][col].set_title(train_labels[count])
        count+=1
plt.show()
x_train_small.shape,x_validate.shape
x_train_small=x_train_small.reshape(-1,img_width,img_height,num_channels)
x_validate=x_validate.reshape(-1,img_width,img_height,num_channels)
x_test=x_test.reshape(-1,img_width,img_height,num_channels)
x_train_small.shape,x_validate.shape,x_test.shape
cnn_model = keras.Sequential(
    [
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",input_shape=img_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(32,activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
cnn_model.summary()
cnn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history=cnn_model.fit(x_train_small, y_train_small_categorical,epochs=40,validation_data=(x_validate,y_validate_categorical))
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Vs Epochs')

plt.subplot(2,2,2)

plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Vs Epochs')

plt.show()
x_test.shape,y_test.shape
score=cnn_model.evaluate(x_test,y_test_categorical)
print('Test Loss:',round(score[0],2))
print('Test Accuracy:',round(score[1],2))
y_test_predicted=cnn_model.predict_classes(x_test)
print(classification_report(y_test,y_test_predicted))
print(confusion_matrix(y_test,y_test_predicted))