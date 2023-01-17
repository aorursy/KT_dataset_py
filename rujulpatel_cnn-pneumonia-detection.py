import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2

import tensorflow as tf
from matplotlib import pyplot as plt
data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/'

train_dir = data_dir + 'train/'
test_dir = data_dir + 'test/'
val_dir = data_dir + 'val/'
from glob import glob

normal_img = glob(train_dir + "NORMAL/*.jpeg")
pneumonia_img = glob(train_dir + "PNEUMONIA/*.jpeg")

plt.figure(figsize=(12,8))

plt.subplot(1,2,1)
plt.imshow(cv2.imread(normal_img[0]))
plt.title("Normal")


plt.subplot(1,2,2)
plt.imshow(cv2.imread(pneumonia_img[0]))
plt.title("Pneumonia")
labels = ['NORMAL','PNEUMONIA']

#Image dimensions
img_size = 200


def read_data(image_path):
    print("Reading from Directory : ",image_path)
    
    #read, reshape and save images to dataframe,
    #label according to subfoldera
    X = []
    y = [] #0 -> NORMAL & 1 -> PNEUMONIA
    
    
    for label in labels:
        print("Subfolder : ", label)
        
        img_cnt = 0;
        path = image_path + label
        
        #Read all images from
        for image in os.listdir(path):
            
            img = cv2.imread(path + '/' + image,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(img_size,img_size))
                
            X.append(img)
            y.append(labels.index(label))
            
            img_cnt += 1
        pass        
            
        print("Done. Read ",img_cnt," Images")
    
    ## Normalize
    X = np.array(X)/255
    X = X.reshape(-1,img_size,img_size,1)
    y = np.array(y)
        
    return X,y    
X_train,y_train = read_data(train_dir)
X_test,y_test = read_data(test_dir)
X_val,y_val = read_data(val_dir)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(5,5),activation='relu',input_shape=(img_size,img_size,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(48,(5,5),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
#Train Model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=10,batch_size=50,validation_data=(X_val,y_val),shuffle=True)
model.summary()
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Test Accuracy')
plt.legend()
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.legend()
score = model.evaluate(X_test,y_test)
print('Loss : ',score[0],' Accuracy ',round(score[1]*100,2)," %")
y_pred = model.predict(X_test)
y_pred = np.where(y_pred>0.5,1,0)
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

sns.heatmap(cm,annot=True,fmt='g',xticklabels=labels,yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
