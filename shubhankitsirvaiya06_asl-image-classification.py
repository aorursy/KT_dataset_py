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
train_path='../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'
test_path='../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn 
import numpy as np
import os
import cv2
def show_image():
    size_img=(100,100)
    image_for_plot=[]
    label_for_plot=[]
    for folder in os.listdir(train_path):
        for file in os.listdir(train_path+'/'+folder):
            filepath=train_path+'/'+folder+'/'+file
            image=cv2.imread(filepath)
            final_img=cv2.resize(image,size_img)
            final_img=cv2.cvtColor(final_img,cv2.COLOR_BGR2RGB)
            image_for_plot.append(final_img)
            label_for_plot.append(folder)
            break
    return image_for_plot,label_for_plot
image_for_plot,label_for_plot=show_image()
print('total unique labels= ',label_for_plot)
fig=plt.figure(figsize=(15,15))

def plot_img(fig,image,labels,rows,col,index):
    fig.add_subplot(rows,col,index)
    plt.imshow(image)
    plt.title(labels)    

image_index = 0
rows = 5
col = 6
for i in range(1,rows*col):
    plot_img(fig,image_for_plot[image_index],label_for_plot[image_index],rows,col,i)
    image_index=image_index+1
plt.show
labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'space':26,'del':27,'nothing':28}
def load():
    
    size_img=(100,100)
    images=[]
    label=[]
    for folder in os.listdir(train_path):
        for image in os.listdir(train_path+'/'+folder):
            filepath=train_path+'/'+folder+'/'+image
            temp_image=cv2.imread(filepath)
            temp_image=cv2.resize(temp_image,size_img)
            
            images.append(temp_image)
            label.append(labels_dict[folder])
    images=np.array(images)
    images=images.astype('float32')/255
    label = keras.utils.to_categorical(label)
    X_train, X_test, Y_train, Y_test = train_test_split(images, label, test_size = 0.05)
    return X_train, X_test, Y_train, Y_test
X_train, X_test, Y_train, Y_test = load()




import tensorflow as tf
from sklearn.model_selection import train_test_split
train_data,val_data=train_test_split(train_generator,test_size=.2)
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(29,activation='softmax')
])
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.fit(X_train,Y_train,epochs=20,validation_split=.2)