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
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mlxtend.evaluate import confusion_matrix  # to plot the confusion matrix
from mlxtend.plotting import plot_confusion_matrix # to plot the confusion matrix
train = pd.read_csv('../input/digit-recognizer/train.csv',sep=',')
test = pd.read_csv('../input/digit-recognizer/test.csv',sep=',')
print(train.shape)
print(test.shape)
Y_train= train.label
X_train = train.iloc[:,1:]
X_test = test
# Checking for null value in data if any
X_train.isnull().any().describe()
X_test.isnull().any().describe()
# Plotting the count of each digit in dataset
plt.figure(figsize=(7,7))
sns.countplot(Y_train)
Y_train.value_counts()
# Reshapping the data
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = np.array(Y_train)
# Normalising the data
X_train = X_train/255.0
X_test = X_test/255.0
# Plot random 4 images from training data of MNIST data set
import random
ncol = 2
nrow = 2
fig,ax = plt.subplots(nrow,ncol)
fig.set_size_inches(5,5)

for num in range(4):
  m = random.choice(range(len(Y_train)))
  plt.subplot(nrow,ncol,num+1)
  plt.imshow(X_train[m][:,:,0])
# Make train and validation set
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size = 0.1,random_state =1)
# Model in CNN
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(5,5),activation='relu',input_shape=(28,28,1)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Conv2D(128,(5,5),activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Conv2D(256,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Dense(512,activation='relu'),
                                    tf.keras.layers.Dense(10,activation='softmax')])
model.summary()
# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Data Agumentation
epochs = 30
batch_size = 32

data_gen  = ImageDataGenerator(width_shift_range=0.05,
                               height_shift_range =0.05,
                               horizontal_flip = False,
                               shear_range = 0.1,
                               vertical_flip = False,
                               rotation_range=5,
                               fill_mode='nearest')

data_gen.fit(X_train)

train_generator = data_gen.flow(X_train,Y_train,batch_size= batch_size)

val_generator = data_gen.flow(X_val,Y_val,batch_size= batch_size)

history = model.fit_generator(train_generator, epochs = epochs,steps_per_epoch= int(X_train.shape[0]/batch_size),
                              validation_data = (val_generator),verbose = 2)
epochs = range(epochs)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs,train_acc,'b',label='train_acc')
plt.plot(epochs,val_acc,'r',label='val_acc')
plt.title('train_acc vs val_acc')
plt.legend()

plt.figure()
plt.plot(epochs,train_loss,'b',label='train_loss')
plt.plot(epochs,val_loss,'r',label='val_loss')
plt.title('train_loss vs val_loss')
plt.legend()

Y_pred = model.predict(X_val)
Y_pred = np.argmax(Y_pred,axis = 1)

# Plotting the confusion matrix
cm = confusion_matrix(Y_pred,Y_val,binary=False)
fig,ax = plot_confusion_matrix(conf_mat=cm)
fig.set_size_inches(8,8)
plt.show()
# Identify index number where digits are missclassified
Y_val = np.array(Y_val)
X_val = np.array(X_val)
index = np.where(Y_val-Y_pred != 0)
print("Missclassified images:{}".format(len(index[0])))
# Display random 9 numbers where our model classified it wrong
rows = 3
cols = 3
f1,a1 = plt.subplots(rows,cols)
fig.set_size_inches(10,10)
for num in range(rows*cols):
  choice = random.choice(range(len(index[0])))
  i = index[0][choice]
  a1 = plt.subplot(rows,cols,num+1)
  plt.imshow(X_val[i][:,:,0])
  a1.title.set_text('True value {},\n Predicted value {}'.format(Y_val[i],Y_pred[i]))
  plt.tight_layout()
# Visualizing across different Convolution and Maxpooling layer
m = X_train.shape[0]
rand_img = random.choice(range(m+1)) 
nr = 2
nc =3
conv_no = 1 
f2,a2 = plt.subplots(nr,nc)
#fig.set_size_inches(10,10)
layers=[]                             
for la in model.layers:                   
  layers.append(la.output)         
input_output = tf.keras.models.Model(inputs=model.input,outputs=layers)

# Iterate over 3 Conv + 3 Maxpool Layers = 6
for i in range(6):
  a2 = plt.subplot(nr,nc,i+1)
  layer_predict = input_output.predict(X_train[rand_img].reshape(1,28,28,1))[i]
  plt.imshow(layer_predict[0,:,:,conv_no])
  a2.title.set_text('layer:{}'.format(i+1))
  plt.tight_layout()

print("Output label is : {}".format(Y_train[rand_img]))
# Submission
prediction = model.predict(X_test)

prediction = np.argmax(prediction,axis =1)

prediction = pd.Series(prediction,name='Label')

submission = pd.Series(range(1,28001),name='ImageId')

submission = pd.concat([submission,prediction],axis=1)

submission.to_csv("submission.csv",index=False)