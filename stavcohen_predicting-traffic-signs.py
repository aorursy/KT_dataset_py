# Libraries 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import tensorflow as tf

import cv2

import matplotlib.image as mpimg

import os

from PIL import Image

import seaborn as sns

from sklearn.model_selection import  train_test_split

from keras.utils import to_categorical







#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



print(os.listdir('../input/gtsrb-german-traffic-sign'))


df=pd.read_csv('../input/gtsrb-german-traffic-sign/Train.csv')

# finding the best values for Height and Width

print(df.head())

print(df['Height'].value_counts()[:5].sort_values(ascending=False))

print(df['Width'].value_counts()[:5].sort_values(ascending=False))

height=33

width=33



# examing the pictures

data_dir = "../input/gtsrb-german-traffic-sign"

img_path= list((data_dir + '/' + str(df.Path[i])) for i in range(len(df.Path)))

for i in range(0,9):

    plt.subplot(331+i)

    seed=np.random.randint(0,29222)

    img= mpimg.imread(img_path[seed])

    plt.imshow(img)

    

plt.show()



data=[]

labels=[]



classes = 43

n_inputs = height * width*3



for i in range(classes) :

    path = "../input/gtsrb-german-traffic-sign/train/{0}/".format(i)

    print(path)

    Class=os.listdir(path)

    for a in Class:

        try:

            image=cv2.imread(path+a)

            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((height, width))

            data.append(np.array(size_image))

            labels.append(i)

        except AttributeError:

            print(" ")

            

train_data=np.array(data)

labels=np.array(labels)



data=np.arange(train_data.shape[0])

train_data=train_data[data]

labels=labels[data]
plt.figure(figsize=(20,5))

sns.countplot(labels)

plt.title('Pictures per Label', fontsize = 20)

plt.xlabel('Labels', fontsize=20)

plt.show()





x_train, x_val, y_train, y_val = train_test_split(train_data, labels , test_size = 0.2, random_state = 68)

print("Train :", x_train.shape)

print("Valid :", x_val.shape)

x_train = x_train.astype('float32')/255 

x_val = x_val.astype('float32')/255



y_train = to_categorical(y_train, 43)

y_val = to_categorical(y_val, 43)

class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self,epoch,logs={}):

    if logs.get('accuracy') is not None and logs.get('accuracy') > 0.985:

      print("\n reached 98.5% accuracy so canceling training!")

      self.model.stop_training=True
#Definition of the DNN model



model = tf.keras.models.Sequential([

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(x_train.shape[1:])),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

     tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(43, activation='softmax')

])



model.compile(

    loss='categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy']

)

model.summary()
epochs = 20

callbacks=myCallback()

history = model.fit(x_train, y_train, batch_size=32, epochs=epochs,

validation_data=(x_val, y_val),callbacks=[callbacks])





plt.figure(0)

plt.plot(history.history['accuracy'], label='training accuracy')

plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.figure(1)

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Model Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()
y_test=pd.read_csv("../input/gtsrb-german-traffic-sign/Test.csv")

labels=y_test['Path']

y_test=y_test['ClassId']

y_test = to_categorical(y_test,43)



data=[]



for f in labels:

    image=cv2.imread('../input/gtsrb-german-traffic-sign/test/'+f.replace('Test/', ''))

    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image_from_array = Image.fromarray(image, 'RGB')

    size_image = image_from_array.resize((height, width))

    data.append(np.array(size_image))



X_test=np.array(data)

X_test = X_test.astype('float32')/255 



results = model.evaluate(X_test, y_test, batch_size=128)



print("test loss, test acc:", results)

image=cv2.imread('../input/stopsign2/stop.jpg')







data=[]

image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image_from_array = Image.fromarray(image, 'RGB')

size_image = image_from_array.resize((height, width))

data.append(np.array(size_image))



X_test=np.array(data)

X_test = X_test.astype('float32')/255     

pred=model.predict_classes(X_test)

print(pred)

print(model.predict(X_test)[0][14])



plt.imshow(size_image)



plt.show()
image=cv2.imread('../input/stopsign/.png')







data=[]

image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image_from_array = Image.fromarray(image, 'RGB')

size_image = image_from_array.resize((height, width))

data.append(np.array(size_image))



X_test=np.array(data)

X_test = X_test.astype('float32')/255     

pred=model.predict_classes(X_test)

print(pred)

print(model.predict(X_test)[0][13])



plt.imshow(size_image)



plt.show()