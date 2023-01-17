



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('accuracy')>0.98):

            print('\n reached desired accuracy so cancelling training ')

            self.model.stop_training=True

            

            

x_train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
label=x_train.label

pixels=x_train.iloc[:,1:]

pixels.head()
img=np.array(pixels[0:1]).reshape((28,28))
plt.imshow(img, cmap='gray')

plt.show()

# preparing data

y_train=x_train["label"]

x_train=x_train.drop("label",axis=1)

print(x_train.head())

print(y_train.head())

callbacks = myCallback()

model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(input_shape=(28,28)),

        tf.keras.layers.Dense(512,activation=tf.nn.relu),

        tf.keras.layers.Dense(10,activation=tf.nn.softmax)

        

    ])
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



#fit model

model.fit(x_train,y_train,epochs=15,callbacks=[callbacks])
x_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

x_test=pd.DataFrame(x_test)
image=np.array(x_test.iloc[0])

image=image.reshape((28,28))

plt.imshow(image, cmap='gray')

plt.show()
prediction=model.predict(x_test)

prediction = [np.argmax(y,axis=None,out=None)for y in prediction]

prediction
sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
sub["Label"] = prediction

sub.head()
sub.to_csv("submmision.csv",index=False)