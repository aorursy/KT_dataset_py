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
import pandas as pd

import seaborn

import numpy as np

import matplotlib.pyplot as plt 

from sklearn import datasets, preprocessing, metrics, model_selection, pipeline

seaborn.set_style("white")



%matplotlib inline
import keras 

import tensorflow as tf

#load data

digits=datasets.load_digits()

X=digits.data

Y=digits.target
#standardization of the data

#axis=0 -> column

#axis=1 -> row

X_mean=np.mean(X,axis=1,keepdims=True)

X_std=np.std(X,axis=1,keepdims=True)

X=(X-X_mean)/X_std
def plot_images(x, **kwargs):

    n_pix = int(np.sqrt(np.prod(X.shape[1:]))) #assumes images are square

    im_indices = np.random.choice(len(x), 20)

    fig, axes = plt.subplots(nrows=4,ncols=5, figsize=(5,5), sharex=True, sharey=True, frameon=False)

    for i,ax in enumerate(axes.flat):

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

        curr_i = im_indices[i]

        ax.imshow(x[curr_i].reshape(n_pix,n_pix), aspect="equal", **kwargs)

        ax.axis('off')
plot_images(X, cmap="gray", interpolation="spline16")
#split the data to test and train data

from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_keras_train=X_train.reshape(1437,64,1)

X_keras_test=X_test.reshape(360,64,1)

# Create a Keras Sequential model

# We do this by passing a list of layers to the Sequential model



model = keras.Sequential([

  #  keras.layers.InputLayer(input_shape=X.shape[1:]),

  #  keras.layers.Flatten(),

  # keras.layers.Conv1D(64,kernel_size=3,input_shape=X.shape[1:]),

    keras.layers.Conv1D(filters=64,kernel_size=2,input_shape=(64,1)),

    keras.layers.Dropout(0.5),

    keras.layers.MaxPool1D(pool_size=2),

    keras.layers.Flatten(),

    keras.layers.Dense(100, activation="relu"),

    keras.layers.Dense(100 ,activation="relu"),

    keras.layers.Dense(64 ,activation="relu"),

    keras.layers.Dense(32, activation="relu"),

    keras.layers.Dense(32 ,activation="relu"),

    

    #keras.layers.Flatten(),

    keras.layers.Dense(10, activation="softmax")

])



model.summary() #summary provides an at-a-glance look at the model we've built
acc_list=[]

# Compile the network

model.compile("nadam", "sparse_categorical_crossentropy", metrics=["acc"])

model.fit(X_keras_train,Y_train,epochs=30,

    validation_split=0.25, verbose=0,callbacks=[ keras.callbacks.ReduceLROnPlateau(

        factor=.5, patience=1, min_lr=1e-7, verbose=0),

    keras.callbacks.EarlyStopping(patience=4, verbose=0)])

 



acc= model.evaluate(X_keras_test,Y_test)

acc_list.append(acc)
print(acc_list)
#test our data

sample=X_keras_test[0]

ss=sample.reshape(1,64,1)

ss.shape

predi=model.predict(ss)

prediY=predi.argmax()
print("predicted :",prediY)

print("True : ",Y_test[0])