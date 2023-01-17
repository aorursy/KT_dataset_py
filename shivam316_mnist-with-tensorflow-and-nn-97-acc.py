# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X=train.drop('label',axis=1)

y=train.label
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
from sklearn.model_selection import train_test_split

Xtrain,Xval,ytrain,yval=train_test_split(X,y,test_size=0.2,random_state=42)
k=keras.backend

k.clear_session()

np.random.seed(42)

tf.random.set_seed(42)
early_stopppin_cb=keras.callbacks.EarlyStopping(patience=20)

checkpoint_cb=keras.callbacks.ModelCheckpoint('my_mnist_viz_dl12.h5',save_best_only=True)
def exponential_decay(lr0,s):

    def exp_decay_fn(epoch):

        return lr0*0.1**(epoch/s)

    return exp_decay_fn

exp_decay_fn=exponential_decay(0.01,20)

lr=keras.callbacks.LearningRateScheduler(exp_decay_fn)
model=keras.models.Sequential([

    keras.layers.Flatten(input_shape=[1,784]),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(rate=0.2),

    keras.layers.Dense(300,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.01),use_bias=False),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(rate=0.2),

    keras.layers.Activation('elu'),

    keras.layers.Dense(150,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.01),use_bias=False),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(rate=0.2),

    keras.layers.Activation('elu'),

    keras.layers.Dense(10,activation='softmax')

])
opt=keras.optimizers.Nadam()

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
def save_show_model_metrics(metrics,run_index):

    path=os.path.join(os.curdir,"Run_{:03d}".format(run_index))

    os.mkdir(path, mode = 0o777)

    txt=os.path.join(path,"Metrics_{:03d}.txt".format(run_index))

    png=os.path.join(path,"Metrics_{:03d}.png".format(run_index))

    f=open(txt,'w')

    f.write("Total Epochs: {0}   Loss: {1}   Accuracy: {2}   Val_Loss: {3}   Val_Accuracy: {4}".format(max(metrics.epoch)+1,metrics.history['loss'][-1],metrics.history['accuracy'][-1],metrics.history['val_loss'][-1],metrics.history['val_accuracy'][-1]))

    f.close()

    pd.DataFrame(metrics.history).plot(figsize=(8,5))

    plt.grid(True)

    plt.gca().set_ylim(-0.5,1.5)

    plt.savefig(png)

    plt.show()
metrics=model.fit(Xtrain,ytrain,epochs=400,validation_data=(Xval,yval),callbacks=[lr,early_stopppin_cb,checkpoint_cb])
save_show_model_metrics(metrics,15)
model.save('my_mnist_dl_97-15.h5')
Label=model.predict_classes(test)

ImageId=pd.Series(range(1,28001))

Label=pd.Series(Label)

sol=pd.concat([ImageId, Label],axis=1)

sol=sol.rename(columns={0: "ImageId", 1: "Label"})

sol.to_csv('mnist_via_nn97_14.csv',index=False)