# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf



train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

def train_mnist_conv(): 



    training_images=train.drop('label',axis=1)

    training_labels=train.label 

    

    training_images=training_images.values.reshape(-1, 28, 28, 1)

    training_images=training_images / 255.0



    model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),

            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(512, activation='relu'),

            tf.keras.layers.Dense(10, activation='softmax')

    ])



    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



    history = model.fit(

        training_images,training_labels,epochs=19

    )

    return history.epoch, history.history['accuracy'][-1],model
new_model = train_mnist_conv()
test_id=test.id



test_images=test.drop('id',axis=1)



test_images = test_images.values.reshape(-1, 28, 28, 1)

test_images=test_images/255.0



y_pre=new_model[2].predict(test_images)     ##making prediction

y_pre=np.argmax(y_pre,axis=1) ##changing the prediction intro labels



sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)



sample_sub.head()