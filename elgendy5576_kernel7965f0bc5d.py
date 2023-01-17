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
#reading data 

train_data=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

#splitiing data

x_train =train_data.drop('label',axis =1)

y_train = train_data['label']
#normalize data

x_train =x_train/255.0

x_train=np.array(x_train).reshape(-1,28,28,1)

test_data=np.array(test_data).reshape(-1,28,28,1)

from keras.utils.np_utils import to_categorical

y_train = to_categorical((y_train), num_classes = 10)
import tensorflow as tf

model = tf.keras.models.Sequential([

 

      tf.keras.layers.Conv2D(64,(3,3),activation ='relu',input_shape=(28,28,1)),

        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam', loss='categorical_crossentropy' , metrics=['acc'])

print('done')
model.fit(x_train ,y_train ,epochs = 5 )
test_data=test_data.astype(float)

res=model.predict(test_data)

res
submissions=pd.DataFrame({"ImageId": list(range(1,len(res)+1)),

                         "Label":res})

submissions.to_csv("elgendy.csv", index=False, header=True)

submissions