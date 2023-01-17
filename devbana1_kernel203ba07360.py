# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import layers

from tensorflow.keras import Sequential



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk('/kaggle/output/Kannada-MNIST/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

y = raw_data['label']

y = to_categorical(y)

raw_data.drop('label',axis=1,inplace=True)
y
x = np.array(raw_data)

x = x.reshape(x.shape[0],28,28,1)

x = x/255
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
model_cnn_ken = Sequential()

model_cnn_ken.add(layers.Conv2D(128,kernel_size=(3,3),input_shape=(28,28,1), activation='relu'))

model_cnn_ken.add(layers.MaxPool2D(pool_size=(2,2)))

model_cnn_ken.add(layers.Conv2D(64,kernel_size=(2,2), activation='relu'))

model_cnn_ken.add(layers.MaxPool2D(pool_size=(2,2)))

model_cnn_ken.add(layers.Conv2D(20,kernel_size=(2,2), activation='relu'))

model_cnn_ken.add(layers.Flatten())

model_cnn_ken.add(layers.Dense(50,activation='relu'))

model_cnn_ken.add(layers.Dense(10,activation='softmax'))

model_cnn_ken.compile(optimizer='Adam',metrics=['acc'],loss='categorical_crossentropy')

model_cnn_ken.fit(x_train,y_train,batch_size=500, epochs=10)

model_cnn_ken.evaluate(x_test,y_test)
raw_data_y = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
raw_data_y.drop('id',axis=1,inplace=True)
test_data = np.array(raw_data_y)

test_data = test_data.reshape(test_data.shape[0],28,28,1)

test_data = test_data/255
y_pred = model_cnn_ken.predict_classes(test_data)
sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
sub['label'] = y_pred
sub2 = sub.loc[:,['id','label']]
sub2.columns
sub2.to_csv('submission.csv')