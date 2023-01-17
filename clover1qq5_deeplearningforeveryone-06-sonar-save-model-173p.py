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
from keras.models import Sequential, load_model

from keras.layers.core import Dense



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
import tensorflow as tf
seed = 0

np.random.seed(seed)

tf.random.set_seed(3)

df = pd.read_csv('../input/sonarcsv/sonar.csv', header = None)



dataset = df.values

X = dataset[:, 0:60]

Y_obj = dataset[:, 60]



e = LabelEncoder()

e.fit(Y_obj)

Y=e.transform(Y_obj)
dataset = df.values

X = dataset[:, 0:60]

Y_obj = dataset[:, 60]



e = LabelEncoder()

e.fit(Y_obj)

Y=e.transform(Y_obj)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()

model.add(Dense(24, input_dim=60, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])



X_train = np.asarray(X_train).astype(np.float32)

Y_train = np.asarray(Y_train).astype(np.float32)


model.fit(X_train, Y_train, epochs=130, batch_size=5)

model.save('my_model.h5')

del model

model = load_model('my_model.h5')





X_test = np.asarray(X_test).astype(np.float32)

Y_test = np.asarray(Y_test).astype(np.float32)




print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))