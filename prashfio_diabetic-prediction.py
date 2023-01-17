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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

#plt.style.use('ggplot')

#ggplot is R based visualisation package that provides better graphics with higher level of abstraction
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')



#Print the first 5 rows of the dataframe.

data.head()
data.shape
X = data.iloc[:, :-1].values

y = data.iloc[:, 8].values
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)
X
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras import regularizers
y.shape
y = keras.utils.to_categorical(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train
model = Sequential()

model.add(Dense(4, activation='relu', input_shape=(8,)))

model.add(Dense(4, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

              metrics=['accuracy'])
history = model.fit(X_train, y_train,

                    batch_size=10,

                    epochs=100,

                    verbose=1,

                    validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])