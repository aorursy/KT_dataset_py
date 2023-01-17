



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

check = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train.head()
train.info()
sns.countplot(train['Embarked'])
#Importing LabelEncoder from Sklearn

from sklearn.preprocessing import LabelEncoder

label_encoder_sex = LabelEncoder()

# Transforming sex column values using label Encoder

train.iloc[:,3] = label_encoder_sex.fit_transform(train.iloc[:,3])



test.iloc[:,1] = label_encoder_sex.fit_transform(test.iloc[:,1])
X_train = train.iloc[:,0:5]   # Inputs

y_train = train.iloc[:,5]     # Output
import keras 

from keras.models import Sequential

from keras.layers import Dense
from keras import layers

from keras import models

classifier = Sequential()

classifier.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

classifier.add(layers.Dense(10, activation='softmax'))
from keras.optimizers import SGD

sgd = SGD(lr = 0.01, momentum = 0.9)



classifier.compile(optimizer = 'sgd',

loss = 'categorical_crossentropy',

metrics = ['accuracy'])
classifier.summary()