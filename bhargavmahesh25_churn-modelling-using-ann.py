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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
data.head()
data.shape
data.info()
data.isnull().sum()
x = data.drop("Exited", axis=1)
y = data['Exited']
x.shape
y.shape
data['Surname'].value_counts()
data['Geography'].value_counts()
sns.catplot('Geography', data=data, kind='count')

plt.xlabel("Country")

plt.ylabel("Number of People")

plt.title("Number of People for Each country")
data['Gender'].value_counts()
sns.catplot('Gender', data=data, kind='count')

plt.xlabel("Gender Category")

plt.ylabel("Number of People")

plt.title("Gender Classification")
data['HasCrCard'].value_counts()
sns.catplot('HasCrCard', data=data, kind='count')

plt.xlabel("Category")

plt.ylabel("Number of People")

plt.title("Identify People with Credit Card")
data['IsActiveMember'].value_counts()
sns.catplot('IsActiveMember', data=data, kind='count')

plt.xlabel("Category")

plt.ylabel("Number of People")

plt.title("Identify Active Members")
geography = pd.get_dummies(x['Geography'], drop_first=True)

gender = pd.get_dummies(x['Gender'], drop_first=True)
geography.shape
gender.shape
x.shape
x = pd.concat([x, geography, gender], axis=1)
x.shape
x.info()
x=x.drop(["Geography", "Gender", "Surname","RowNumber","CustomerId"], axis=1)
x.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
x_train.head()
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)
x_train.shape
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LeakyReLU, PReLU, ELU

from keras.layers import Dropout
model = Sequential()
model.add(Dense(units = 10, kernel_initializer = "he_normal", activation = "relu", input_dim = 11))

model.add(Dropout(0.3))
model.add(Dense(units=20, kernel_initializer="he_normal", activation = "relu"))

model.add(Dropout(0.4))
model.add(Dense(units=15, kernel_initializer="he_normal", activation = "relu"))

model.add(Dropout(0.2))
model.add(Dense(units=1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
model.summary()
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics =['accuracy'])
model_fit = model.fit(x_train, y_train, validation_split = 0.25, batch_size=10, epochs=100)
print(model_fit.history.keys())
plt.plot(model_fit.history['accuracy'])

plt.plot(model_fit.history['val_accuracy'])

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epochs")

plt.legend(['train','test'], loc = 'upper left')

plt.show()
plt.plot(model_fit.history['loss'])

plt.plot(model_fit.history['val_loss'])

plt.title("Model Accuracy")

plt.ylabel("Loss")

plt.xlabel("Epochs")

plt.legend(['train','test'], loc = 'upper left')

plt.show()
y_pred = model.predict(x_test)
y_pred
y_pred = (y_pred>0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

cm
acc_sc = accuracy_score(y_test, y_pred)

acc_sc