# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
import seaborn as sns
sns.pairplot(df,hue = "species")
df.head()
X = df.drop('species', axis = 1)
X.head()
target_names = df['species'].unique()
target_names
target_dict = {n:i for i, n in enumerate(target_names)}
target_dict
y = df['species'].map(target_dict)
y.head()
from keras.utils.np_utils import to_categorical
y_cat = to_categorical(y)
y_cat[:10]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat,test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
model = Sequential()
model.add(Dense(3,input_shape = (4,), activation = 'softmax'))
model.compile(Adam(lr = 0.1), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train , epochs = 20, validation_split = 0.1)
y_pred = model.predict(X_test)
y_pred[:5]
y_test_class = np.argmax(y_test,axis =1)
y_pred_class = np.argmax(y_pred,axis =1)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
confusion_matrix(y_test_class,y_pred_class)

