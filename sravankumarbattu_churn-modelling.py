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
dataset=pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
dataset.head()
dataset.isnull().any()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataset['Geography'] = le.fit_transform(dataset['Geography'])

dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset.head()
x = dataset.iloc[:,3:13] 

y =  dataset.iloc[:,13] 
x = dataset.iloc[:,3:13].values 

y =  dataset.iloc[:,13] .values
from sklearn.preprocessing import OneHotEncoder

one = OneHotEncoder()

z = one.fit_transform(x[:,1:2]).toarray()

x = np.delete(x,1,axis = 1)

x = np.concatenate((z,x),axis = 1)
x.shape
x  = x[:,1:]
x.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

x_train.shape
y_train.shape
x_test.shape
y_test.shape
x_train
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
#importing the model & layer

import keras 

from keras.models import Sequential

from keras.layers import Dense
classifer = Sequential()
classifer.add(Dense(units = 11,kernel_initializer = 'uniform', activation = 'relu'))
classifer.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifer.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
classifer.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifer.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifer.fit(x_train,y_train, batch_size = 16, epochs = 200)
y_pred = classifer.predict(x_test)
y_pred = (y_pred>0.5)
y_pred[60]
y_test[60]
from sklearn.metrics import accuracy_score

a = accuracy_score(y_pred , y_test)
from sklearn.metrics import confusion_matrix

cm  = confusion_matrix(y_test,y_pred)
a
cm
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)



import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()