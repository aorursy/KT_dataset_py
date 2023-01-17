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
df = pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")

df.head()
data = df.iloc[:,3:13]

y = df.iloc[:,13]

data
datanew = pd.get_dummies(data,columns=['Geography','Gender'])

datanew
x= datanew.drop(['Geography_Spain','Gender_Male'],axis=1)

x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train = ss.fit_transform(x_train)

x_test = ss.transform(x_test)

import keras 

from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))
classifier.summary()
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
classifier.fit(x_train,y_train,batch_size = 10 ,nb_epoch=10)
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

y_pred
datapred = pd.DataFrame(data= y_pred, columns = ['y_pred'])

y_test = y_test.reset_index(drop = True)

datapred['y_test'] = y_test

datapred.head(20)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
(1510 + 159)/(2000) 