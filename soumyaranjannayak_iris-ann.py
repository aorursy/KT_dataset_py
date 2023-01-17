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
dataset=pd.read_csv(r'/kaggle/input/datasets_Iris.csv')
print(dataset.columns)

dataset=dataset.drop('Id',axis=1)
dataset.head()
sns.pairplot(dataset,hue='Species')
plt.show()
x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
lenc=LabelEncoder()
y=lenc.fit_transform(y)
y=pd.get_dummies(y).values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
std_sclr=StandardScaler()
std_sclr.fit_transform(x_train)
std_sclr.transform(x_test)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,adam
model=Sequential()
model.add(Dense(units=20,activation='relu',kernel_initializer='RandomUniform',input_dim=4))
model.add(Dense(units=15,activation='relu',kernel_initializer='RandomUniform'))
model.add(Dense(units=3,activation='softmax',kernel_initializer='RandomUniform'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)
y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred,axis=1)
y_test=np.argmax(y_test,axis=1)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))