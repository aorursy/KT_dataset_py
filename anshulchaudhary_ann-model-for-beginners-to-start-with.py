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

import tensorflow as tf



from keras.layers import *

from keras.models import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
print(tf.__version__)
dataset = pd.read_csv('/kaggle/input/Social_Network_Ads.csv')
dataset.head() 
dataset.isnull().sum()
dataset.drop(columns=['User ID'],inplace = True)
x = dataset.iloc[:,:3]

y = dataset.iloc[:,3]
le = LabelEncoder()

x['Gender'] = le.fit_transform(x['Gender'])

x
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 5)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
model = Sequential()



model.add(Dense(16))

model.add(Activation('relu'))



model.add(Dense(16))

model.add(Activation('relu'))



model.add(Flatten())

model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.fit(x_train,y_train ,batch_size = 16 ,epochs = 100)
y_pred = model.predict(x_test)

y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))
print('percentage Accuracy : ',100*accuracy_score(y_test,y_pred))
pred = model.predict(sc.transform([[1, 36, 33000]])) > 0.5

if pred == True:

    print('1 : True')

else:

    print('0 : False')