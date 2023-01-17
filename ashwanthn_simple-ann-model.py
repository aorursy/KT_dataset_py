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
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.head()
df = df[~(df.TotalCharges == ' ')]
X = df.drop(columns=['customerID','Churn'])

X.head()
y = df['Churn']
def PreProcess(X,features):

    

    for feature in features:

        new_feature = pd.get_dummies(X[feature],drop_first=True)

        X = pd.concat([X,new_feature],axis=1)

        X = X.drop(columns=[feature])

    

    return X
X_new = PreProcess(X,['gender','Partner','Dependents','PhoneService','MultipleLines',

                     'InternetService','OnlineSecurity','OnlineBackup',

                      'DeviceProtection','TechSupport','StreamingTV',

                     'StreamingMovies','Contract','PaperlessBilling','PaymentMethod'])

X_new.head()
X_new.TotalCharges = X_new.TotalCharges.astype('float64')
y_new = pd.get_dummies(y,drop_first=True)

y_new
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_new,y_new,test_size = 0.25,

                                                 random_state = 6)
X_train.TotalCharges.unique()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train.shape
import keras

from keras.models import Sequential

from keras.layers import Dense 

from keras.layers import Dropout
classifier = Sequential()
classifier.add(Dense(units = 20, kernel_initializer = 'he_uniform', activation = 'relu',input_dim = 30))

classifier.add(Dropout(0.3))
classifier.add(Dense(units = 12, kernel_initializer = 'he_uniform', activation = 'relu'))

classifier.add(Dropout(0.4))

classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu'))

classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
classifier.summary()
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train,y_train,validation_split = 0.33, batch_size = 10, epochs =100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

y_pred
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score

score = accuracy_score(y_pred,y_test)

score