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

%matplotlib inline

import seaborn as sns
train = pd.read_csv('/kaggle/input/machinehack-financial-risk-prediction/Train.csv')

test = pd.read_csv('/kaggle/input/machinehack-financial-risk-prediction/Test.csv')

sub = pd.read_csv('/kaggle/input/machinehack-financial-risk-prediction/Sample_Submission.csv')
print(train.shape)

train.head()
print(test.shape)

test.head()
sns.countplot(train['IsUnderRisk'])

plt.show()
train.describe()
train['Location_Score'] = np.log(train['Location_Score'])
train.describe()
var_with_na = [var for var in train.columns if train[var].isnull().sum()>1]



for var in var_with_na:

    print(train, var)
X = train.iloc[:, train.columns != 'IsUnderRisk']

y = train.iloc[:, train.columns == 'IsUnderRisk']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.20, random_state=0)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()
scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix

cr = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(cr)
from keras.models import Sequential 

from keras.layers import Dense, Activation, Dropout

from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Dense(30,activation='relu'))

model.add(Dense(15,activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
model.fit(x=X_train,y=y_train.values,epochs=2, validation_data=(X_test,y_test.values), verbose=1)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
model = Sequential()

model.add(Dense(30,activation='relu'))

model.add(Dense(15,activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=25)
model.fit(x=X_train,y=y_train.values,epochs=600,validation_data=(X_test, y_test.values), verbose=1,callbacks=[early_stop])
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
predictions = model.predict_classes(test)
submission = pd.DataFrame(data=predictions)

submission.head()

submission.to_csv('Predictions.csv')