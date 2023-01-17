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
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head()
df.info()
df.describe()
df.drop(["time"],axis=1,inplace=True)
df.head()
y=df['DEATH_EVENT']
x=df.iloc[0:,:11]
x.head()
y.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2,test_size=.20)
scale=StandardScaler()
x_train_scale=scale.fit_transform(x_train)
x_test_scale=scale.transform(x_test)
rn=RandomForestClassifier()
rn.fit(x_train_scale,y_train)
rn.score(x_test_scale,y_test)
for i in range(2,11):
    kn=KNeighborsClassifier()
    kn.fit(x_train_scale,y_train)
    print(i,kn.score(x_test_scale,y_test))
lg=LogisticRegression()
lg.fit(x_train_scale,y_train)
lg.score(x_test_scale,y_test)
sv=SVC(kernel="poly")
sv.fit(x_train_scale,y_train)
sv.score(x_test_scale,y_test)
sv=SVC()
sv.fit(x_train_scale,y_train)
sv.score(x_test_scale,y_test)
sv=SVC(kernel="linear")
sv.fit(x_train_scale,y_train)
sv.score(x_test_scale,y_test)
nv=BernoulliNB()
nv.fit(x_train_scale,y_train)
nv.score(x_test_scale,y_test)
dt=DecisionTreeClassifier()
dt.fit(x_train_scale,y_train)
dt.score(x_test_scale,y_test)
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
hidden1=Dense(units=32,activation='relu')
hidden2=Dense(units=56,activation='relu')
hidden3=Dense(units=26,activation='relu')
hidden4=Dense(units=46,activation='relu')
out=Dense(units=1,activation='relu')
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
model.add(hidden4)
model.add(out)
model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
history=model.fit(np.asarray(x_train_scale),np.asarray(y_train),epochs=100,batch_size=10, 
          validation_data=(np.asarray(x_test_scale),np.asarray(y_test)))
model.evaluate(np.asarray(x_test_scale),np.asarray(y_test))
model.summary()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

