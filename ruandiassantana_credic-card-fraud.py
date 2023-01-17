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
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.info()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X = df.drop('Class',axis=1)

y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
svm_pred = model.predict(X_test)
print(classification_report(y_test,svm_pred))
print(confusion_matrix(y_test,svm_pred))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test,knn_pred))
from sklearn.cluster import KMeans
kmc = KMeans(n_clusters=2)
kmc.fit(X_train)
kmc_pred = kmc.predict(X_test)
print(classification_report(y_test,kmc_pred))
print(confusion_matrix(y_test,kmc_pred))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree_pred = dtree.predict(X_test)
print(classification_report(y_test,dtree_pred))
print(confusion_matrix(y_test,dtree_pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
model = Sequential()



model.add(Dense(30,activation='relu'))



model.add(Dense(30,activation='relu'))



model.add(Dense(units=1,activation='sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy')
model.fit(x=X_train,y=y_train,epochs=100,batch_size=256,validation_data=(X_test,y_test))
tf_pred = model.predict_classes(X_test)
print(classification_report(y_test,tf_pred))
print(confusion_matrix(y_test,tf_pred))