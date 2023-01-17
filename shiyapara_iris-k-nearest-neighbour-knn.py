# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
iris_data= pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

iris_data.dataframeName= 'Iris.csv'

iris_data.head()
iris_data.shape
iris_data.info()
iris_data.groupby('species').size()
iris_data.describe()
sns.lmplot(x='sepal_length', y= 'sepal_width', hue ='species',data= iris_data)
sns.lmplot(x='petal_length', y= 'petal_width', hue ='species', data= iris_data)
sns.pairplot(hue ='species', data= iris_data)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(iris_data.drop("species",axis =1))

scaled_features = scaler.transform(iris_data.drop("species",axis =1))
df_feat= pd.DataFrame(scaled_features, columns=iris_data.columns[: -1])

df_feat.head()
X = np.array(df_feat.iloc[:, 0:4]) 

y = np.array(iris_data['species']) 
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,

                                                    test_size=0.20)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(pred)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
accuracy = accuracy_score(y_test, pred)

print('accuracy:{}'.format(100*accuracy))
error_rate = []



for i in range(1,50):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
#NOW WITH K=10

knn = KNeighborsClassifier(n_neighbors=10)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=10')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#NOW WITH K=40

knn = KNeighborsClassifier(n_neighbors=40)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=40')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#NOW WITH K=50

knn = KNeighborsClassifier(n_neighbors=50)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=50')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))