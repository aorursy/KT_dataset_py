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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
data = pd.read_csv('/kaggle/input/kickstarter-data-set/proyectos.csv')
data.head()
X=data
X=X[X.state!='live']
X=X.rename({'usd pledged': 'usd_pledged'}, axis=1)
X=X.drop(['name',"ID"],axis=1)

#Preparing  data Objects and Numbers
X_object = X.select_dtypes(include=object)
X_num = X.select_dtypes(include=np.number)
dummies = pd.get_dummies(X.main_category, prefix="main_category")
X = X.drop('main_category',axis = 1)
X = X.join(dummies)
dummies = pd.get_dummies(X.category, prefix="category")
X = X.drop('category',axis = 1)
X = X.join(dummies)
dummies = pd.get_dummies(X.country, prefix="country")
X = X.drop('country',axis = 1)
X = X.join(dummies)
from datetime import timedelta
from datetime import datetime 
import datetime as dt
tiempo = np.array(X['deadline'])
endtime=np.array([datetime.strptime(t, '%Y-%m-%d') for t in tiempo ])
X['deadline']=endtime
tiempo = np.array(X['launched'])
starttime=np.array([datetime.strptime(t, '%Y-%m-%d  %H:%M:%S') for t in tiempo])
X['launched']=starttime
X['duration'] = (endtime-starttime)
tiempo =X['duration']
duration=np.array([t.days for t in tiempo])

X['duration']=duration
X = X.drop('deadline',axis = 1)
X = X.drop('launched',axis = 1)
def missing_data(data):
#source>
#https://www.kaggle.com/ajaykgp12/ecommerce-eda/notebook

        test = data.isnull().sum()
        total = test.sort_values(ascending = False)
        percent= (data.isnull().sum() * 100 / data.isnull().count() ).sort_values(ascending = False)
        df = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    
        return df[df['Total'] != 0]

missing_data(X)

#realmente no se puede predecir muy bien
X = X.drop(['usd_pledged'], axis =1)
X = X.drop(['goal','pledged','currency'],axis = 1)
y=X.state
y=y.replace(['canceled','suspended','undefined'],'failed')
   
X=X.drop("state",axis=1)
   
#definition of data later we do cross validation.
from sklearn.model_selection import train_test_split
# Test-train split. Consider a test set of (+ -) 30%.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#y= pd.get_dummies(y)

#y=y.drop('failed',axis=1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

y_prima = model.fit(X_train, y_train).predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test, y_prima))
# 10-k-fold CV
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10)

print("CV:",scores)
print("Accuracy:",scores.mean())
class_names = (model.classes_)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_prima, target_names=class_names))
from sklearn.metrics import accuracy_score
print("Acc:",accuracy_score(y_test, y_prima))

from sklearn.metrics import plot_confusion_matrix

 
plot_confusion_matrix(model, X,y,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
k=[1, 3, 5, 7, 9]
preds_kNN=[]


for i in range(len(k)):
    kNN=(KNeighborsClassifier(n_neighbors=k[i]).fit(X_train, y_train))
    # Lo evaluamos contra el grid de posiciones generadas anteriormente
    preds_kNN.append(kNN.predict(X_test))
    error_rate.append(np.mean(kNN.predict(X_test) != y_test))

plt.figure(figsize=(10,6))
plt.plot(k,error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
kNN=(KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train))
# 10-k-fold CV
scores = cross_val_score(kNN, X_test, y_test, cv=10)

#print("CV:",scores)
print("Accuracy:",scores.mean())
class_names = (kNN.classes_)
y_prima =(kNN.predict(X))
from sklearn.metrics import classification_report
print(classification_report(y, y_prima, target_names=class_names))
from sklearn.metrics import accuracy_score
print("Acc:",accuracy_score(y, y_prima))

from sklearn.metrics import plot_confusion_matrix

 
plot_confusion_matrix(kNN, X,y,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train, y_train)
# lo evaluamos contra los mismos datos de entrada
print((model.predict(X_test) == y_test).sum(), ' / ', y_test.shape[0], ' clasificados correctamente en aprendizaje')

y_prima = model.fit(X_train, y_train).predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test, y_prima))
# 10-k-fold CV
scores = cross_val_score(model, X, y, cv=10)

#print("CV:",scores)
print("Accuracy:",scores.mean())
class_names = (model.classes_)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_prima, target_names=class_names))
from sklearn.metrics import accuracy_score
print("Acc:",accuracy_score(y_test, y_prima))

from sklearn.metrics import plot_confusion_matrix

 
plot_confusion_matrix(model, X,y,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)