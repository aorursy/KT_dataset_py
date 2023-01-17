import numpy as np 

import pandas as pd 

import seaborn as sns
df=pd.read_csv('/kaggle/input/iris-data/iris.csv')
df.head()
df.describe()



sns.pairplot(df,hue="target")
X=np.array(df.drop('target',1))
X
y=np.array(df.target)
y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
for x,y in zip(X_test,y_test):

    print(y," | ",knn.predict([x]))


knn.score(X_train,y_train)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
for x,y, in zip(X_test,y_test):

    print(y," | ", nb.predict([x]))
nb.score(X_train,y_train)
import joblib





joblib.dump(knn, 'iris_modelo.pkl')


knn2 = joblib.load('iris_modelo.pkl')
knn2.classes_
knn2.predict([[6.5,6.5,4.7,1.3]])