import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
data=pd.read_csv("../input/weatherAUS.csv",parse_dates=["Date"])
data.head()
data.dtypes
data.isnull().sum()
data=data.drop(columns=['Date','Location','Sunshine','Cloud9am','Cloud3pm','RISK_MM','Evaporation'])
data.head()
data.corr()
data=data.dropna(how='any')

x=data.drop(columns=['RainTomorrow'])
x=pd.get_dummies(x)
y=data.RainTomorrow
print(y.shape,x.shape)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

X_scaled = pd.DataFrame(min_max_scaler.fit_transform(x), columns = x.columns)

X_scaled.head()
from sklearn.feature_selection import chi2,SelectKBest
xnew=SelectKBest(chi2,k=10) #choosing top 10 features.
xnew.fit(X_scaled,y)

x.columns[xnew.get_support()]
pd.Series(xnew.scores_,X_scaled.columns).sort_values(ascending=False)
X_final=x[['Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',

       'WindGustDir_E', 'WindDir9am_E', 'WindDir9am_N', 'WindDir9am_NNW',

       'RainToday_No', 'RainToday_Yes']]
X_final.head(5)
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,accuracy_score
X_train,X_test,Y_train,Y_test=train_test_split(X_final,y,test_size=0.30,random_state=10)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix
parameters={'max_depth':[1,2,3,4,5],'min_samples_split':[2,3,4,5],'min_samples_leaf':[1,2,3,4,5],'criterion':['gini','entropy']}

dt=DecisionTreeClassifier()
clf=GridSearchCV(dt,parameters,scoring='accuracy')
clf.fit(X_train,Y_train)
clf.best_score_
clf.best_params_
tree=DecisionTreeClassifier(criterion='entropy',

 max_depth= 5,min_samples_leaf=1,min_samples_split= 2)
tree.fit(X_train,Y_train)
Y_pred=tree.predict(X_test)
accuracy_score(Y_train,tree.predict(X_train))
accuracy_score(Y_test,Y_pred)
dict={'Y_test':Y_test,"Y_pred":Y_pred}
result=pd.DataFrame(dict)
result.replace(['Yes','No'],[1,0])
#Creating Confusion MAtrix



conmat=confusion_matrix(result.Y_test,result.Y_pred)

conmat
plt.figure(figsize=(9,6))

plt.subplot(1,2,1)

sns.countplot(result.Y_test)

plt.title("Actual Value counts")

plt.subplot(1,2,2)

plt.title("Predicted Value counts")

sns.countplot(result.Y_pred)

plt.show()
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier()
para={'n_neighbors':range(1,15,1)}



knn=GridSearchCV(KNN,para,scoring="accuracy")
knn.fit(X_train,Y_train)
knn.best_estimator_
knn.best_score_
KNN=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_jobs=None, n_neighbors=14, p=2,

           weights='uniform')
KNN.fit(X_train,Y_train)
ypredknn=KNN.predict(X_test)
print("accuracy score in Train:", accuracy_score(KNN.predict(X_train),Y_train))
print("accuracy score in Test:", accuracy_score(ypredknn,Y_test))
confusion_matrix(ypredknn,Y_test)