import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading data from csv file

data=pd.read_csv('/kaggle/input/wine-quality/winequalityN.csv')

print(data.head())
data.info()
dummies=pd.get_dummies(data['type'])

data=pd.concat([dummies,data],axis=1)

#1st column is of type,not required

data= data.drop('type', 1)

print(data.head())
data.isna().sum()
data=data.apply(lambda x: x.fillna(x.mean(),axis=0))

data.isna().sum()
import seaborn as sns

corr=data.corr(method='pearson')

#using heatmap to plot correlation

sns.heatmap(corr,cmap="YlGnBu",annot=True)
#coorelation for density with quality is minimum hence dropping it

data= data.drop('density', 1)
sns.pairplot(data)
X = data.iloc[:, :-1].values

y = data.iloc[:,12].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

KNNclassifier=KNeighborsClassifier()

KNNclassifier.fit(X_train,y_train)

KNNpredict=KNNclassifier.predict(X_test)
from sklearn.svm import SVC

SVMlinear=SVC(kernel='linear')

SVMlinear.fit(X_train,y_train)

SVMlinear_predict=SVMlinear.predict(X_test)
from sklearn.svm import SVC

SVMrbf=SVC(kernel='rbf')

SVMrbf.fit(X_train,y_train)

SVMrbf_predict=SVMrbf.predict(X_test)
from sklearn.svm import SVC

SVMpoly=SVC(kernel='poly')

SVMpoly.fit(X_train,y_train)

SVMpoly_predict=SVMpoly.predict(X_test)
from sklearn.naive_bayes import GaussianNB

NB=GaussianNB()

NB.fit(X_train,y_train)

NB_predict=NB.predict(X_test)
from sklearn.tree import DecisionTreeClassifier

DecisionTree=DecisionTreeClassifier(criterion='entropy',random_state=0)

DecisionTree.fit(X_train,y_train)

DecisionTree_predict=DecisionTree.predict(X_test)
from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier(n_estimators=12,criterion='entropy',random_state=0)

RFC.fit(X_train,y_train)

RFC_predict=RFC.predict(X_test)
from sklearn.metrics import accuracy_score

print("Accuracy of KNN Classifier",accuracy_score(y_test,KNNpredict)*100)

print("Accuracy of SVM Classifier(kernel=linear))",accuracy_score(y_test,SVMlinear_predict)*100)

print("Accuracy of SVM Classifier(kernel=rbf)",accuracy_score(y_test,SVMrbf_predict)*100)

print("Accuracy of SVM Classifier(kernel=poly)",accuracy_score(y_test,SVMpoly_predict)*100)

print("Accuracy of naive Bayes Classifier",accuracy_score(y_test,NB_predict)*100)

print("Accuracy of Decision Tree Classifier",accuracy_score(y_test,DecisionTree_predict)*100)

print("Accuracy of Random Forest Classifier",accuracy_score(y_test,RFC_predict)*100)