







import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno

data=pd.read_csv("../input/sales-analysis/SalesKaggle3.csv")
data1=pd.read_csv("../input/sales-analysis/SalesKaggle3.csv")
data.head()
data.isnull().sum()
data.shape
missingno.matrix(data)
data=data.drop(["SKU_number","Order","SoldCount","ReleaseYear"],axis=1)
data.head()
sns.distplot(data.StrengthFactor)
sns.distplot(data.PriceReg)
sns.distplot(data.LowUserPrice)
data.MarketingType.value_counts()
sns.countplot(data.SoldFlag)
data.head()
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
scdata=pd.DataFrame(sc.fit_transform(data.drop(["File_Type","SoldFlag","MarketingType","New_Release_Flag"],axis=1)),columns=data.drop(["File_Type","SoldFlag","MarketingType","New_Release_Flag"],axis=1).columns)
scdata.head()
scdata.isnull().sum()
scdata[["File_Type","SoldFlag","MarketingType","New_Release_Flag"]]=data[["File_Type","SoldFlag","MarketingType","New_Release_Flag"]]
scdata.head()
data=pd.get_dummies(scdata)
data.head()
sns.pairplot(data)
test=data[data.SoldFlag.isnull()]
test.SoldFlag.value_counts()
test.head()
test.shape
train=data[data.SoldFlag.notnull()]
train.head()
train["SoldFlag"].unique()
test.head()
xtest11=test.drop("SoldFlag",axis=1)
xtrain1=train.drop("SoldFlag",axis=1)
ytrain1=train.SoldFlag
# Train Test Split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(xtrain1,ytrain1,test_size=0.30,random_state=2)
xtrain.head()
ytrain.head()
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
dt=DecisionTreeClassifier()
ypred=dt.fit(xtrain,ytrain).predict(xtest)
print (classification_report(ytest, ypred))

print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred) * 100))



sns.heatmap(confusion_matrix(ytest, ypred), annot=True, fmt='.2f')

plt.xlabel("Predicted")

plt.ylabel("Actal")

plt.show()
# Random Forest

from sklearn.ensemble import RandomForestClassifier

batman=RandomForestClassifier()
ypred1=batman.fit(xtrain,ytrain).predict(xtest)
print (classification_report(ytest, ypred1))

print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred1) * 100))



sns.heatmap(confusion_matrix(ytest, ypred1), annot=True, fmt='.2f')

plt.xlabel("Predicted")

plt.ylabel("Actal")

plt.show()
from sklearn.model_selection import GridSearchCV



param_dist = {'max_depth': [2, 3, 4],'bootstrap': [True, False],'max_features': ['auto', 'sqrt', 'log2', None],'criterion': ['gini', 'entropy']}



cv_rf = GridSearchCV(batman, cv = 10,param_grid=param_dist)

cv_rf.fit(xtrain,ytrain)
cv_rf.best_params_
batman=RandomForestClassifier(bootstrap = True, criterion = 'gini', max_depth = 4, max_features = None)
ypred2=batman.fit(xtrain,ytrain).predict(xtest)
print (classification_report(ytest, ypred2))

print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred2) * 100))



sns.heatmap(confusion_matrix(ytest, ypred2), annot=True, fmt='.2f')

plt.xlabel("Predicted")

plt.ylabel("Actal")

plt.show()
# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
ypred3=lr.fit(xtrain,ytrain).predict(xtest)
print (classification_report(ytest, ypred3))

print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred3) * 100))



sns.heatmap(confusion_matrix(ytest, ypred3), annot=True, fmt='.2f')

plt.xlabel("Predicted")

plt.ylabel("Actal")

plt.show()
# KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
ypred7=knn.fit(xtrain,ytrain).predict(xtest)
print (classification_report(ytest, ypred7))

print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred7) * 100))



sns.heatmap(confusion_matrix(ytest, ypred7), annot=True, fmt='.2f')

plt.xlabel("Predicted")

plt.ylabel("Actal")

plt.show()
# Applying Grid Search

l=[]

for i in range(1,10):

    l.append(i)

from sklearn.model_selection import GridSearchCV



param_dist = {"n_neighbors": l, "p":[1,2,3]}

cv_rf = GridSearchCV(knn, cv = 3,param_grid=param_dist)

cv_rf.fit(xtrain, np.ravel(ytrain))

print(cv_rf.best_params_)

knn=KNeighborsClassifier(n_neighbors= 8, p= 3)
ypred8=knn.fit(xtrain,ytrain).predict(xtest)
print (classification_report(ytest, ypred8))

print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred8) * 100))



sns.heatmap(confusion_matrix(ytest, ypred8), annot=True, fmt='.2f')

plt.xlabel("Predicted")

plt.ylabel("Actal")

plt.show()
# Trying out XGBoost
from xgboost import XGBRFClassifier

xg=XGBRFClassifier()
ypred9=xg.fit(xtrain,ytrain).predict(xtest)
print (classification_report(ytest, ypred9))

print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred9) * 100))



sns.heatmap(confusion_matrix(ytest, ypred9), annot=True, fmt='.2f')

plt.xlabel("Predicted")

plt.ylabel("Actal")

plt.show()
# Predicting the values of SaleFlag using out model using Random Forest Model
x2=test.drop("SoldFlag",axis=1)
ypred10=batman.predict(x2)
pd.DataFrame(ypred10).head()     # Predicted values
xyz=data1[data1.SoldFlag.isnull()]
xyz.head()
xyz.SoldFlag=ypred10
xyz.isnull().sum()
xyz.head()