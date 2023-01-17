import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
train = pd.read_csv('../input/iris/Iris.csv', index_col=0)
train.head()
train.info()
train.describe()
print(train['Species'].nunique())
print(train['Species'].unique())
fig, ax = plt.subplots(2,2, figsize=(12,8))
ax[0,0] = sns.distplot(train['SepalLengthCm'], ax=ax[0,0])
ax[0,1] = sns.distplot(train['SepalWidthCm'], ax=ax[0,1])
ax[1,0] = sns.distplot(train['PetalLengthCm'], ax=ax[1,0])
ax[1,1] = sns.distplot(train['PetalWidthCm'], ax=ax[1,1])
species = train.groupby('Species').mean()
species
fig,ax = plt.subplots(2,2,figsize=(16,11))
ax[0,0] = sns.boxplot(x=train['Species'],y=train['SepalLengthCm'], ax=ax[0,0])
ax[0,0].set_title('SepalLengthCm')

ax[0,1] = sns.boxplot(x=train['Species'],y=train['SepalWidthCm'], ax=ax[0,1])
ax[0,1].set_title('SepalWidthCm')

ax[1,0] = sns.boxplot(x=train['Species'],y=train['PetalLengthCm'], ax=ax[1,0])
ax[1,0].set_title('PetalLengthCm')

ax[1,1] = sns.boxplot(x=train['Species'],y=train['PetalWidthCm'], ax=ax[1,1])
ax[1,1].set_title('PetalWidthCm')
plt.figure(figsize=(7,7))
sns.heatmap(train.corr(),annot = True)
sns.pairplot(train,hue='Species')
fig, ax= plt.subplots(1,2,figsize=(12,5))
ax[0] = sns.scatterplot(x=train['PetalLengthCm'],y=train['SepalWidthCm'],hue=train['Species'],ax=ax[0])
ax[0].set_title('PetalLengthCm x SepalWidhtCm')
ax[1] = sns.scatterplot(x=train['PetalWidthCm'],y=train['PetalLengthCm'],hue=train['Species'],ax=ax[1])
ax[1].set_title('PetalWidthCm x PetalLengthCm')
y = train['Species']
x = train.drop('Species',axis=1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.3)
from sklearn import svm
model_svm = svm.SVC()
model_svm.fit(xtrain,ytrain)
svm_predict = model_svm.predict(xtest)
model_svm_acc = accuracy_score(ytest,svm_predict)
print(f'Accuracy score: {model_svm_acc}')
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=1)
model_tree = model.fit(xtrain,ytrain)
model_tree_predict = model.predict(xtest)
model_tree_acc = accuracy_score(ytest,model_tree_predict)
model_tree_acc
from sklearn.ensemble import RandomForestClassifier
for i in [100,200,300,400,500]:    
    model_random_forest = RandomForestClassifier(n_estimators=i,random_state = 42)
    model_random_forest.fit(xtrain,ytrain)
    model_forest_predict = model.predict(xtest)
    model_forest_accuaracy = accuracy_score(ytest,model_forest_predict)
    print(model_forest_accuaracy)
from sklearn.linear_model import LogisticRegression
logist_regression = LogisticRegression(max_iter=200,random_state=42)
logist_regression.fit(xtrain,ytrain)
predict = logist_regression.predict(xtest)
logist_regression = accuracy_score(ytest,predict)
logist_regression
from sklearn.neighbors import KNeighborsClassifier
model_KN=KNeighborsClassifier(n_neighbors=5) 
model_KN.fit(xtrain,ytrain)
KN_predict =model_KN.predict(xtest)
KN_accuracy = accuracy_score(ytest,KN_predict)
KN_accuracy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
index = np.arange(1,6)
my_pipeline = Pipeline(steps=[('model', LogisticRegression(max_iter=200))])
scores =cross_val_score(my_pipeline, x, y,
                              cv=len(index),scoring='accuracy')
plt.figure(figsize=(7,7))
sns.lineplot(x=index,y=scores)
plt.ylabel('Accuracy')
plt.xlabel('Cv')
plt.title('Accuracy with LogisticRegression')
print(scores.mean())
