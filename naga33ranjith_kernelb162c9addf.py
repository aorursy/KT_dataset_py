# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))







from sklearn.model_selection import train_test_split,GridSearchCV,KFold,LeaveOneOut

from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,accuracy_score,confusion_matrix

from sklearn.tree import DecisionTreeClassifier











# Any results you write to the current directory are saved as output.
dat=pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
dat.head(2)
dat.info()
dat.nunique()
dat['Department'].unique()
dat.groupby(by=["Department",'Attrition']).size().plot(kind='bar')

plt.show()
dat.groupby(by=['BusinessTravel','Attrition']).size().plot(kind='bar')

plt.show()
dat.groupby(by=['EducationField','Attrition']).size().plot(kind='bar')

plt.show()


dat.groupby(by=['EnvironmentSatisfaction','Attrition']).size().plot(kind='bar')

plt.show()
dat.groupby(by=['Gender','Attrition']).size().plot(kind='bar')

plt.show()

dat.groupby(by=['JobInvolvement','Attrition']).size().plot(kind='bar')

plt.show()









dat.groupby(by=['JobLevel','Attrition']).size().plot(kind='bar')

plt.show()

dat.groupby(by=['JobRole','Attrition']).size().plot(kind='bar')

plt.show()

                        
dat.groupby(by=['JobSatisfaction','Attrition']).size().plot(kind='bar')

plt.show()





dat.groupby(by=['MaritalStatus','Attrition']).size().plot(kind='bar')

plt.show()
print("Average monthly income for males is {}".format(dat[dat['Gender']=='Male']['MonthlyIncome'].mean()))

print("Average monthly income for males is {}".format(dat[dat['Gender']=='Female']['MonthlyIncome'].mean()))



sns.violinplot(x = 'Gender',y = 'MonthlyIncome',data=dat, hue='Attrition',split=True,palette='Set2')

plt.show()
sns.distplot(dat.Age,kde=False)

plt.show()
dat.dtypes
dat['BusinessTravel'] = dat['BusinessTravel'].astype('category')

dat['Department'] = dat['Department'].astype('category')

dat['EducationField'] = dat['EducationField'].astype('category')

dat['EnvironmentSatisfaction'] = dat['EnvironmentSatisfaction'].astype('category')

dat['Gender'] = dat['Gender'].astype('category')

dat['JobInvolvement'] = dat['JobInvolvement'].astype('category')

dat['JobLevel'] = dat['JobLevel'].astype('category')

dat['JobRole'] = dat['JobRole'].astype('category')

dat['JobSatisfaction'] = dat['JobSatisfaction'].astype('category')

dat['MaritalStatus'] = dat['MaritalStatus'].astype('category')

dat['NumCompaniesWorked'] = dat['NumCompaniesWorked'].astype('category')

dat['OverTime'] = dat['OverTime'].astype('category')

dat['RelationshipSatisfaction'] = dat['RelationshipSatisfaction'].astype('category')

dat['StockOptionLevel'] = dat['StockOptionLevel'].astype('category')

dat['WorkLifeBalance'] = dat['WorkLifeBalance'].astype('category')
#This will return the percentage of Attrition datasets

dat.Attrition.value_counts(normalize=True)*100

#this shows it is an imbalanced dataset

x=dat.drop(columns=['Attrition'])
y=dat['Attrition']
x=pd.get_dummies(x)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
print(X_train.shape)

print(y_train.shape)
modelgini=DecisionTreeClassifier(criterion='gini')
modelgini.fit(X_train,y_train)
predictors_gini=modelgini.predict(X_test)
modelentropy=DecisionTreeClassifier(criterion='entropy')
modelentropy.fit(X_train,y_train)
predictors_entropy=modelentropy.predict(X_test)
Matrix_Gini=confusion_matrix(y_test,predictors_gini)

Matrix_Entropy=confusion_matrix(y_test,predictors_entropy)





print("confusion matrix for gini = \n",Matrix_Gini)



print("confusion matrix for Entropy = \n",Matrix_Entropy)
#Accuracy Score

print("Accuracy Score for Gini :",accuracy_score(y_test,predictors_gini))

print("Accuracy Score for Entropy :",accuracy_score(y_test,predictors_entropy))
#Classification Report

print(classification_report(y_test,predictors_gini))

print(classification_report(y_test,predictors_entropy))
clf_pruned=DecisionTreeClassifier(criterion='gini',max_depth=3,max_leaf_nodes=5)

clf_pruned.fit(X_train,y_train)
predictors_pruned=clf_pruned.predict(X_test)
#Confusion matrix



mat_pruned = confusion_matrix(y_test,predictors_pruned)



print("confusion matrix = \n",mat_pruned)
#Accuracy Score

print("Accuracy Score is : {}".format(accuracy_score(y_test,predictors_pruned)))
print(classification_report(y_test,predictors_pruned))