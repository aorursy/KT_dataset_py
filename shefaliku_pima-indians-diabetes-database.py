# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
df=pd.read_csv('../input/diabetes.csv')
# It shows you the top 5 rows of data

df.head()
#It print the full summary.The summary includes list of all columns with their data types and the number of non-null values in each column

df.info()
df.shape
#It Check for the missing values, and print total count of null values.

df.isnull().sum()
#Countplot show the counts of observations in each categorical bin using bars.

sns.countplot(x='Outcome',data=df)
#It show some basic statistical details like percentile, mean, std etc. 

df.describe()
df.hist(figsize=(15,10))
#It shows pairwise correlation of all columns in the dataframe

d=df.corr()

plt.figure(figsize=(10,7))

sns.heatmap(d, annot=True)
#Y is a target variable and X is a set of predictors

Y=df['Outcome']

X=df.drop(['Outcome'],axis=1)

X.head()
#Split and train the model

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.80,random_state=12)
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
# Model1- Decision Tree

DT= DecisionTreeClassifier()

model1=DT.fit(X_train,Y_train)

print(model1)
Y_prd=model1.predict(X_test)

print(accuracy_score(Y_test,Y_prd))

print(confusion_matrix(Y_test,Y_prd))

print(classification_report(Y_test,Y_prd))
#Plot confusion matrix

sns.set(font_scale=1)

cm = confusion_matrix(Y_test,Y_prd)

sns.heatmap(cm, annot=True, fmt='g', cmap="Blues")

plt.show()
#Standardization of data

from sklearn.preprocessing import StandardScaler

Std= StandardScaler()

X_train1=Std.fit_transform(X_train)

X_test1=Std.fit_transform(X_test)

model1_1=DT.fit(X_train1,Y_train)

print(model1_1)
#Standardization not necessary to perform in decision trees. Tree based models are not distance based models and can handle varying ranges of features

Y_prd1=model1.predict(X_test1)

print(accuracy_score(Y_test,Y_prd))

print(confusion_matrix(Y_test,Y_prd))

print(classification_report(Y_test,Y_prd))
#RandomForest Classifier

RF=RandomForestClassifier()

R_model=RF.fit(X_train1,Y_train)

print(R_model)
Y_prd2=R_model.predict(X_test1)

print(accuracy_score(Y_test,Y_prd2))

print(confusion_matrix(Y_test,Y_prd2))

print(classification_report(Y_test,Y_prd2))
grid_param = {  

    'n_estimators': [5,7,10,15,20],

    'criterion': ['gini', 'entropy'],

    'max_depth': [4,6,7,8,9,10,15],

    'max_leaf_nodes':[8,10,11,12]

}
GS = GridSearchCV(estimator=RF,  

                     param_grid=grid_param,

                     cv=5,

                    )
GS_model=GS.fit(X_train1,Y_train)
print(GS_model.best_params_)

result = GS.best_score_  

print(result)  