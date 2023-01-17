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
df = pd.read_csv('../input/train.csv')
# Missing Values ( NaN )
df.info()
import missingno as misnan

misnan.matrix(df)
print(df.isnull().sum())
df["Age"]=df["Age"].fillna(df["Age"].median())
df=df.drop(["Cabin"], axis=1)
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df.info()
df.head()
df.tail()
# Binary Variables
df["Sex"].unique()
df["Sex_F"]=df["Sex"].apply(lambda x : 1 if x=="female" else 0 )
# Dummy Variables
df["Pclass"].unique()
X_Pclass=pd.get_dummies(df["Pclass"], prefix = 'Pclass_', drop_first = True)
df["Embarked"].unique()
X_Embarked=pd.get_dummies(df["Embarked"], prefix = 'Embarked_', drop_first = True)
df.head()
df_cleaned=pd.concat([df,X_Pclass, X_Embarked], axis=1 )
df_cleaned.head()
df_cleaned.tail()
df_cleaned.shape
df_cleaned.describe()
# Benchmark Model
df_cleaned.columns
df_cleaned.info()
df_bench2=pd.DataFrame(df_cleaned, columns=['PassengerId', 'Survived', 'Age', 'SibSp',

       'Parch', 'Fare', 'Sex_F', 'Pclass__2',

       'Pclass__3', 'Embarked__Q', 'Embarked__S'] )
df_bench2 = df_bench2.set_index('PassengerId')
df_bench2 = df_bench2.sort_index()
df_bench2.shape
df_bench2.head()
df_bench2.info()
df_bench2.describe()
df_bench2.groupby("Age").mean()
df_bench2.groupby("Sex_F").mean()
# Correlation Matrix

corr = df_bench2.corr()  

corr
import matplotlib.pyplot as plt

import seaborn as sns
f, ax = plt.subplots(figsize=(18, 10))

sns.heatmap(corr, annot=True, ax=ax)
from sklearn.model_selection import train_test_split

X= df_bench2.iloc[:,1:]

y= df_bench2.iloc[:,0]

train_X, test_X, train_y, test_y = train_test_split( X, y, test_size=1/2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train_X = sc.fit_transform(train_X) 

test_X = sc.fit_transform(test_X) 
from sklearn.linear_model import LogisticRegression

logisticreg = LogisticRegression()

logisticreg.fit(train_X, train_y)

test_y_pred = logisticreg.predict(test_X) 

Acc_BM=logisticreg.score(test_X, test_y) 

print ("The Benchmark Model has an accuracy of : %.2f%%" % (Acc_BM * 100.0))
from sklearn import metrics

cm = metrics.confusion_matrix(test_y, test_y_pred) # on calcule la matrice de confusion

print(cm)
def accuracy(confusion_matrix):

    diagonal_sum = confusion_matrix.trace()

    sum_of_all_elements = confusion_matrix.sum()

    return diagonal_sum / sum_of_all_elements
print ("Accuracy_CM:  %.2f%%" % (accuracy(cm) * 100.0))
# New features
# New Age Categories of passenger
df_cleaned["Child"]=df_cleaned["Age"].apply(lambda x : 1 if x<15 else 0 )
df_cleaned["Teenager"]=df_cleaned["Age"].apply(lambda x : 1 if (x>=15) and (x<25) else 0 )
df_cleaned["Adult"]=df_cleaned["Age"].apply(lambda x : 1 if (x>=25) & (x<65) else 0 )
df_cleaned["Old"]=df_cleaned["Age"].apply(lambda x : 1 if x>=65 else 0 )
df_cleaned.head()
# New isAlone Category of passenger
df_cleaned['isAlone'] = ((df_cleaned['SibSp'] == 0) & (df_cleaned['Parch'] == 0)).apply(int)
df_cleaned.head()
df_cleaned.columns
df_cleaned.info()
# Data scaling
df= pd.DataFrame(df_cleaned,columns=['PassengerId', 'Survived', 'SibSp',

       'Parch', 'Fare', 'Sex_F', 'Pclass__2',

       'Pclass__3', 'Embarked__Q', 'Embarked__S', 'Child', 'Teenager', 'Adult',

       'Old', 'isAlone'])
df = df.set_index('PassengerId')
df = df.sort_index()
df.head()
df.info()
df.columns
from sklearn.model_selection import train_test_split

X= df.iloc[:,1:]

y= df.iloc[:,0]

train_X, test_X, train_y, test_y = train_test_split( X, y, test_size=1/2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train_X = sc.fit_transform(train_X) 

test_X = sc.fit_transform(test_X) 
train_X
test_X
# Modeling
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logisticreg = LogisticRegression()

logisticreg.fit(train_X, train_y)

test_y_pred = logisticreg.predict(test_X) 
Acc_LR=logisticreg.score(test_X, test_y) 

print("The Logistic Redgression Model has an accuracy of : %.2f%%" % (Acc_LR * 100.0))
logisticreg.coef_
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier(n_estimators=1000, random_state=0)

rfc.fit(train_X, train_y)

y_pred = rfc.predict(test_X)
Acc_RFC = accuracy_score(test_y, y_pred)

print("The Random Forest Classifier Model has an accuracy of : %.2f%%" % (Acc_RFC * 100.0))
# Naive Bayes
import sklearn.naive_bayes as nb
nbc = nb.GaussianNB()

nbc.fit(train_X, train_y)

y_pred = nbc.predict(test_X)
Acc_NBC = accuracy_score(test_y, y_pred)

print("The Naive Bayes MModel has an accuracy of : %.2f%%" % (Acc_NBC * 100.0))
# SVM
from sklearn.svm import SVC
svclf = SVC()

svclf.fit(train_X, train_y)

y_pred = svclf.predict(test_X)
Acc_svclf = accuracy_score(test_y, y_pred)

print("The Naive Bayes Model has an accuracy of : %.2f%%" % (Acc_svclf * 100.0))
# Random Forest Classifier Hyperparameters
from sklearn.model_selection import GridSearchCV
# grid_param dictionary

grid_param = {  

    'n_estimators': [10,20,30,60,100],

    'criterion': ['gini', 'entropy'],

    'bootstrap': [True, False]

}
# instance of the GridSearchCV class

gds_rfc = GridSearchCV(estimator=rfc,     

                     param_grid=grid_param,    

                     scoring='accuracy',       

                     cv=5,                     

                     n_jobs=-1) 
gds_rfc.fit(train_X, train_y)
# Optimal hyperparameters: best_params_

gds_rfc.best_params_
# Best score found (mean score on all folds used as validation set): best_score_

Acc_gds_rfc=gds_rfc.best_score_
print("Random Forest Classifier-Hyperparameters Model has an accuracy of : %.2f%%" % (Acc_gds_rfc * 100.0))
# SVM Hyperparameters
# grid_param dictionary

grid_param_svm = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', 'poly', 'sigmoid']}
# instance of the GridSearchCV class

gds_svm = GridSearchCV(svclf, grid_param_svm, cv=3, verbose=10) 
gds_svm.fit(train_X, train_y)
# Best score found (mean score on all folds used as validation set): best_score_

Acc_gds_svm=gds_svm.best_score_
print("SVM Classifier-Hyperparameters Model has an accuracy of : %.2f%%" % (Acc_gds_svm * 100.0))
# Best estimator: best_estimator_

gds_svm.best_estimator_
# Hyperparameters tested: param_grid

gds_svm.param_grid
# XGBoost Classifier
from xgboost import XGBClassifier
xgb_cl = XGBClassifier()

xgb_cl.fit(train_X, train_y)

y_pred = xgb_cl.predict(test_X)
Acc_xgb = accuracy_score(test_y, y_pred)

print("The XGBoost Classifier Model has an accuracy of : %.2f%%" % (Acc_xgb * 100.0))
# Performances Viz
df_Accuracy=pd.DataFrame(columns=["Algo_name","Accuracy"])
df_Accuracy["Algo_name"]=["Logistic Regression", "Random Forest Classifier", "Naive Bayes", "SVM Classifier", "Random Forest Classifier Hyp", "SVM Classifier Hyp", "XGBoost Classifier"]
df_Accuracy["Accuracy"]=[Acc_LR, Acc_RFC, Acc_NBC,Acc_svclf, Acc_gds_rfc, Acc_gds_svm, Acc_xgb]
df_Accuracy.sort_values('Accuracy', ascending=False)
fig, ax = plt.subplots(figsize=(12,7))

h=sns.barplot(x = "Accuracy", y = "Algo_name", data = df_Accuracy, ax=ax)

h.set_title("Performances")

plt.show()
# Prediction
##  Data load
test = pd.read_csv('../input/test.csv')
test.info()
# Missing Values
test["Age"]=test["Age"].fillna(test["Age"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].mean())
test=test.drop(["Cabin"], axis=1)
# Binary Variables
test["Sex_F"]=test["Sex"].apply(lambda x : 1 if x=="female" else 0 )
# Dummy Variables
X_Pclass=pd.get_dummies(test["Pclass"], prefix = 'Pclass_')
X_Embarked=pd.get_dummies(test["Embarked"], prefix = 'Embarked_')
test_cleaned=pd.concat([test, X_Pclass, X_Embarked], axis=1 )
test_cleaned.head()
# New features
test_cleaned["Child"]=test_cleaned["Age"].apply(lambda x : 1 if x<15 else 0 )

test_cleaned["Teenager"]=test_cleaned["Age"].apply(lambda x : 1 if (x>=15) and (x<25) else 0 )

test_cleaned["Adult"]=test_cleaned["Age"].apply(lambda x : 1 if (x>=25) & (x<65) else 0 )

test_cleaned["Old"]=test_cleaned["Age"].apply(lambda x : 1 if x>=65 else 0 )
test_cleaned['isAlone'] = ((test_cleaned['SibSp'] == 0) & (test_cleaned['Parch'] == 0)).apply(int)
# Data scaling
test_cleaned.head()
test_cleaned.columns
test_file=pd.DataFrame(test_cleaned, columns=['PassengerId', 'SibSp', 'Parch',

        'Fare', 'Sex_F', 'Pclass__2', 'Pclass__3',

       'Embarked__Q', 'Embarked__S', 'Child', 'Teenager', 'Adult', 'Old',

       'isAlone'])
test_file = test_file.set_index('PassengerId')
test_file = test_file.sort_index()
test_file.head()
test_file.info()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

test_file = sc.fit_transform(test_file) 
test_file
# Predict & Submit
y_pred = gds_svm.predict(test_file) 
submission=pd.concat([test_cleaned.iloc[:,0], pd.DataFrame(y_pred, columns=["Survived"])], axis=1)
submission.to_csv("submission.csv", index=False)