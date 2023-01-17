import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))
data=pd.read_csv('../input/cs-training.csv')
data.head()
print(data.isnull().sum())
data.describe()
data['MonthlyIncome'].fillna(data['MonthlyIncome'].mean(),inplace=True)
data['NumberOfDependents'].fillna(data['NumberOfDependents'].mode()[0], inplace=True)
print(data.isnull().sum())
import matplotlib.pyplot as plt

import seaborn as sns
cor=data.corr()

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,annot=True,ax=ax)
attributes=['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome']

sol=['SeriousDlqin2yrs']

X=data[attributes]

y=data[sol]
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier #Since the Dataset is Imbalanced, we go for boosting model.

model = XGBClassifier(tree_method = 'gpu_exact')

model.fit(X,y.values.ravel())

y_pred = model.predict(X)

print("The Accuracy score is : ",accuracy_score(y,y_pred)*100,"%")

print(confusion_matrix(y,y_pred))
from sklearn.model_selection import KFold,cross_val_score

kf = KFold(n_splits=5, random_state=None) 

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(np.mean(cross_val_score(model, X, y.values.ravel(), cv=10))*100)
data_test=pd.read_csv('../input/cs-test.csv')
print(data_test.isnull().sum())
data_test['MonthlyIncome'].fillna(data_test['MonthlyIncome'].mean(),inplace=True)
xtest=data_test[attributes]
xtest.head()
ytest=model.predict_proba(xtest)
print(ytest)
df=pd.DataFrame(ytest,columns=['Id','Probability'])
df.head()
ind=data['Unnamed: 0']
df['Id']=ind
df.head()
export_csv = df.to_csv('export_dataframe.csv',index = None,header=True)