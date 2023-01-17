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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
file= pd.ExcelFile('../input/Bank_Personal_Loan_Modelling.xlsx')
description=pd.read_excel(file, 'Description')
df=pd.read_excel(file, 'Data')
description.head(10)
description.drop('Unnamed: 0',axis=1,inplace=True)
description.drop(axis=0,index=[0,1,2,3,4],inplace=True)
description.rename(columns={'Unnamed: 1':'Column Name','Unnamed: 2':'Column Description'}, inplace=True)
pd.set_option('display.max_colwidth',-1)
print(description)
df.head()
df.info()
df.describe()

df_loan_accept=df[df['Personal Loan']==1]
sns.set_style('darkgrid')
g=sns.FacetGrid(df,row='Education',col='Family',hue='Personal Loan',palette='Set2')
g=g.map(plt.hist, 'CCAvg', alpha=0.5)
plt.legend(bbox_to_anchor=(1.7,3))
sns.set_style('white')
sns.countplot(data=df,x='Education',hue='Personal Loan',palette='RdBu_r')
sns.barplot('Education','Mortgage',hue='Personal Loan',data=df,palette='viridis',ci=None)
plt.legend(bbox_to_anchor=(1.2,1))
sns.set_style('white')
sns.countplot(data=df,x='Securities Account',hue='Personal Loan',palette='Set2')
plt.figure(figsize=(10,6))
sns.boxplot('CreditCard','CCAvg',hue='Personal Loan',data=df,palette='RdBu_r')
plt.legend(bbox_to_anchor=(1.2,1))
df.columns
X=pd.DataFrame(columns=['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','CreditCard','Online'],data=df)
y=df['Personal Loan']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y)
from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier(max_leaf_nodes=3)
dtree.fit(X_train,y_train)
predictions= dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
plt.figure(figsize=(9,7))
sns.distplot(y_test-predictions)


