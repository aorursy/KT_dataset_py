# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('/kaggle/input/wns-inno/train_LZdllcl.csv')
df1=pd.read_csv('/kaggle/input/wns-inno/test_2umaH9m.csv')
df.head(3)
df1.head(3)
df.shape
df.describe()
df.info()
df.is_promoted.value_counts()
df.is_promoted.value_counts(normalize=True)
df.isnull().sum()/len(df)
df.isnull().sum().sum()/len(df)
a=df[df['previous_year_rating'].isnull()]
a.head()
a['length_of_service'].value_counts()
df['previous_year_rating'].fillna(value=0,inplace=True)
df.isnull().sum()
for i in df[df['education'].isnull()]['KPIs_met >80%']==0:
    df['education'].fillna(value="Bachelor's",inplace=True)
for i in df[df['education'].isnull()]['KPIs_met >80%']==1:
    df['education'].fillna(value="Master's & above",inplace=True)
df['education'].isnull().sum()
df.isnull().sum()
df.drop(['employee_id'],axis=1,inplace=True)
df.shape
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True,linewidths=0.8)
plt.show()
df_new=pd.get_dummies(df,['department','region','education','gender','recruitment_channel'],drop_first=True)
df_new.head(2)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score
X=df_new.drop('is_promoted',axis=1)
y=df_new['is_promoted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lr=LogisticRegression(max_iter=10000)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print("The Train score of Logistic Regression is :",lr.score(X_train,y_train))
print("The Test score of Logistic Regression is :",lr.score(X_test,y_test))
print("The accuracy of Logistic Regression is ",accuracy_score(y_test,pred))
print("The f1 score of Logistic Regression is ",f1_score(y_test,pred))
print("The Confusion Matrix of Logistic Regression is \n \n",confusion_matrix(y_test,pred))
print("\n")
print("The Classification Report of Logistic Regression is \n \n",classification_report(y_test,pred))
rf = RandomForestClassifier(random_state=101)
rf.fit(X_train,y_train)
pred_rf=rf.predict(X_test)
print("The Train score of Random Forest Classifier is :",rf.score(X_train,y_train))
print("The Test score of Random Forest Classifier is :",rf.score(X_test,y_test))
print("The accuracy of Random Forest Classifier is ",accuracy_score(y_test,pred_rf))
print("The f1 score of Random Forest Classifier is ",f1_score(y_test,pred_rf))
print("The Confusion Matrix of Random Forest Classifier is \n \n",confusion_matrix(y_test,pred_rf))
print("\n")
print("The Classification Report of Random Forest Classifier is \n \n",classification_report(y_test,pred_rf))
xgbc=xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_xgbc=xgbc.predict(X_test)
print("The Train score of XGBoosting is :",xgbc.score(X_train,y_train))
print("The Test score of XGBoosting is :",xgbc.score(X_test,y_test))
print("The accuracy of XGBoosting is ",accuracy_score(y_test,pred_xgbc))
print("The f1 score of XGBoosting is ",f1_score(y_test,pred_xgbc))
print("The Confusion Matrix of XGBoosting is \n \n",confusion_matrix(y_test,pred_xgbc))
print("\n")
print("The Classification Report of XGBoosting is \n \n",classification_report(y_test,pred_xgbc))
df1.head()
df1.info()
df1.isnull().sum()
df1.isnull().sum().sum()/len(df1)
b=df1[df1['previous_year_rating'].isnull()]
b.head()
b['length_of_service'].value_counts()
df1['previous_year_rating'].fillna(value=0,inplace=True)
df1.isnull().sum()
for i in df1[df1['education'].isnull()]['KPIs_met >80%']==0:
    df1['education'].fillna(value="Bachelor's",inplace=True)
for i in df1[df1['education'].isnull()]['KPIs_met >80%']==1:
    df1['education'].fillna(value="Master's & above",inplace=True)
df1['education'].isnull().sum()
df1.isnull().sum()
df_test=pd.get_dummies(df1,['department','region','education','gender','recruitment_channel'],drop_first=True)
df_test.head(2)
emp_id=df_test['employee_id']
df_test.drop(['employee_id'],axis=1,inplace=True)
X_test1=df_test
test_pred=xgbc.predict(X_test1)
final=pd.DataFrame()
final['employee_id']=pd.Series(emp_id)
final['is_promoted']=pd.Series(test_pred)
final