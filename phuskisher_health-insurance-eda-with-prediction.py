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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
df=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
df.head()
for i in df.columns:
    rows=df.shape[0]
    cols=df.shape[1]
print(f'total number of rows are {rows}')
print(f'total number of columns are {cols}')
#info
df.info()

# drop not necessary columns
df.drop('id', axis=1, inplace=True)
#details about data
df.describe()


## cat and num columns
cat_cols=[i for i in df.columns if df[i].dtype=='object']
num_cols=[j for j in df.columns if (df[j].dtype=='int64')|(df[j].dtypes=='float64')]
print(f'Categorical columns are {cat_cols}\n')
print(f'Numerical columns are {num_cols}')
#to find nan values
df.isna().sum()
#correlation between the columns
df.corr()
#unique values and number of unique values
for vars in df.columns:
    print(f'unique values of {vars} are {df[vars].unique()}\n')
    print(f'number of unique values of {vars} are {df[vars].nunique()}\n')
# categorical values
cat_cols
fig, axes=plt.subplots(2,2, figsize=(10,10))
for i, j in enumerate(cat_cols):
    ax=axes[int(i/2),i%2]
    sns.countplot(df[j], ax=ax)
fig.delaxes(axes[1,1])


# driving_lisensce holders(1= with DL,0=without DL)
df['Driving_License'].value_counts().plot(kind='bar')
#previously insured data(1=insured, 0=not insured)
df['Previously_Insured'].value_counts().plot(kind='bar')
#kde plot of data
kde_data=['Age','Annual_Premium','Vintage']
fig, axes=plt.subplots(2,2, figsize=(10,10))
for i,j in enumerate(kde_data):
    ax=axes[int(i/2), i%2]
    sns.kdeplot(df[j], ax=ax)
fig.delaxes(axes[1,1])
#to find outliers box plot
num_cols
fig, axes=plt.subplots(4,2, figsize=(20,10))
for i,j in enumerate(num_cols):
    ax=axes[int(i/2), i%2]
    sns.boxplot(df[j],ax=ax)
df.head()
## bivariate plots
plt.figure(figsize=(20,6))
sns.lineplot(x=df['Age'],y=df['Annual_Premium'])

plt.figure(figsize=(20,6))
sns.lineplot(x=df['Age'],y=df['Vintage'])
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True)
wrt_res_data=['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']
fig, axes=plt.subplots(3,2, figsize=(20,10))
for i, j in enumerate(wrt_res_data):
    ax=axes[int(i/2),i%2]
    sns.countplot(df[j], hue=df['Response'] ,ax=ax)
fig.delaxes(axes[2,1])

df.reset_index(drop='First')
change={'> 2 Years':'grt2','1-2 Year':'1to2','< 1 Year':'less1'}
df['Vehicle_Age']=df['Vehicle_Age'].map(change)
df.head()
dummies=['Gender','Vehicle_Age','Vehicle_Damage']
df_encoding_1=pd.get_dummies(df['Gender'])
df_encoding_2=pd.get_dummies(df['Vehicle_Age'])
df_encoding_3=pd.get_dummies(df['Vehicle_Damage'])
df_new=pd.concat([df,df_encoding_1,df_encoding_2,df_encoding_3], axis=1)
df_new.head()
df_new.drop(['Gender','Vehicle_Age','Vehicle_Damage'], axis=1, inplace=True)
df_new.head()
df_new.info()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
y=df_new['Response']
X=df_new.drop('Response', axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
lor=LogisticRegression()
lor.fit(X_train,y_train)
y_pred_lor=lor.predict(X_test)

lor.score(X_test,y_test)
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred_dtc=dtc.predict(X_test)
dtc.score(X_test,y_test)
hdtc=DecisionTreeClassifier(ccp_alpha= 1,
 max_depth= 7.0,
 min_samples_leaf= 4,
 min_samples_split= 10)
hdtc.fit(X_train,y_train)
y_pred_hdtc=hdtc.predict(X_test)
hdtc.score(X_test,y_test)
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred_rfc=rfc.predict(X_test)
rfc.score(X_test,y_test)
import re
xgb=XGBClassifier()

xgb.fit(X_train,y_train)
y_pred_xgb=xgb.predict(X_test)
xgb.score(X_test,y_test)
nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred_nb=nb.predict(X_test)
nb.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
cm=confusion_matrix(y_test,y_pred_lor)
sns.heatmap(cm, annot=True)
fpr,tpr, threshold=roc_curve(y_test,y_pred_dtc)
roc_auc=auc(fpr,tpr)
roc_auc
fpr, tpr, thresholds = roc_curve(y_test,y_pred_dtc)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label = 'AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
