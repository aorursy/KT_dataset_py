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
from sklearn.model_selection import train_test_split, cross_val_score

from xgboost.sklearn import XGBClassifier

from sklearn import metrics   #Additional scklearn functions

from sklearn.model_selection import GridSearchCV 

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
train.head()
df = train.copy()
df.info()
df.describe()
df.isnull().sum()
sns.set_style('whitegrid')

plt.figure(figsize=(12,8))

sns.countplot(x = 'Gender', hue = 'Vehicle_Damage', data=df)
plt.figure(figsize=(12,8))

sns.countplot(x = 'Vehicle_Age', hue = 'Vehicle_Damage', data=df)
plt.figure(figsize=(12,8))

print(df['Age'].value_counts()[:5])

sns.distplot(df.Age, color='darkred')
print(f"Youngest Customer's age : {df['Age'].min()}")

print(f"Oldest Customer's age : {df['Age'].max()}")
pd.set_option('display.max_rows', None)

age_range = pd.DataFrame(df['Age'].value_counts())

age_range
bins = [18, 30, 40, 50, 60, 70, 120]

labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']

df['Age_Range'] = pd.cut(df.Age, bins, labels = labels,include_lowest = True)
plt.figure(figsize=(12,8))

sns.countplot(x = 'Age_Range', hue = 'Vehicle_Damage', data=df)
plt.figure(figsize=(12,8))

sns.countplot(x = 'Gender', hue = 'Response', data=df)
plt.figure(figsize=(12,8))

sns.countplot(x = 'Gender', hue = 'Previously_Insured', data=df)
df['Driving_License'].value_counts()
plt.figure(figsize=(24,8))

sns.countplot(x = 'Region_Code', hue = 'Vehicle_Damage', data=df)
plt.figure(figsize=(24,8))

sns.countplot(x = 'Region_Code', hue = 'Response', data=df)
plt.figure(figsize=(12,8))

sns.countplot(x = 'Previously_Insured', hue = 'Response', data=df)
plt.figure(figsize=(12,8))

sns.countplot(x = 'Age_Range', hue = 'Response', data=df)
print(df['Annual_Premium'].value_counts().head(15))

plt.figure(figsize=(12,8))

sns.distplot(df['Annual_Premium'])
plt.figure(figsize=(12,8))

sns.countplot(x = 'Vehicle_Age',data=df)
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True)
df.corr()['Response']
df.info()
df['Vehicle_Age'].value_counts()
df=pd.concat([df,pd.get_dummies(df['Vehicle_Damage'],prefix='Vehicle_Damage')],axis=1).drop(['Vehicle_Damage'],axis=1)
df.head()
df=pd.concat([df,pd.get_dummies(df['Gender'],prefix='Gender')],axis=1).drop(['Gender'],axis=1)
df.head()
df.drop(['Vehicle_Damage_No','Gender_Female'], axis=1, inplace=True)

df.head()
df['Vehicle_Age'] = pd.Categorical(df['Vehicle_Age'].values).codes

df['Age_Range'] = pd.Categorical(df['Age_Range'].values).codes

df.head()
df.info()
df.shape
X = df.drop(['Response','id', 'Age_Range'], axis=1)

y = df['Response']
scaler = MinMaxScaler()

scaled_X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(scaled_X)

scaled_X.columns = X.columns

scaled_X.head()
print(y.value_counts())

y.value_counts().plot(kind='bar')
from collections import Counter

from imblearn.over_sampling import SMOTE

print('Original dataset shape %s' % Counter(y))

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(scaled_X, y)

print('Resampled dataset shape %s' % Counter(y_res))
X_res = pd.DataFrame(X_res)

y_res = pd.Series(y_res)

X_res.columns = scaled_X.columns

X_res.head()
print(y_res.value_counts())

y_res.value_counts().plot(kind='bar')
X_res.shape
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.3, random_state = 42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
np.random.seed(42)

xgb1 = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1,

gamma=0,subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',nthread=4,

scale_pos_weight=1,seed=27)

xgb1.fit(X_train, y_train)
np.random.seed(42)

etc = ExtraTreesClassifier(n_estimators=200).fit(X_train, y_train)

print(etc.score(X_test, y_test))

etc_pred = etc.predict(X_test)

print(classification_report(etc_pred, y_test))
np.random.seed(42)

rf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)

print(rf.score(X_test, y_test))

rf_pred = rf.predict(X_test)

print(classification_report(rf_pred, y_test))
print(f"ROC_AUC Score of XGBoost Classifier is : {metrics.roc_auc_score(xgb1.predict(X_test), y_test)}")

print(f"ROC_AUC Score of RandomForestClassifier is : {metrics.roc_auc_score(rf.predict(X_test), y_test)}")

print(f"ROC_AUC Score of ExtraTreesClassifier is : {metrics.roc_auc_score(etc.predict(X_test), y_test)}")
test.head()
t_copy = test.copy()
t_copy.head()
t_copy=pd.concat([t_copy,pd.get_dummies(t_copy['Gender'],prefix='Gender')],axis=1).drop(['Gender'],axis=1)

t_copy=pd.concat([t_copy,pd.get_dummies(t_copy['Vehicle_Damage'],prefix='Vehicle_Damage')],axis=1).drop(['Vehicle_Damage'],axis=1)
t_copy.head()
t_copy['Vehicle_Age'] = pd.Categorical(t_copy['Vehicle_Age'].values).codes

t_copy.head()
X_train.columns
t_copy.columns
t_copy.drop(['id','Vehicle_Damage_No','Gender_Female'], axis=1, inplace=True)
scaled_test = scaler.transform(t_copy)

scaled_test = pd.DataFrame(scaled_test)

scaled_test.columns = t_copy.columns

scaled_test.head()
pred = etc.predict(scaled_test)

f_pred = pd.concat([pd.DataFrame(test['id']),pd.DataFrame(pred)], axis=1)

f_pred.columns = ['id','Response'] 

f_pred.head()

f_pred.to_csv('Submission.csv')