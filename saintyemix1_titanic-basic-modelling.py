# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install pandas==0.25.3
!pip install scikit-learn==0.22
!pip install numpy==1.18.0
!pip install ppscore
import numpy as np 
import pandas as pd
import sklearn
import matplotlib.pyplot as plt 
import seaborn as sns
import ppscore as pps
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import f1_score
from sklearn.externals import joblib

seed = 2020
print('numpy version: '+ np.__version__)
print('pandas version: '+ pd.__version__)
print('sklearn version: '+ sklearn.__version__)
pwd
path ='/kaggle/input/titanic/'
train = pd.read_csv(os.path.join(path,'train.csv'))
test = pd.read_csv(os.path.join(path,'test.csv'))
sample_sub = pd.read_csv(os.path.join(path,'gender_submission.csv'))
train.head()
test.head()
train.describe()
train.info()
# checking for missing values 
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(train.isnull(), cbar=False)
plt.show()
sns.countplot(train.Survived)
sns.barplot(x='Sex', y='Survived', data=train)
plt.ylabel("Rate of Surviving")
plt.title("Plot of Survival as function of Sex", fontsize=16)
plt.show()
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Pclass', y='Survived', data=train)
plt.ylabel("Survival Rate")
plt.title("Plot of Survival as function of Pclass", fontsize=16)
plt.show()
train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.figure(figsize=(10,5))
sns.heatmap(train.corr(),annot=True)
plt.figure(figsize=(10,5))
ax = train.corr()['Survived'].plot(kind='bar',title='correlation of target variable to features')
ax.set_ylabel('correlation')
# Predictive Power Score Plot
plt.figure(figsize=(20,10))
sns.heatmap(pps.matrix(train),annot=True)
train_copy = train.copy()
train_copy.dropna(inplace = True)
sns.distplot(train_copy.Age)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_copy)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train_copy)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")
train_null = train.isnull().sum()
test_null = test.isnull().sum()
print(train_null[train_null !=0])
print('-'*40)
print(test_null[test_null !=0])
from sklearn.impute import SimpleImputer
age_imp = SimpleImputer(strategy= 'median')
age_imp.fit(np.array(train.Age).reshape(-1,1))

train.Age = age_imp.transform(np.array(train.Age).reshape(-1,1))
test.Age = age_imp.transform(np.array(test.Age).reshape(-1,1))
train.head()
#save age imputer 
with open('age_imputer.joblib', 'wb') as f:
  joblib.dump(age_imp,f)
emb_imp = SimpleImputer(strategy= 'most_frequent' )
emb_imp.fit(np.array(train.Embarked).reshape(-1,1))

train.Embarked = emb_imp.transform(np.array(train.Embarked).reshape(-1,1))
test.Embarked = emb_imp.transform(np.array(test.Embarked).reshape(-1,1))
train.head()
#save embark imputer 
with open('embark_imputer.joblib', 'wb') as f:
  joblib.dump(emb_imp,f)
train.isnull().sum() 
print('-'*40)
test.isnull().sum()
drop_cols = ['PassengerId','Ticket','Cabin','Name']
train.drop(columns=drop_cols,axis=1,inplace = True)
test_passenger_id = test.PassengerId
test.drop(columns=drop_cols,axis=1,inplace = True)
test.fillna(value = test.mean(),inplace=True)
train.isnull().sum().any() , test.isnull().sum().any()
train['Number_of_relatives'] = train.Parch + train.SibSp
test['Number_of_relatives'] = test.Parch + test.SibSp

train.drop(columns=['Parch','SibSp'],axis=1,inplace=True)
test.drop(columns=['Parch','SibSp'],axis=1,inplace=True)
train.head()
gender_dic = {'male':1,'female':0}
train.Sex = train.Sex.map(gender_dic)
test.Sex = test.Sex.map(gender_dic)
train.head()
cat_col = ['Embarked', 'Pclass']
One_hot_enc = OneHotEncoder(sparse=False,drop='first',dtype=np.int)
encoded_train = pd.DataFrame(data=One_hot_enc.fit_transform(train[cat_col]), columns=['emb_2','emb_3','Pclass_2','Pclass_3'])
encoded_test = pd.DataFrame(data=One_hot_enc.transform(test[cat_col]),columns=['emb_2','emb_3','Pclass_2','Pclass_3'])
#save One_hot_enc 
with open('One_hot_enc.joblib', 'wb') as f:
  joblib.dump(One_hot_enc,f)
train.drop(columns=cat_col,axis=1,inplace=True)
test.drop(columns=cat_col,axis=1,inplace=True)

train = pd.concat([train,encoded_train],axis=1)
test = pd.concat([test,encoded_test],axis=1)
train.head()
features = test.columns
X = train[features]
y = train.Survived
scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)
#save scaler 
with open('scaler.joblib', 'wb') as f:
  joblib.dump(scaler,f)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
logistic_model  = LogisticRegression()
logistic_model.fit(X_train,y_train)
print('f1_score on training set: {}'.format(f1_score(logistic_model.predict(X_train),y_train)))
print('f1_score on test set: {}'.format(f1_score(logistic_model.predict(X_test),y_test)))
logistic_model.fit(X,y)
#save model 
with open('model-v1.joblib', 'wb') as f:
  joblib.dump(logistic_model,f)
