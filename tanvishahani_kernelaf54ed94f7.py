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
train_df = pd.read_csv('/kaggle/input/summeranalytics2020/train.csv',index_col = 'Id')

sample_df = pd.read_csv('/kaggle/input/summeranalytics2020/Sample_submission.csv',index_col = 'Id')

test_df = pd.read_csv('/kaggle/input/summeranalytics2020/test.csv',index_col = 'Id')
train_df.head()
Att = train_df.Attrition
train_df.columns
train_df.describe()
train_df.dtypes
list1 = train_df.select_dtypes(include = object).columns.tolist()

train_dummy = pd.get_dummies(train_df,columns = list1,prefix = list1,drop_first = True)

test_dummy = pd.get_dummies(test_df,columns = list1,prefix = list1,drop_first = True)

#new_train=
train_dummy.shape
train_dummy.dtypes
train_dummy.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize = (20,20))

sns.heatmap(train_df.corr(),annot = True,fmt ='.1g',cmap = 'coolwarm')
df_object = train_df[train_df.select_dtypes(object).columns]

df_object.shape
fig, ax = plt.subplots(7,2,figsize = (15,30))

for i in np.arange(7):

    s = df_object.iloc[:,i]

    sns.countplot(s, ax = ax[i,0], order = s.value_counts().index.tolist())

    ax[i,0].set_title(df_object.columns[i]+' Training Dataset')

    ax[i,0].tick_params(labelrotation=90)

    sns.countplot(s, ax = ax[i,1], order = s.value_counts().index.tolist(), hue=Att)

    ax[i,1].tick_params(labelrotation=90)

    
df_int = train_df[train_df.select_dtypes(int).columns]

df_int.shape
df_int.nunique()
train_df.duplicated().sum(),test_df.duplicated().sum()
train_df.drop_duplicates(inplace = True)
sns.countplot(x = 'PerformanceRating',hue = 'Attrition',data = train_df)
train_df.PerformanceRating.describe()
train_df.PerformanceRating.value_counts()
train_df.drop(['Behaviour','PerformanceRating'],axis = 1,inplace = True)

test_df.drop(['Behaviour','PerformanceRating'],axis = 1,inplace = True)
train_df.columns
import matplotlib.pyplot as plt

import seaborn as sns
train_df.Age.unique()
train_df['age_bins'] = pd.cut(x=train_df['Age'], bins=[18,20,29,39,49,60],labels=['Teens','20s','30s','40s','50s'])

test_df['age_bins'] = pd.cut(x=test_df['Age'], bins=[18,20,29,39,49,60],labels=['Teens','20s','30s','40s','50s'])

age = train_df.groupby('age_bins')

age.groups.keys()
train_df.head()
sns.countplot(x = 'age_bins',hue = 'Attrition',data = train_df)
train_df.YearsAtCompany.unique()
train_df['years_in_company'] = pd.cut(x=train_df['YearsAtCompany'], bins=[0,5,10,15,20,25,30,37],labels=['0-5','5-10','10-15','15-20','20-25','25-30','30-35'])

test_df['years_in_company'] = pd.cut(x=test_df['YearsAtCompany'], bins=[0,5,10,15,20,25,30,37],labels=['0-5','5-10','10-15','15-20','20-25','25-30','30-35'])
sns.countplot(x = 'years_in_company',hue = 'Attrition',data = train_df)
train_df.TotalWorkingYears.unique()
train_df['years_in_working'] = pd.cut(x=train_df['TotalWorkingYears'], bins=[0,5,10,15,20,25,30,39],labels=['0-5','5-10','10-15','15-20','20-25','25-30','30-35'])

test_df['years_in_working'] = pd.cut(x=test_df['TotalWorkingYears'], bins=[0,5,10,15,20,25,30,39],labels=['0-5','5-10','10-15','15-20','20-25','25-30','30-35'])

sns.countplot(x = 'years_in_working',hue = 'Attrition',data = train_df)

#TotalWorkingYears',

#       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
train_df.shape,test_df.columns
plt.scatter(x= train_df.YearsAtCompany,y = train_df.MonthlyIncome)
train_df.columns
train_df.dtypes
#now since we already have bins for 3 categories lets remove them

new_train = train_df.drop(['Age','YearsAtCompany','TotalWorkingYears'],axis =1,inplace = True)

new_test = test_df.drop(['Age','YearsAtCompany','TotalWorkingYears'],axis =1,inplace = True)
#new_train.dtypes

#new_train = train_df
from sklearn.model_selection import train_test_split,cross_val_score

train_df.dtypes

list1 = train_df.select_dtypes(include = object).columns.tolist()

train_dummy1 = pd.get_dummies(train_df,columns = list1,prefix = list1,drop_first = True)

list2 = train_dummy1.select_dtypes(include = 'category').columns.tolist()

train_dummy2 = pd.get_dummies(train_dummy1,columns = list2,prefix = list2,drop_first = True)

list3 = test_df.select_dtypes(include = object).columns.tolist()

test_dummy1 = pd.get_dummies(test_df,columns = list3,prefix = list3,drop_first = True)

list4 = test_dummy1.select_dtypes(include = 'category').columns.tolist()

test_dummy2 = pd.get_dummies(test_dummy1,columns = list4,prefix = list4,drop_first = True)
Y = train_dummy2.Attrition

X = train_dummy2.drop(['Attrition'],axis = 1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .2)
train_dummy2.shape,test_dummy2.shape
train_dummy2.dtypes
from sklearn.ensemble import RandomForestClassifier

onemodel1 = RandomForestClassifier(n_estimators=400,n_jobs = -1,min_samples_leaf=10)

onemodel1.fit(x_train,y_train)
onemodel1.score(x_test,y_test)
cross_val_score(onemodel1,x_train,y_train,cv = 5,n_jobs = -1,verbose = 1,scoring = 'roc_auc').mean()
onemodel2 = RandomForestClassifier(n_estimators=400,n_jobs = -1,min_samples_leaf=10)

onemodel2.fit(X_train,Y_train)
cross_val_score(onemodel2,X_train,Y_train,cv = 5,n_jobs = -1,verbose = 1,scoring = 'roc_auc').mean()
onemodel2.score(X_test,Y_test)
onemodel2.predict_proba(test_dummy2)
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



onemodel5 = LGBMClassifier(random_state=7, n_estimators=100, colsample_bytree=0.5, 

                       max_depth=2, learning_rate=0.1, boosting_type='gbdt')

cross_val_score(onemodel5, X_train, Y_train, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
X_train.shape,test_dummy2.shape
onemodel5.fit(X_train,Y_train)

onemodel5.predict_proba(test_dummy2)
onemodel4 = XGBClassifier(seed=7, n_jobs=-1, n_estimators=100, random_state=7, max_depth=2, learning_rate=0.1)

cross_val_score(onemodel4, X_train, Y_train, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(estimators=[('onemodel4', onemodel4), ('onemodel5', onemodel5),('onemodel2', onemodel2)],

                                         voting='soft', n_jobs=-1)

cross_val_score(ensemble, X, Y, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
ensemble.fit(X, Y)

y_pred = ensemble.predict_proba(test_dummy2)[:, 1]

sub_df = pd.DataFrame({"Id":test_dummy2.index, "Attrition": y_pred})

sub_df.to_csv("SA_submission_1.csv", index=False)
train_df.dtypes
train_dummy.dtypes
train_dummy.drop(['Behaviour','PerformanceRating'],axis = 1,inplace = True)

test_dummy.drop(['Behaviour','PerformanceRating'],axis = 1,inplace = True)
train_dummy.drop_duplicates(inplace = True)
Y = train_dummy.Attrition

X = train_dummy.drop(['Attrition'],axis = 1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .2)

onemodel6 = RandomForestClassifier(n_estimators=400,n_jobs = -1,min_samples_leaf=10)

onemodel6.fit(X_train,Y_train)
#for submission

cross_val_score(onemodel6,X_train,Y_train,cv = 5,n_jobs = -1,verbose = 1,scoring = 'roc_auc').mean()
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



lgbmc = LGBMClassifier(random_state=7, n_estimators=100, colsample_bytree=0.5, 

                       max_depth=2, learning_rate=0.1, boosting_type='gbdt')

cross_val_score(lgbmc, X_train, Y_train, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
xgbc = XGBClassifier(seed=7, n_jobs=-1, n_estimators=100, random_state=7, max_depth=2, learning_rate=0.1)

cross_val_score(xgbc, X_train, Y_train, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(estimators=[('onemodel6', onemodel6), ('xgbc', xgbc), ('lgbmc', lgbmc)],

                                         voting='soft', n_jobs=-1)

cross_val_score(ensemble, X, Y, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
ensemble.fit(X, Y)

y_pred = ensemble.predict_proba(test_dummy)[:, 1]

sub_df = pd.DataFrame({"Id":test_dummy.index, "Attrition": y_pred})

sub_df.to_csv("SA_submission_2.csv", index=False)