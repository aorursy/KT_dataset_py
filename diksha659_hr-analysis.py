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
#importing Libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Importing Datasets

train = pd.read_csv("../input/hranalysis/train.csv")

test = pd.read_csv("../input/hranalysis/test.csv")
train.head()
test.head()
train.shape
train.isnull().any()
import missingno as msno

msno.matrix(train)
train.info()
train.describe().T
import pandas_profiling

train.profile_report()
dict(train.dtypes)
#Heatmap

plt.figure(figsize=(10,10))

sns.heatmap(train.corr(),annot=True,cmap='Greys')
#Pairplot

plt.figure(figsize=(20,20))

plt.style.use("fivethirtyeight")

sns.pairplot(train,palette='rainbow')

plt.show()
#Lets explore Categorical Variables

#Department

plt.figure(figsize=(6,6))

train.department.value_counts().plot(kind='pie',subplots=True)

plt.ylabel("")

plt.title("Department")
#Region

plt.figure(figsize=(10,5))

train.region.value_counts().plot(kind='bar',color='g')

plt.xlabel('Region')

plt.ylabel('count')

plt.show()
#Education

plt.figure(figsize=(10,5))

sns.countplot('education',data=train)

plt.xlabel("Education")

plt.ylabel("Count")

plt.title("Different Education")

plt.legend()

plt.show()
#Donut chart on gender

train.gender.value_counts()

labels = ['Male','Female']

colors = ['b','r']

sizes = [55092,23206]

plt.figure(figsize=(6,6))

plt.pie(sizes, labels=labels, colors=colors, autopct = '%.2f%%',shadow=True)

#Now make circle at the centre to make it donut

circle = plt.Circle((0,0),0.75,facecolor='white',edgecolor='black')

fig=plt.gcf()  #gcf represent "get current figure"

fig.gca().add_artist(circle)

plt.legend()

plt.show()
#recruitment channel

plt.figure(figsize=(6,6))

train.recruitment_channel.value_counts().plot(kind='pie')

plt.ylabel("")

plt.title("Recritment Channel")

plt.legend()

plt.show()
#Now Exploring Numerical Variables

#Age

plt.figure(figsize=(10,5))

sns.distplot(train.age,color='purple')

plt.xlabel("Age")

plt.title("Age of Employees")

plt.show()
#Previous year rating

plt.figure(figsize=(10,5))

train.previous_year_rating.value_counts().plot(kind='bar',color=['green','purple','red','orange','black'])

plt.xlabel("Ratings")

plt.ylabel("Count")

plt.title("Previous Year Rating of Employees")

plt.legend()

plt.show()
#Length of Service

plt.figure(figsize=(10,5))

sns.distplot(train['length_of_service'],color='orange')

plt.xlabel("Length of Service")

plt.title("Length of Service of Employee")

plt.legend()

plt.show()
# KPIs_met >80%

plt.figure(figsize=(6,6))

train['KPIs_met >80%'].value_counts().plot(kind='pie',subplots=True,explode=(0,0.05),labels = ['KPIs_met <80%','KPIs_met >80%'])

plt.ylabel("")

plt.title("KPIs_met >80%")

plt.legend()

plt.show()
#Award Won

plt.figure(figsize=(7,5))

train['awards_won?'].value_counts().plot(kind='bar',color=['red','blue'])

plt.xlabel('Award won')

plt.ylabel('Count')

plt.title("Employee who won Awards")

plt.show()
#Average Training Score

plt.figure(figsize=(10,5))

sns.distplot(train.avg_training_score,color='green')

plt.xlabel("Average Training Score")

plt.title("Average Training Score of Employees")

plt.legend()

plt.show()
#Number of Trainings

plt.figure(figsize=(10,5))

sns.countplot('no_of_trainings',data=train)

plt.xlabel("Number of Trainings")

plt.ylabel("Count")

plt.title("Number of Trainings taken by Employees")

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x = 'education', hue='gender',data=train,palette='twilight')

plt.xlabel("Education")

plt.ylabel("Count")

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x = 'recruitment_channel', hue='gender',data=train,palette='Greys')

plt.xlabel("Recruitment_channel")

plt.ylabel("Count")

plt.show()
#Avg training score and Is promoted



ct = pd.crosstab(train['avg_training_score'], train['is_promoted'])

ct.plot.bar(stacked=True,figsize=(15,5),color=['purple','green'])

plt.legend()

plt.xlabel('Average Training Score')
#Awards won and is promoted



ct = pd.crosstab(train['awards_won?'],train['is_promoted'])

ct.div(ct.sum(1).astype('float'),axis=0).plot(kind='bar',stacked=True,color=['pink','red'])

plt.legend()

plt.xlabel('Awards Won')

plt.title("Awards Won vs Is Promoted")
#Region vs Is Promoted

ct = pd.crosstab(train['region'], train['is_promoted'])

ct.div(ct.sum(1).astype('float'),axis=0).plot.bar(stacked=True,figsize=(15,5),color=['blue','cyan'])

plt.legend()

plt.xlabel('region')
#length of service vs is promoted

ct = pd.crosstab(train['length_of_service'], train['is_promoted'])

ct.plot.bar(stacked=True,figsize=(15,5),color=['purple','green'])

plt.legend()

plt.xlabel('Length of Service')
#Previous year rating vs promotion

ct = pd.crosstab(train['previous_year_rating'], train['is_promoted'])

ct.plot.bar(stacked=True,figsize=(15,5),color=['purple','green'])

plt.legend()

plt.xlabel('Previous Year Rating')
train.head()
dict(train.dtypes)
#Divide Numerical and categorical variables

numeric_var_names = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['float32','float64','int32','int64']]

cat_var_names = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object','O']]
hr_num = train[numeric_var_names]

hr_cat = train[cat_var_names]
#create Data Audit Report

def var_summary(x):

    return pd.Series([x.count(),x.isnull().sum(),x.sum(),x.var(),x.std(),x.mean(),x.median(),x.min(),x.dropna().quantile(0.01),x.dropna().quantile(0.05),

              x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75),x.dropna().quantile(0.90),

              x.dropna().quantile(0.95),x.dropna().quantile(0.99)],index=['N','NMISS','SUM','VAR','STD','MEAN','MEDIAN','MIN','P1','P5','P10','P25','P50','P75','P90','P95','P99'])

num_summary = hr_num.apply(lambda x : var_summary(x)).T

num_summary



#As we can see that there are no outliers.
#Missing Value Treatment

hr_num['previous_year_rating'].fillna(3,inplace=True)

hr_num['previous_year_rating'].isnull().sum()

test['previous_year_rating'].fillna(3,inplace=True)

test['previous_year_rating'].isnull().sum()
def cat_summary(x):

    return pd.Series([x.count(),x.isnull().sum(),x.value_counts()],index=['N','NMISS','COUNT'])



cat_summary = hr_cat.apply(lambda x : cat_summary(x)).T

cat_summary
#Missing value Treatment of categorical variables



hr_cat.education.fillna(hr_cat.education.mode()[0],inplace=True)

hr_cat.education.isnull().sum()

test.education.fillna(test.education.mode()[0],inplace=True)

test.education.isnull().sum()
#Converting Categorical variables into Numeric by label encoding

from sklearn.preprocessing import LabelEncoder

hr_cat['department'] = LabelEncoder().fit_transform(hr_cat['department'])

hr_cat['region'] = LabelEncoder().fit_transform(hr_cat['region'])

hr_cat['education'] = LabelEncoder().fit_transform(hr_cat['education'])

hr_cat['gender'] = LabelEncoder().fit_transform(hr_cat['gender'])

hr_cat['recruitment_channel'] = LabelEncoder().fit_transform(hr_cat['recruitment_channel'])
hr = pd.DataFrame(pd.concat([hr_num,hr_cat],axis=1))

hr.head()
#Dropping employee id column

hr.drop(['employee_id'],axis=1,inplace=True)
test.drop(['employee_id'],axis=1,inplace=True)

test.head()
#Converting Categorical variables into Numeric by label encoding

from sklearn.preprocessing import LabelEncoder

test['department'] = LabelEncoder().fit_transform(test['department'])

test['region'] = LabelEncoder().fit_transform(test['region'])

test['education'] = LabelEncoder().fit_transform(test['education'])

test['gender'] = LabelEncoder().fit_transform(test['gender'])

test['recruitment_channel'] = LabelEncoder().fit_transform(test['recruitment_channel'])
test.columns
#Predictive Modelling

feature_columns = hr.columns.difference(['is_promoted'])

feature_columns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



train_x, val_x, train_y, val_y = train_test_split(hr[feature_columns], hr['is_promoted'], test_size=0.2)

print(train_x.shape)

print(train_y.shape)

print(val_x.shape)

print(val_y.shape)
logreg = LogisticRegression()

logreg.fit(train_x,train_y)
logreg.coef_
list(zip(feature_columns, logreg.coef_[0]))
pred_y = logreg.predict(test)
pred_y
#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

rf.fit(train_x,train_y)



rf_pred = rf.predict(test)

print("Accuracy " +str(rf.score(train_x,train_y)))
#CatBoost Classifier

from catboost import CatBoostClassifier



cb = CatBoostClassifier()

cb.fit(train_x,train_y)



cb_pred = cb.predict(test)



print("Accuracy " +str(cb.score(train_x,train_y)))
from lightgbm import LGBMClassifier

lgb = LGBMClassifier()

lgb.fit(train_x, train_y)



lgb_pred = lgb.predict(test)



print("Accuracy "+str(lgb.score(train_x, train_y)))