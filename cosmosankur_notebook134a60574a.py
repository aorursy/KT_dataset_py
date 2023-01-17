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
df = pd.read_csv('/kaggle/input/clicks-conversion-tracking/KAG_conversion_data.csv')
df.head()
df = df.drop(['ad_id','xyz_campaign_id','fb_campaign_id'],axis=1)
df.head(3)
df['Approved_Conversion'].value_counts()
df['gender'].value_counts()
df['age'].value_counts()
import seaborn as sns
#check if we have empty values or not

sns.heatmap(df.isnull())
import seaborn as sns
sns.countplot(x='Approved_Conversion',hue='gender',data=df)
sns.distplot(df['Clicks'],kde=False,bins=30)
sns.distplot(df['interest'],kde=False,bins=100)
sns.countplot(x='Approved_Conversion',hue='age',data=df)
sns.distplot(df['Spent'],bins=30)
df['gender'].value_counts()
df = df.replace('M',1)

df = df.replace('F',0)
df['age'].value_counts()
df = df.replace('30-34',0.0011)

df = df.replace('35-39',0.0012)

df = df.replace('40-44',0.0013)

df = df.replace('45-49',0.0014)

df.head()
df.columns
x = df[['age', 'gender', 'interest', 'Impressions', 'Clicks', 'Spent',

       'Total_Conversion']]

y = df['Approved_Conversion']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(x_train,y_train)

linear.score(x_test,y_test)
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()

tree.fit(x_train,y_train)

tree.score(x_test,y_test)
from sklearn.linear_model import Ridge

rid = Ridge(alpha=1)

rid.fit(x_train,y_train)

rid.score(x_test,y_test)
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(x_train,y_train)

forest.score(x_test,y_test)

from sklearn.ensemble import GradientBoostingRegressor

boost = GradientBoostingRegressor()

boost.fit(x_train,y_train)

boost.score(x_test,y_test)
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor()

ada.fit(x_train,y_train)

ada.score(x_test,y_test)
ada.feature_importances_
from sklearn.metrics import r2_score

from rfpimp import permutation_importances



def r2(forest, x_train, y_train):

    return r2_score(y_train, forest.predict(x_train))



perm_imp_rfpimp = permutation_importances(forest, x_train, y_train, r2)
perm_imp_rfpimp
from sklearn.metrics import r2_score

from rfpimp import permutation_importances



def r2(linear, x_train, y_train):

    return r2_score(y_train, linear.predict(x_train))



perm_imp_rfpimp_linear = permutation_importances(linear, x_train, y_train, r2)
perm_imp_rfpimp_linear
from sklearn.metrics import r2_score

from rfpimp import permutation_importances



def r2(rid, x_train, y_train):

    return r2_score(y_train, rid.predict(x_train))



perm_imp_rfpimp_rid = permutation_importances(rid, x_train, y_train, r2)
perm_imp_rfpimp_rid
x_new = df[[ 'interest', 'Impressions', 'Clicks', 'Spent',

       'Total_Conversion']]

y = df['Approved_Conversion']
x_train_new,x_test_new,y_train_new,y_test_new = train_test_split(x_new,y,test_size=0.3,random_state=101)
linear.fit(x_train_new,y_train_new)

linear.score(x_test_new,y_test_new)
rid.fit(x_train_new,y_train_new)

rid.score(x_test_new,y_test_new)
forest.fit(x_train_new,y_train_new)

forest.score(x_test_new,y_test_new)