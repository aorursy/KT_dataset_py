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

plt.style.use('bmh')
df = pd.read_csv('../input/campaign/assign.csv')
df.head()
df.shape

df.info()
df.describe()
print(list(df.columns))
# Counting the unique values of each column
uniqueValues = df.nunique()
print('Count of unique value sin each column :')
print(uniqueValues)
df = df.drop(['product','phase'], axis=1)
df1=df.copy()
df1.head()

for c in df.columns:
    print("---- %s ---" % c)
    print(df[c].value_counts())

grouped_multiple = df.groupby(['campaign_platform','campaign_type','communication_medium']).agg({'age': ['count']})
grouped_multiple.columns = ['c']
grouped_multiple = grouped_multiple.reset_index()
print(grouped_multiple)

sns.countplot(df.age)
df.isnull().any(axis = 1).sum()

from sklearn.preprocessing import LabelEncoder
df['campaign_platform'] = LabelEncoder().fit_transform(df['campaign_platform'])
df['communication_medium'] = LabelEncoder().fit_transform(df['communication_medium'])
df['subchannel'] = LabelEncoder().fit_transform(df['subchannel'])
df['device'] = LabelEncoder().fit_transform(df['device'])
df['age'] = LabelEncoder().fit_transform(df['age'])
df['audience_type'] = LabelEncoder().fit_transform(df['audience_type'])
df['creative_type'] = LabelEncoder().fit_transform(df['creative_type'])
df['creative_name'] = LabelEncoder().fit_transform(df['creative_name'])
df['campaign_type'] = LabelEncoder().fit_transform(df['campaign_type'])






df=df.dropna()
df.shape
df.info()
'''
for c in df.columns:
    print("---- %s ---" % c)
    print(df[c].value_counts())
    from sklearn.linear_model import LinearRegression

data1=df[['campaign_platform','subchannel','audience_type','creative_type','creative_name','device','age','spends','impressions','clicks','link_clicks']]
data=data1.dropna()
x=data.iloc[:,:10]
y=data.iloc[:,10]
model = LinearRegression()
df.isna().sum()

df['link_clicks']= df['link_clicks'].replace(r'^\s+$', 'NULL', regex=True)
DF_new_row=df.loc[df['link_clicks']== 'NULL']
print(df['link_clicks'].unique())
DF_new_row
'''
df.device.value_counts()
df.link_clicks.value_counts()
''' device                16288 non-null  int64  
 9   age                   16288 non-null  int64  
 10  spends                16288 non-null  float64
 11  impressions           16288 non-null  int64  
 12  clicks                16288 non-null  int64  
 13  link_clicks '''
sns.countplot(df.age)

sns.countplot(df.device)

sns.countplot(df.subchannel)

sns.distplot(df.spends)
sns.distplot(df.clicks)
sns.distplot(df.impressions)
sns.countplot(df.age)
print('Campaign platform wise impressions total')
print((df1.groupby(['campaign_platform']).impressions.sum()))
print('Campaign platform wise total of clicks ')
print((df1.groupby(['campaign_platform'])).clicks.sum())
print('Amount of money spent by each platform  ')
print((df1.groupby(['campaign_platform'])).spends.count())
print('Sum of redirects to the ad page by each platform')
print((df1.groupby(['campaign_platform'])).link_clicks.sum())

print('device wise impressions total')
print((df1.groupby(['device']).impressions.sum()))
print('device wise clicks total')
print((df1.groupby(['device']).clicks.sum()))
print('age wise impressions total')
print((df1.groupby(['age']).impressions.sum()))
print('age wise clicks total')
print((df1.groupby(['age']).clicks.sum()))
print('age wise link_clicks total')
print((df1.groupby(['age']).link_clicks.sum()))
print('Subchannel wise sum of clicks')
print((df1.groupby(['subchannel']).clicks.sum()))
print('Campaign platform wise link total')
print((df1.groupby(['subchannel']).link_clicks.sum()))
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=df.dropna()
df.info()

X = df.iloc[:,0:13]
Y = df.iloc[:,13]

# split data into train and test sets
seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = XGBClassifier()
model.fit(X_train, y_train)
