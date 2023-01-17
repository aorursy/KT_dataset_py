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
train = pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip')

train.head()
test = pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/test.json.zip')

test.head()
train.shape
train.columns
train.describe()
train.dtypes
train['interest_level'].unique()
train.dtypes
train.drop(['features','photos'],1).nunique()
import matplotlib.pyplot as plt 

import seaborn as sns



plt.figure(figsize=(10,8))

sns.boxplot(train['interest_level'],train['bedrooms'])
train['bedrooms'].nunique()
train['bathrooms'].nunique()
figure, (a,b) = plt.subplots(nrows= 2)

figure.set_size_inches(10,15)

sns.countplot(train['bedrooms'],ax=a)

sns.countplot(train['bathrooms'],ax=b)
plt.figure(figsize=(10,8))

sns.countplot(train['bedrooms'],hue=train['interest_level'])
train.groupby('interest_level')['building_id'].count() 

# Since 'low' has the most values, it is natural that low has the highest shape

plt.figure(figsize=(10,8))

sns.countplot(train['bathrooms'],hue=train['interest_level'])
train['created'] = train['created'].astype('datetime64')

train['day'] = train['created'].dt.day

train['month'] = train['created'].dt.month

train['year'] = train['created'].dt.year
train['month'].unique()
train['year'].unique()
plt.figure(figsize=(20,10))

sns.boxplot(train['interest_level'],train['day'])
plt.figure(figsize=(20,10))

sns.countplot(train['day'],hue=train['interest_level'])
plt.figure(figsize=(10,8))

sns.countplot(train['month'],hue=train['interest_level'])
train.groupby('month')['interest_level'].value_counts()
train = pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip')

alldata = pd.concat([train,test])
alldata.isnull().any() 
alldata2 = alldata.drop(['features','interest_level','photos'],axis=1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for i in alldata2.columns[alldata2.dtypes == object] :

    alldata2[i] = le.fit_transform(alldata2[i])
train2 = alldata2[:len(train)]

test2 = alldata2[len(train):]
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0, n_jobs=-1)

rf.fit(train2, train['interest_level'])

result = rf.predict_proba(test2)
result
sub = pd.read_csv('/kaggle/input/two-sigma-connect-rental-listing-inquiries/sample_submission.csv.zip')

sub.head()
sub['high'] = result[:,0]

sub['medium'] = result[:,2]

sub['low'] = result[:,1]



sub.head()
sub.to_csv('rental.csv',index=False)