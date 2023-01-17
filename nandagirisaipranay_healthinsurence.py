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
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
train.head()
len(train)
len(test)
test.head()
test.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
train_sample = train
train_sample.dtypes

train_sample['Response'] = train_sample['Response'].astype('category')
sns.pairplot(train_sample[['Region_Code','Age','Response']], hue='Response',  size=2.5)
sns.pairplot(train)
train_sample.drop(['id'], inplace=True, axis =1)
train_sample = pd.get_dummies(train_sample, drop_first=True)
train_sample.head()
train_sample.dtypes
train_sample['Gender_Male'] = train_sample['Gender_Male'].astype('int64')
train_sample['Vehicle_Age_< 1 Year'] = train_sample['Vehicle_Age_< 1 Year'].astype('int64')
train_sample['Vehicle_Age_> 2 Years'] = train_sample['Vehicle_Age_> 2 Years'].astype('int64')
train_sample['Vehicle_Damage_Yes'] = train_sample['Vehicle_Damage_Yes'].astype('int64')
train_sample.dtypes

sns.pairplot(train_sample)
from sklearn.linear_model import LogisticRegression 

lr = LogisticRegression()
X = train_sample.iloc[:,:-1]
Y = train_sample.iloc[:,-1]
lr.fit(X,Y)
test_sample = pd.get_dummies(test, drop_first =True)
test_sample.dtypes
test_sample['Gender_Male'] = test_sample['Gender_Male'].astype('int64')
test_sample['Vehicle_Age_< 1 Year'] = test_sample['Vehicle_Age_< 1 Year'].astype('int64')
test_sample['Vehicle_Age_> 2 Years'] = test_sample['Vehicle_Age_> 2 Years'].astype('int64')
test_sample['Vehicle_Damage_Yes'] = test_sample['Vehicle_Damage_Yes'].astype('int64')
test_sample.drop(['id'], inplace = True, axis = True)
y_pred = lr.predict(test_sample)
y_pred

