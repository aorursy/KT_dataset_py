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

print(f'The shape of the training set is: {train.shape}')
print(f'The shape of the testing set is: {test.shape}')
train.isnull().sum()
test.isnull().sum()
train = train.drop('id', 1)
test = test.drop('id', 1)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns

plt.figure(figsize=(25,30))
plt.subplot(331)
sns.countplot(x='Response', data=train, hue='Vehicle_Age')


plt.subplot(332)
sns.countplot(x='Response', data=train, hue='Vehicle_Damage')

plt.subplot(333)
sns.countplot(x='Response', data=train, hue='Previously_Insured')
from sklearn.utils import resample

minority = train[train.Response==1]
majority = train[train.Response==0]

downsample = resample(majority, replace=False, n_samples=46710)

data = pd.concat([downsample, minority])
data.head()
plt.figure(figsize=(25,30))
plt.subplot(331)
sns.countplot(train.Response).set_title('Before Downsampling')

plt.subplot(333)
sns.countplot(data.Response).set_title('After Downsampling')
x = data.drop('Response', 1)
y = data.Response
from sklearn.preprocessing import LabelEncoder

x.Gender = pd.get_dummies(x.Gender)
x.Vehicle_Damage = pd.get_dummies(x.Vehicle_Damage)

encoder = LabelEncoder()
x.Vehicle_Age = encoder.fit_transform(x.Vehicle_Age) 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x = scaler.fit_transform(x)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

random_forest = RandomForestClassifier()
knn = KNeighborsClassifier()
xgb = XGBClassifier()

estimators = [random_forest, knn, xgb]

for i in estimators:
    score = cross_val_score(i, x, y, cv=3, n_jobs=5)
    mean = score.mean()
    print(f'{i} score: {mean}')
    