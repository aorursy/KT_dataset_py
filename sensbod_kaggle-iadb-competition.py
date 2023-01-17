# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.info()
test.head()
test.info()
train.select_dtypes(include='object').head()
train['Id'].size
train['idhogar'].unique().size
households = train.groupby('idhogar').apply(lambda x: len(x))
print(households.describe())
plt.hist(households, bins=range(1, 13), align='left')
plt.xlabel("Number of household's members")
plt.ylabel('Number of households')
plt.grid(True)
plt.xlim([1, 13])
plt.xticks(range(1, 14))
plt.show()
print(train['dependency'].unique())
print(train['edjefe'].unique())
print(train['edjefa'].unique())
def change_and_convert_object(df):
    di = {"yes": 1, "no": 0}
    df['dependency'].replace(di, inplace=True)
    df['edjefe'].replace(di, inplace=True)
    df['edjefa'].replace(di, inplace=True)
    
    df['dependency'] = df['dependency'].astype(float)
    df['edjefe'] = df['edjefe'].astype(float)
    df['edjefa'] = df['edjefa'].astype(float)
change_and_convert_object(train)
change_and_convert_object(test)
print(train['dependency'].unique())
print(train['edjefe'].unique())
print(train['edjefa'].unique())
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.boxplot(x='dependency', data=train, color='g', ax=ax1);
sns.boxplot(x='edjefe', data=train, color='r',ax=ax2);
sns.boxplot(x='edjefa', data=train, ax=ax3);

plt.tight_layout()
print(test['dependency'].unique())
print(test['edjefe'].unique())
print(test['edjefa'].unique())
train.describe()
numerical = train.select_dtypes(exclude='object').columns
train[numerical].isnull().sum().sort_values(ascending = False).head(10)
train[['SQBmeaned', 'meaneduc']].describe()
def fill_meaneduc_and_SQBmeaned(df):
    df['meaneduc'].fillna(df['meaneduc'].mean(), inplace = True)
    df['SQBmeaned'].fillna(df['SQBmeaned'].median(), inplace = True)
fill_meaneduc_and_SQBmeaned(train)
fill_meaneduc_and_SQBmeaned(test)
def fill_with_zero(df):
    df['rez_esc'].fillna(0, inplace = True)
    df['v18q1'].fillna(0, inplace = True)
    df['v2a1'].fillna(0, inplace = True)
fill_with_zero(train)
fill_with_zero(test)
train[numerical].isnull().sum().sum()
numerical = test.select_dtypes(exclude='object').columns
test[numerical].isnull().sum().sum()
def add_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['rent_to_bedrooms'] = df['v2a1']/df['bedrooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['tamhog_to_bedrooms'] = df['tamhog']/df['bedrooms']
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms']
    df['r4t3_to_bedrooms'] = df['r4t3']/df['bedrooms']
    df['rent_to_r4t3'] = df['v2a1']/df['r4t3']
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1'])
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms']
    df['hhsize_to_bedrooms'] = df['hhsize']/df['bedrooms']
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize']
    df['qmobilephone_to_r4t3'] = df['qmobilephone']/df['r4t3']
#     df['qmobilephone_to_v18q1'] = df['qmobilephone']/df['v18q1']
add_features(train)
add_features(test)

Id = test[['Id']]
y = train['Target']
train.drop(['Target', 'Id', 'idhogar'], axis=1, inplace=True)
X = train[train.columns]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(random_state=17, n_jobs=-1).fit(X, y)
accuracy_score(y, rf.predict(X))
test.drop(['Id', 'idhogar'], axis=1, inplace=True)
X_test = test[test.columns]
rf_pred = rf.predict(X_test)
# rf_pred
d = {'Id': Id['Id'], 'Target': rf_pred}
submission_df = pd.DataFrame(data=d)
# submission_df
submission_df.to_csv('submission.csv', sep=',', index=False)
