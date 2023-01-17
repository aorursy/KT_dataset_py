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
#加载模块

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold,train_test_split,GridSearchCV

#训练模型

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

#模型整合

from sklearn.pipeline import Pipeline

#模型指标查看

from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score

#屏蔽warnings

import warnings

warnings.filterwarnings('ignore')

df=pd.read_csv(r'/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')

df.describe().round(2)
df.shape
df.isnull().sum()
df['Average User Rating'].hist()
df['Average User Rating'].dropna(inplace=True)

sns.set()

sns.countplot(x="Average User Rating", data=df)
plt.figure(figsize=(20,14))

sns.heatmap(df.corr(),linewidths=0.1,vmax=1.0,

           square=True,linecolor='white',annot=True)

plt.show()
a=df.groupby('Age Rating')['ID'].count()

a.sort_values(ascending=False,inplace=True)
plt.figure(figsize=(12,5))

sns.countplot(x="Age Rating", data=df)
b=a/len(df)

print(f'{round(b[0]*100,2)}% strategy games are in the 4+ age range.')

print(f'{round(b[1]*100,2)}% strategy games are in the 9+ age range.')

print(f'{round(b[2]*100,2)}% strategy games are in the 12+ age range.')

print(f'{round(b[3]*100,2)}% strategy games are in the 17+ age range.')
# since there is a comma delimited list in languages, lets separate each lang into its own column

languages = pd.DataFrame(df['Languages'].str.split(', ',expand=True))

# now lets merge all the columns into one master languages column

languages = pd.DataFrame(languages.values.ravel(), columns = ["Languages"])

# get a total of all the languages and their counts into a df

languages = pd.DataFrame(languages['Languages'].value_counts().reset_index())

# rename columns

languages.columns = ['Language', 'Count']



# grab top 10 (out of 115) most used languages for display

sns.barplot(x="Language", y="Count", data=languages.head(10));
# count the number of commas in each ['Languages'] then add 1 (to count the original Language)

# this will show you how many languages some apps are

multi = df['Languages'].str.count(r', ') + 1

multi_lingual = pd.DataFrame(multi.value_counts().reset_index())



multi_lingual.columns = ['Languages', 'Count']



sns.barplot(x="Languages", y="Count", data=multi_lingual.head(10));