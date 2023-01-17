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
import pandas as pd
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
train.head()
test.head()
train.shape
test.shape
train.info()
train.isnull().sum()
test.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state



train['Province_State'].fillna(EMPTY_VAL, inplace=True)

train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

test['Province_State'].fillna(EMPTY_VAL, inplace=True)

test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

test.head()
train.corr()
sns.heatmap(train.corr(), annot=True, linewidths=2, cmap="YlGnBu")
plt.figure(figsize =(20,15))

sns.set_style('whitegrid')

plot = sns.countplot(train['Country_Region'])

plot.set_xticklabels(plot.get_xticklabels(),rotation=90)
train.groupby('Country_Region')['ForecastId'].mean().plot(kind = 'bar', figsize= (40,20), title= "Countries with COVID-19 MAX", color='red')
unique = pd.DataFrame(train.groupby(['Country_Region', 'Province_State'],as_index=False).count())

unique.head()
X = train.iloc[:,1:2]

Y = train.iloc[:,2]
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

sub.to_csv('submission.csv',index=False)