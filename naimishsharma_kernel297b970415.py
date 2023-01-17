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
df = pd.read_csv('/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')
df
df = df.drop(['Date','State'],axis=1)
df.isnull().sum()
import seaborn as sns
sns.scatterplot(x=df.Value,y=df.Measure)
sns.countplot(df.Measure)
df.info()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

df['Port Name']=label.fit_transform(df['Port Name'])
oh = pd.get_dummies(df['Border'])

df= df.drop('Border',axis=1)

df= df.join(oh)
oh1 = pd.get_dummies(df['Measure'])

df= df.drop('Measure',axis=1)

df= df.join(oh1)
df.info()
from sklearn.model_selection import train_test_split

train, test = train_test_split(df,test_size=0.1,random_state=1)
def data_splitting(df):

    x=df.drop(['Value'], axis=1)

    y=df['Value']

    return x,y

x_train,y_train = data_splitting(train)

x_test,y_test = data_splitting(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

log = LinearRegression()

log.fit(x_train,y_train)

log_train = log.score(x_train,y_train)               

print(log_train*100)
from sklearn.ensemble import RandomForestRegressor

regress = RandomForestRegressor()

regress.fit(x_train , y_train)

reg_train = regress.score(x_train , y_train)

print(reg_train*100)