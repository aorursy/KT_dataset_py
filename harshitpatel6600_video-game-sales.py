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
import matplotlib.pyplot as plt 

import seaborn as sns
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head()
df.info()
df.isna().sum()
df.drop(['Year'], axis=1, inplace=True)
df['Publisher'].replace(np.nan, df['Publisher'].mode()[0], inplace=True)
df.Publisher.isna().sum()
cate_feat = [col for col in df.columns if df[col].dtypes == 'O']
cate_feat
cate_unique = list(map(lambda x: df[x].nunique(), cate_feat))
l = list(zip(cate_feat, cate_unique))
l
df.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
df1 = df.copy()
encoder = LabelEncoder()
df1.head()
df1['Platform'] = encoder.fit_transform(df1[cate_feat[1]]) 

df1['Genre'] = encoder.fit_transform(df1[cate_feat[2]])
df1['Publisher'] = df1['Publisher'].replace('<','', inplace=True)
df1['Publisher'] = encoder.fit_transform(df1[cate_feat[3]])
X = df1.drop(['Global_Sales', 'Name'], axis=1)

y = df1['Global_Sales']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)



tree = DecisionTreeRegressor(random_state=1)

tree.fit(X_train, y_train)
pred = tree.predict(X_test)

mae = mean_absolute_error(y_test, pred)

print("\033[32mMean Absolute Error: {}\033[00m" .format(mae)) 