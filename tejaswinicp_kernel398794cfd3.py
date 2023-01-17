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
df = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')
df.head()
df.info()
df.describe()
df = df.drop(columns=['Id','Open Date'],axis=1)
df.City.value_counts()
len(df.City.unique())
# Group by cities to visualize cities with highers mean revenue
df_test = df[['City','revenue']]
df_grp = df_test.groupby(['City'],as_index=False).mean()
df_grp.head()
df_grp_sorted = df_grp.sort_values(by='revenue')
plt.figure(figsize=(20,8))
plt.xticks(rotation=45)
sns.barplot(df_grp_sorted['City'],df_grp_sorted['revenue'])
# Follow the above same steps to visualize the type of restaurant with highest revenue
df_test = df[['Type','revenue']]
df_grp = df_test.groupby(['Type'],as_index=False).mean()
df_grp_sorted = df_grp.sort_values(by='revenue')
sns.barplot(df_grp_sorted['Type'],df_grp_sorted['revenue'])
df['City Group'].value_counts()
df1 = df.drop(columns=['City Group','Type'],axis=1)
# Label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['City'] = le.fit_transform(df['City'])
import lightgbm as lgb
y = df.revenue
x = df.drop(columns=['revenue','City Group','Type'],axis=1)
model = lgb.LGBMRegressor()
model.fit(x,y)
model.score(x,y)
