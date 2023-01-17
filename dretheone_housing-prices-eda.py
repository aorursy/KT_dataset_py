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
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

%matplotlib inline

pd.set_option('display.max_columns', 500)
df = pd.read_csv('../input/house-prices-dataset/train.csv')

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
plt.subplots(figsize=(20, 5))
sns.distplot(df_train['SalePrice'], rug=True)
plt.subplots(figsize=(15, 20))
sns.barplot(x=df_train.count()[:], y=df_train.count().index)
obj_features = df_train.select_dtypes('object').columns.values

num_features = df_train.select_dtypes(['int', 'float64']).columns.values
df[num_features].hist(figsize=(40, 40), bins=20)
categorical = [
    'BedroomAbvGr',
    'BsmtFullBath',
    'BsmtHalfBath',
    'Fireplaces',
    'FullBath',
    'KitchenAbvGr',
    'OverallCond',
    'OverallQual',
    'TotRmsAbvGrd'
]
categorical_features = categorical + list(obj_features)

numerical_features = list(set(num_features) - set(categorical_features))

numerical_features.remove('Id')
nc = 4
nr = len(categorical_features)//nc
f, ax = plt.subplots(nrows=nr,ncols=nc,squeeze=False,figsize=(21, 4*nr))
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

for col, ax in zip(categorical_features, ax.flatten()[:len(categorical_features)]):
    b = sns.boxplot(data=df_train,x=col,y='SalePrice', ax=ax)
    b.twinx()
    ct = sns.countplot(x=col, data=df_train, color='red', alpha=0.3)
    ct.set(xlabel=None)
    
plt.show()
plot_data = df_train[numerical_features].sample(n=200, replace=False, random_state=1)

g = sns.PairGrid(plot_data)
g.map_upper(sns.scatterplot, hue="SalePrice", data=plot_data,
            palette=sns.cubehelix_palette(plot_data['SalePrice'].nunique(), start=.5, rot=-.75))
g.map_diag(sns.kdeplot, bw=10)
g.map_lower(sns.kdeplot, bw=10, cmap="Blues", shade=True, shade_lowest=False)
nc = 4
nr = len(numerical_features)//nc
f, ax = plt.subplots(nrows=nr,ncols=nc,squeeze=False,figsize=(21, 4*nr))
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

for col, ax in zip(numerical_features, ax.flatten()[:len(numerical_features)]):
    _, _, r_value, _, _ = stats.linregress(df_train[col], df_train['SalePrice'])
    r_squared = r_value**2
    ax.set(title= f"r2:{r_squared:.2f}")
    sns.regplot(data=df_train[numerical_features],y='SalePrice',x=col, ax=ax)   
    
plt.show()
correlation = df_train[numerical_features].corr().abs()
sns.clustermap(correlation, cmap='coolwarm', 
               vmin=0, vmax=0.8, center=0, 
               square=True, linewidths=.5, 
               figsize=(50,50), yticklabels=1)