# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_columns',400)
train_df = pd.read_csv("../input/train.csv")

macro_df = pd.read_csv("../input/macro.csv")
n_train_df = train_df.copy()

n_train_df['timestamp'] = n_train_df['timestamp'].apply(lambda x:x[0:4]+x[5:7])
np.mean(n_train_df['life_sq'])
sns.countplot(y = 'product_type',data=n_train_df)
n_train_df.head()
mv = n_train_df.isnull().sum()
x= list(mv)

y= list(n_train_df.columns)
f,ax = plt.subplots(figsize=(10,40))

sns.barplot(ax=ax,x=x,y=y)
for col in n_train_df.columns:

    if n_train_df[col].isnull().any():

        del n_train_df[col]
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
for col in n_train_df.columns:

    if n_train_df[col].dtypes == 'object':

        n_train_df[col] = enc.fit_transform(n_train_df[col])
x_train_df = n_train_df[list(range(0,240))]
y_train_df = n_train_df[[240]]
x_train_df.head()
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

x_train = sc.fit_transform(x_train_df)

y_train = sc.fit_transform(y_train_df)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train,y_train)