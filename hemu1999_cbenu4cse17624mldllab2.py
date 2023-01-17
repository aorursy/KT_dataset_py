# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as skl



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart.csv")
df.head()
df.describe()
df.info()
nrows,ncols = 5,3

count = 0

fig,axes = plt.subplots(nrows,ncols)

for i in range(nrows):

    for j in range(ncols):

        if i==nrows-1 and j==ncols-1:

            break

        df.iloc[:,count].plot.kde(ax = axes[i,j],figsize=(8,10),title=df.columns[count])

        plt.tight_layout()

        count+=1

        
dependent_df = df.iloc[:,-1]

independent_df = df.iloc[:,:-1]

independent_df.head()
df.corr()
plt.figure(figsize=(18,6))

sns.heatmap(df.corr(),annot=True)

plt.tight_layout()

plt.show()
nan_percent = df.isna().mean()*100

nan_count = df.isna().sum()

nan_df = pd.concat([nan_percent.round().rename("Missing Percentage"),nan_count.rename("Missing Count")], axis = 1)

nan_df
#using dropna to remove rows

after_drop_df = df.dropna(axis=0)

after_drop_df.isna().sum()
zero_fill = df.fillna(0)

#In back_fill the last column null values filled with 0,last but one with 1, then before that with 2 ans so on...

back_fill =  df.fillna('bfill')

#In forward fill the first column with 0, second with 1 and so on.

forward_fill = df.fillna('ffill')
#we can check for unique values of columns to decide for categorical since description is not available

df.cp.unique()

cols = df.columns

for i in cols:

    print(i)

    print(df[i].unique())
#Now we traverse with column and fill with theirs means,medians and mode.

#we have to do these based on column nature

#Even though all are in float64 or int64 they doesn't mean numerical data

#Like sex column is a categorical data still. Hence forth I use mode.so for every column it depends so we should do this filling column wise

cols = df.columns

categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

drop_row = ['target']

for i in cols:

    if i in categorical_cols:

        df[i] = df[i].fillna(df[i].mode())

    elif i in drop_row:

        df.dropna(axis=0,inplace=True)

    else:

        df[i] = df[i].fillna(df[i].mean())
df.isna().sum()
#Let's know the each column's max and min to know th range

cols = df.columns

for i in cols:

    if i not in categorical_cols:

        print(f"{i} has min:{df[i].min()} and max:{df[i].max()}")
#Even Though the ranges are still okay just for assignment 

#Purpose we are going to do min_max scaling

def minMaxScaler(df_i):

    mini = df_i.min()

    maxi = df_i.max()

    df_i = (df_i-mini)/(maxi-mini)

    return df_i

for i in cols:

    if i not in categorical_cols:

        df[i] = minMaxScaler(df[i])
#Let's check the range of those functions again!!!

for i in cols:

    if i not in categorical_cols:

        print(f"{i} has min:{df[i].min()} and max:{df[i].max()}")
for i in cols:

    print(df[i].unique())
categorical_cols
dfc = df[categorical_cols]
dfc = dfc.astype('category')
dfc.info()
def myOneHotEncoder(df,i):

    newdf = pd.get_dummies(df[i],drop_first = True)

    df.drop(labels=i,inplace=True,axis=1)

    df = pd.concat([df,newdf],axis=1)

    return df
m = pd.get_dummies(dfc[['slope']])

m
dfc.drop('slope',axis=1)

dfc = pd.concat([dfc,m],axis=1)

dfc
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dependent_df, independent_df, test_size=0.33, random_state=42)
dependent_df = df.iloc[:,:-1]

independent_df = df.iloc[:,-1]
from sklearn.model_selection import KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])

y = np.array([1, 2, 3, 4])

kf = KFold(n_splits=2)

kf.get_n_splits(X)



print(kf)



for train_index, test_index in kf.split(X):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]
df.info()


df.plot.scatter(x='age',y ='target')
plt.figure(figsize=(20,12))

df.plot.hist()

plt.tight_layout()

plt.show()