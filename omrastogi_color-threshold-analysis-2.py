# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/2nd-varible-color-masking-anaysis/data.csv')

df.info
for i in df: 

    sns.scatterplot(x = df[i], y = df['n'])
g = sns.heatmap(df[["n", "lsd", "vsd", "hsd","mean"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
sns.scatterplot(x = df['lsd'], y = df['n'])
sns.scatterplot(x = df['vsd'], y = df['n'])
sns.scatterplot(x = df['hsd'], y = df['n'])
sns.scatterplot(x = df['mean'], y = df['n'])
dataset = df
df = dataset
y = df["n"]

df = df.drop(columns = ["n"])

df = df.drop(columns = ["hsd"])

X = df
df = df.drop(columns = ["lsd"])

X_ = df
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X_train, X_test, y_train, y_test = train_test_split (X,y,random_state=1)



X_train_, X_test_, y_train, y_test = train_test_split (X_,y,random_state=1)
reg = LinearRegression().fit(X,y)

score = reg.score(X_test, y_test)

print (score)

intercept = reg.intercept_

slope = reg.coef_

sd_slope = slope[0]

mean_slope = slope[1]

# print (intercept)

# print (slope)
print (X_)

reg = LinearRegression().fit(X_,y)

score = reg.score(X_test_, y_test)

print ("score =",score)

intercept = reg.intercept_

slope = reg.coef_

mean_slope = slope[0]

vsd_slope = slope[1]

print ("intercept =",intercept)

print ("slopes =",slope)