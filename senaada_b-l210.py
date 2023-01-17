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
df = pd.read_csv("../input/irisdataset/iris.csv")
df.head()
df.shape
df.info()
df.describe()
df.isna().sum()
df.corr()
sns.heatmap(df.corr())
df['variety'].unique()
df['variety'].nunique()
sns.scatterplot(x="sepal.width" , y="sepal.length" , data=df);
sns.jointplot(x="sepal.width" , y="sepal.length" , data=df)
sns.scatterplot(df['sepal.width'], df['sepal.length'], hue = df['variety'])
df['variety'].value_counts()
sns.violinplot(x="sepal.width" , data=df)
sns.distplot(df["sepal.width"])
sns.violinplot(y = 'sepal.length', x = "variety", data = df)
sns.countplot(df['variety'])
sns.jointplot(x="sepal.length" , y="sepal.width" , data=df)
sns.jointplot(df['sepal.length'],df['sepal.width'], kind = 'kde')
sns.scatterplot(df['sepal.length'],df['sepal.width'])
sns.scatterplot(df['sepal.length'], df['sepal.width'], hue = df['variety'])
sns.lmplot(x = "petal.length", y = "petal.width" , data = df)
df.corr()["petal.length"]["petal.width"]
df["total.length"] = df["petal.length"] + df["sepal.length"]
total_length = df["total.length"]
total_length.mean()
total_length.std()
total_length.max()
df[(df["variety"] == "Setosa") & (df["sepal.length"] > 5.5)]
df.groupby(["variety"]).mean()