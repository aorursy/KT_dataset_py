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
magazine=pd.read_csv("/kaggle/input/magazine-covers/archive.csv")

magazine.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(18,10))

sns.countplot(data=magazine,x="Country")
mag2000s=magazine[magazine["Year"]>=2000]

sns.countplot(data=mag2000s,x="Country")
mag2000s[mag2000s["Country"]=="Vatican City"]
mag2000s[mag2000s["Country"]!="United States"]
ages=magazine["Year"]-magazine["Birth Year"]

magazine["Age"]=ages
minindex=magazine["Age"].argmin()

print(magazine.iloc[minindex,:])
minindex=magazine["Age"].argmax()

print(magazine.iloc[minindex,:])
import statistics

statistics.median(ages.dropna())
magazine[magazine["Country"]=="Turkey"]
magazine[magazine["Country"]=="Greece"]
magazine[magazine["Country"]=="China"]
magazine[magazine["Country"]=="Israel"]
set(magazine["Category"])
plt.figure(figsize=(18,10))

sns.countplot(data=magazine,x="Category")
mag2000s=magazine[magazine["Year"]>=2000]

plt.figure(figsize=(18,10))

sns.countplot(data=mag2000s,x="Category")
grouping=magazine.groupby(["Country","Category"]).count()["Year"]

print(grouping)
magazine[magazine["Country"]=="Iran"]
grouping=magazine.groupby(["Country"]).median()["Age"]

grouping
magazine[(magazine["Country"]=="Poland")|(magazine["Country"]=="Ethiopia")]
magazine[magazine["Category"]=="Science"]