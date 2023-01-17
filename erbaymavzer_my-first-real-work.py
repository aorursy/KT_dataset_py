# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

data.info(10)
# dataframe oluşturduk

df = pd.DataFrame(data)

df
# verimizin kolon sayısını verir

len(data.columns)
# oluşturduğumuz df nin kolonlarını verir

df.columns
# 10. indexin bilgilerini getirir

df.loc[10]
# dataframe imizde views i 80 k üzeri olanları getir

df.head(10).views>150000
# ilk 10 verinin line türünde tablosu

data.views.head(10).plot(figsize=(13,13),kind='line',label='Views',linestyle=':')

data.likes.head(10).plot(figsize=(13,13),kind='line',label='Likes',linestyle='-.')



plt.legend(loc='upper right')

plt.show()
# ilişki tablosu

data.corr()
f,ax = plt.subplots(figsize=(13,13))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# ilk 25 verinin izlenme - yorum sayısı tablosu line olarak

data.head(25).comment_count.plot(kind="line",color="b",label="Comments Count",ls="--",lw=1,alpha=.6,grid=True)

plt.legend(loc="upper right")

plt.xlabel("Views")

plt.ylabel("Comments Count")

plt.title("Views-Comment Count Plot")

plt.show()
# ilk 25 verinin izlenme - yorum sayısı tablosu scatter olarak

data.head(25).plot(kind="scatter",color="b",x='views',y='comment_count',label="Comments Count",figsize=(13,13),lw=1,alpha=.6,grid=True)

plt.legend(loc="upper right")

plt.xlabel("Views")

plt.ylabel("Comments Count")

plt.title("Views-Comment Count Plot")

plt.show()
data.plot(grid=True,alpha=.9,subplots=True,figsize=(15,15))

plt.show()