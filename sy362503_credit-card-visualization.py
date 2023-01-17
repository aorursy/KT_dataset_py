# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df
df.info()
df.isnull().sum().sum()
df.describe()
def draw_histograms(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor="b")

        ax.set_title(feature+" Distribution",color='DarkRed')

        ax.set_yscale('log')

    fig.tight_layout()  

    plt.show()

draw_histograms(df,df.columns,7,5)
df.Class.value_counts()
ax = sns.countplot(x='Class',data=df, palette ="muted")

ax.set_yscale("log")

plt.yticks(rotation=30)

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=False)

plt.show();
sns.relplot(x="Time", y="Amount", data=df, kind="scatter", aspect=4, row="Class")

plt.show()
f, ax = plt.subplots(figsize=(35,25))

sns.heatmap(df.corr(), annot=True, fmt= '.1f', linewidths=.1, ax=ax, cmap="YlGnBu")

plt.show()
lm_plot = sns.lmplot(x='V20', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2}, aspect=2)

lm_plot = sns.lmplot(x='V7', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2}, aspect=2)

plt.show()
lm_plot = sns.lmplot(x='V2', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2}, aspect=2)

lm_plot = sns.lmplot(x='V5', y='Amount',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2}, aspect=2)

plt.show()
lm_plot = sns.lmplot(x='Time', y='V3',data=df, hue='Class', fit_reg=True,scatter_kws={'s':2}, aspect=3)

plt.show()