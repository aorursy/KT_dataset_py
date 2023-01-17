# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
BadDataX = pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv", 

                       encoding="latin", thousands = '.')

BadDataX.state.value_counts()
X_duplicated = BadDataX[BadDataX.duplicated(BadDataX.columns.drop('number'))]

#print(X_duplicated)

X_duplicated.drop(X_duplicated[X_duplicated.state=='Alagoas'].index, inplace=True)



# note: firstly replace all "Rio" by "Rio Grande do Norte".

# Then replace duplicated "Rio Grande do Norte" by "Rio Grande do Sul"

newStates = {"Mato Grosso":"Mato Grosso do Sul", "Paraiba":"Parana", "Rio":"Rio Grande do Norte"}

X_duplicated.replace({"state": newStates})



X_duplicated.loc[X_duplicated.duplicated(X_duplicated.columns.drop('number')), 

                 'state'] = "Rio Grande do Sul"

BadDataX.drop_duplicates(subset=BadDataX.columns.drop('number'), inplace=True)

X = pd.concat([BadDataX, X_duplicated])

X
TopDataXStates = X[["state", "number"]].groupby("state").sum().nlargest(5, "number").index

TopDataX = X[X.state.isin(TopDataXStates)].groupby(["state", "year"]).sum()

TopDataX.reset_index(inplace=True)

plt.figure(figsize=(14,6))

sns.lineplot(x=TopDataX["year"], y=TopDataX["number"], hue=TopDataX["state"])
plt.figure(figsize=(15,6))

stateBarplot = sns.barplot(x="state", y="number", 

                           data=X[['state', 'number']].groupby(["state"]).sum()["number"].reset_index())

stateBarplot.set_xticklabels(stateBarplot.get_xticklabels(), horizontalalignment='right', rotation=45)



plt.figure(figsize=(13,6))

sns.barplot(x="month", y="number", 

            data=X[['month', 'number']].groupby(["month"]).sum()["number"].reset_index())
groupedX = X[["month", "state", "number"]].groupby(["month", "state"]).sum()

heatMapDataX = groupedX.reset_index().pivot(index='month', columns='state', values='number')



plt.figure(figsize=(16, 9))

sns.heatmap(data=heatMapDataX)