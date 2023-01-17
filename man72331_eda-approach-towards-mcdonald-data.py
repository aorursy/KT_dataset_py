# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Data visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/nutrition-facts/menu.csv')

df.isnull().sum()
df.head(5)
plot = sns.swarmplot(x="Category", y="Vitamin A (% Daily Value)", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Vitamin A (% Daily Value)")

plt.show()
df.at[df["Vitamin A (% Daily Value)"].idxmax(),'Item']
df.at[df["Vitamin A (% Daily Value)"].idxmin(),'Item']
plot = sns.swarmplot(x="Category", y="Vitamin C (% Daily Value)", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Vitamin C (% Daily Value)")

plt.show()
df.at[df["Vitamin C (% Daily Value)"].idxmax(),'Item']
df.at[df["Vitamin C (% Daily Value)"].idxmin(),'Item']
plot = sns.swarmplot(x="Category", y="Calcium (% Daily Value)", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Calcium (% Daily Value)")

plt.show()
df.at[df["Calcium (% Daily Value)"].idxmax(),'Item']
df.at[df["Calcium (% Daily Value)"].idxmin(),'Item']
plot = sns.swarmplot(x="Category", y="Iron (% Daily Value)", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Iron (% Daily Value)")

plt.show()
df.at[df["Iron (% Daily Value)"].idxmax(),'Item']
df.at[df["Iron (% Daily Value)"].idxmin(),'Item']
plot = sns.swarmplot(x="Category", y="Protein", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Protein")

plt.show()
df.at[df["Protein"].idxmax(),'Item']
df.at[df["Protein"].idxmin(),'Item']
plot = sns.swarmplot(x="Category", y="Trans Fat", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Trans Fat")

plt.show()
df.at[df["Trans Fat"].idxmax(),'Item']
df.at[df["Trans Fat"].idxmin(),'Item']
plot = sns.swarmplot(x="Category", y="Sugars", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Sugars")

plt.show()
df.at[df["Sugars"].idxmax(),'Item']
df.at[df["Sugars"].idxmin(),'Item']
plot = sns.swarmplot(x="Category", y="Cholesterol", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Cholesterol")

plt.show()
df.at[df["Cholesterol"].idxmax(),'Item']
df.at[df["Cholesterol"].idxmin(),'Item']
plot = sns.swarmplot(x="Category", y="Calories", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Calories")

plt.show()
df.at[df["Calories"].idxmax(),'Item']
df.at[df["Calories"].idxmin(),'Item']
corr = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True)

plt.title("Mc D Correlation Heatmap")



plt.show()