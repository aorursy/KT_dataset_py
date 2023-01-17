# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.columns
df.info()
df.head(10)
df.describe()
df.info()
def bar_plot(variable):

    varCount = df[variable].value_counts()

    

    plt.figure(figsize = (10,10))

    plt.bar(varCount.index, varCount)

    plt.xticks(varCount.index, varCount.index.values,rotation=90, horizontalalignment='right',fontweight='light',fontsize='large')

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()
category1 = [ "Platform", "Genre","Year"] # I didn't do plt bar other categorical variables like name and publisher for it not show correct visual

for c in category1:

    bar_plot(c)
category2 = ["Name","Publisher"]

for c in category2:

    print(df[c].value_counts())
def plot_hist(variable):

    plt.figure(figsize = (5,5))

    plt.hist(df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.show()
numericVar = [ "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]

for n in numericVar:

       plot_hist(n)
# Genre vs Global_Sales 

df[["Genre","Global_Sales"]].groupby(["Genre"], as_index = False).mean().sort_values(by="Global_Sales",ascending = False)
# Platform vs Global_Sales

df[["Platform","Global_Sales"]].groupby(["Platform"], as_index = False).mean().sort_values(by="Global_Sales",ascending = False)
# Year vs Global_Sales

df[["Year","Global_Sales"]].groupby(["Year"], as_index = False).mean().sort_values(by="Global_Sales",ascending = False)
# Publisher vs Global_Sales

p = df[["Publisher","Global_Sales"]].groupby(["Publisher"], as_index = False).mean().sort_values(by="Global_Sales",ascending = False)

p.head(15)   

    
df["Publisher"].unique()
# outlier detection(IQR-Quantile) get reference from publisher 



for column in df.columns[6:]: # selected only sales columns

    for p in df["Publisher"].unique():

        selected_p = df[df["Publisher"] == p]

        selected_column = selected_p[column]

        

        q1 = selected_column.quantile(0.25)

        q3 = selected_column.quantile(0.75)

        

        iqr = q3 - q1

        

        minimum = q1 - (1.5 * iqr)

        maximum = q3 + (1.5 * iqr)

        

        print(column,p,"| min=",minimum,"max=",maximum)

        

        

        max_idxs = df[(df["Publisher"] == p) & (df[column] > maximum)].index  

        print(max_idxs)

        min_idxs = df[(df["Publisher"] == p) & (df[column] < minimum)].index  

        print(min_idxs)

        

        df.drop(index= max_idxs,inplace= True)

        df.drop(index= min_idxs,inplace= True)

        
df.info()
df.isna().sum() # find missing values
# filling missing values with mode

df["Year"].fillna(df["Year"].mode()[0], inplace=True)

df["Publisher"].fillna(df["Publisher"].mode()[0], inplace=True)



df.isna().sum()
plt.figure(figsize = (8,8))

ax = sns.heatmap(df.corr(), linewidths=.5)
for column in df.columns[6:]:

    sns.relplot(x="Year", y=column, kind="line",data=df)

    plt.show()
genre_val = df.Genre.value_counts().values

labels = df.Genre.value_counts().index

plt.figure(figsize=(10,10))

plt.pie(genre_val, labels=labels)



plt.show()