# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

sns.set(color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        df = pd.read_csv(os.path.join(dirname, filename),encoding = "ISO-8859-1")

        

#Checking the nature of data quickly.

print(df.describe(include='all').T)

print(df.columns)

df = df.dropna()

df.drop(['Unnamed: 0'],inplace=True,axis=1)



# Any results you write to the current directory are saved as output.
plt.figure(figsize=(15,8))

chart = sns.countplot(

    data = df,

    x='Artist.Name',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=65,horizontalalignment='right')



plt.figure(figsize=(10,8))

chart = sns.countplot(

    data = df,

    x='Genre',

    palette='Set1'

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=65,horizontalalignment='right')
by_genre = (df

            .groupby(['Genre', 'Popularity'])

            .size()

            .unstack()

           )

by_genre

plt.figure(figsize=(10,10))

g = sns.heatmap(

    by_genre, 

    square=True, # make cells square

    cmap='OrRd', # use orange/red colour map

    linewidth=1 # space between cells

)
plt.figure(figsize=(10,10))

plt.title('Correlation heatmap')

sns.heatmap(df.corr(),annot=True,vmin=-1,vmax=1,cmap="OrRd",center=1)
df.dropna()

g = sns.pairplot(df, kind="reg")
#Regression all variables

X = df._get_numeric_data().drop("Popularity",axis=1)

Y = df[["Popularity"]]

X = sm.add_constant(X) # adding a constant



model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 



print_model = model.summary()

print(print_model)
X = df[["Beats.Per.Minute","Energy","Danceability","Liveness","Valence.","Length."]] #Sampled based on multicollinearity 

Y = df[["Popularity"]]

X = sm.add_constant(X) # adding a constant



model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 



print_model = model.summary()

print(print_model)