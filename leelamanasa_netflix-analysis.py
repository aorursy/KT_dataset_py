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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
netflix_titles = pd.read_csv("../input/netflix_titles.csv")
print(netflix_titles.columns)
print(netflix_titles.describe())
netflix_titles.sample(5)
print(netflix_titles.isnull().sum())
print(netflix_titles.count())
netflix_new = netflix_titles.dropna()

netflix_new.count()
netflix_new.sample(30)
sns.countplot(x="type", data=netflix_new)
netflix_new.groupby(['type']).count()
CountryPlot=sns.countplot(x="country", data=netflix_new, order=netflix_new.country.value_counts().iloc[:10].index)

for item in CountryPlot.get_xticklabels():

    item.set_rotation(90)
DirectorPlot=sns.countplot(x="director", data=netflix_new, order=netflix_new.director.value_counts().iloc[:10].index)

for item in DirectorPlot.get_xticklabels():

    item.set_rotation(90)
ReleaseYearPlot=sns.countplot(x="release_year", data=netflix_new, order =netflix_new.release_year.value_counts().iloc[:10].index)

for item in ReleaseYearPlot.get_xticklabels():

    item.set_rotation(90)
YearPlot=sns.countplot(x="duration", data=netflix_new, order=netflix_new.duration.value_counts().iloc[:10].index)

for item in YearPlot.get_xticklabels():

    item.set_rotation(90)
India_data = netflix_new[(netflix_new.country == "India")]

IndiaPlot=sns.countplot(x="release_year", data=India_data, order = netflix_new.release_year.value_counts().iloc[:10].index)

for item in IndiaPlot.get_xticklabels():

    item.set_rotation(45)

df_Plot_byyear = pd.crosstab(index=netflix_new["release_year"], 

                          columns=netflix_new["type"])

df_Plot_byyear['Total'] = df_Plot_byyear['Movie'] + df_Plot_byyear['TV Show']

df_Plot_byyear_Top_10 = df_Plot_byyear.sort_values('Total',ascending = False).head(10)

df_Plot_byyear_Top_10.drop(['Total'], axis=1, inplace=True)
df_Plot_byyear_Top_10.plot(kind="bar", 

                 figsize=(10,10),

                 stacked=True)