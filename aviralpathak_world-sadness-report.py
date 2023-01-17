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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings 

warnings.filterwarnings('ignore')



from pandas.plotting import parallel_coordinates
data_2015 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')
data_2015.columns = data_2015.columns.str.replace(' ','_')
data_2015.head()
data_2015.isnull().sum()
sns.pairplot(data_2015)
region_gdp = data_2015['Economy_(GDP_per_Capita)'].groupby(data_2015['Region'])
region_gdp.mean().sort_values()
sns.barplot(x='Economy_(GDP_per_Capita)', y='Region', data = data_2015,ci=None)
region_gdp_scatter = sns.relplot(x='Economy_(GDP_per_Capita)',y='Happiness_Score',hue='Region',data=data_2015,size="Happiness_Score",

            sizes=(100, 800), alpha=.5, palette="muted",

            height=6)
data_2015[['Country', 'Economy_(GDP_per_Capita)']].sort_values(by = 'Economy_(GDP_per_Capita)',

            ascending = True).head(10)
region_family = data_2015['Family'].groupby(data_2015['Region'])
region_family.mean().sort_values()
sns.barplot(x='Family', y='Region', data = data_2015,ci=None)
region_family_scatter = sns.relplot(x='Family',y='Happiness_Score',hue='Region',data=data_2015,size="Happiness_Score",

            sizes=(100, 800), alpha=.5, palette="muted",

            height=6)
data_2015[['Country', 'Family']].sort_values(by = 'Family',

            ascending = True).head(10)
region_health = data_2015['Health_(Life_Expectancy)'].groupby(data_2015['Region'])
region_health.mean().sort_values()
sns.barplot(x='Health_(Life_Expectancy)', y='Region', data = data_2015,ci=None)
region_health_scatter = sns.relplot(x='Health_(Life_Expectancy)',y='Happiness_Score',hue='Region',data=data_2015,size="Happiness_Score",

            sizes=(100, 800), alpha=.5, palette="muted",

            height=6)
data_2015[['Country', 'Health_(Life_Expectancy)']].sort_values(by = 'Health_(Life_Expectancy)',

            ascending = True).head(10)