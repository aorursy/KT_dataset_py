# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/world-happiness/2019.csv')

df.head()
#info

df.info()
# shape of datset

df.shape
# statistical description of datset

df.describe().T
# to find any missing values

df.isna().sum()
# no missing values
df.head(3)
# correlation matrix

df.corr()
# top 10 countries with Scores

scores=df.sort_values('Score',ascending=False).set_index('Country or region')[:10]



sns.barplot(x=scores['Score'], y=scores.index, data=df)
# distribution plots

dist_plots=['Score','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']

fig, axes=plt.subplots(4,2, figsize=(10,20))

for i,j in enumerate(dist_plots):

    ax=axes[int(i/2), i%2]

    sns.distplot(df[j], ax=ax)

fig.delaxes(axes[3,1])
# check whether it has any outliers

dist_plots=['Score','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']

fig, axes=plt.subplots(4,2, figsize=(10,20))

for i,j in enumerate(dist_plots):

    ax=axes[int(i/2), i%2]

    sns.boxplot(df[j], ax=ax)

fig.delaxes(axes[3,1])
# there are some outliers we need to see weather it affect the model or not
df[df['Country or region']=='India']
#India is at 140th rank from 156 countries
# top 10 countries with GDP

gdp=df.sort_values('GDP per capita',ascending=False).set_index('Country or region')[:10]



sns.barplot(x=gdp['GDP per capita'], y=gdp.index, data=df)
#considering the gdp per capita quatar has highest gdp per capita
#corruptions top countries

corruptions=df.sort_values('Perceptions of corruption',ascending=False).set_index('Country or region')[:10]



sns.barplot(x=corruptions['Perceptions of corruption'], y=corruptions.index, data=df)
# singapore has highest corruption 
#Generosity top countries

generous=df.sort_values('Generosity',ascending=False).set_index('Country or region')[:10]



sns.barplot(x=generous['Generosity'], y=generous.index, data=df)
#myanmar has with highest generosity with value greater than 0.5
df.head(1)
#Freedom to make life choices top countries

freedom=df.sort_values('Freedom to make life choices',ascending=False).set_index('Country or region')[:10]



sns.barplot(x=freedom['Freedom to make life choices'], y=freedom.index, data=df)
#Healthy life expectancy top countries

life=df.sort_values('Healthy life expectancy',ascending=False).set_index('Country or region')[:10]



sns.barplot(x=life['Healthy life expectancy'], y=life.index, data=df)
df.sort_values('Healthy life expectancy',ascending=False).set_index('Country or region')[:10]
# pairplots

sns.pairplot(df)