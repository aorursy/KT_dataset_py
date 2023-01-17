# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/countries of the world.csv')
df.head()
df.columns
df.info()
"""this gives us a notion of the variable types as well as the missing information. We can see that we might need to change variable types 
in order to properly analyze some of the quantitative data"""
a1 = df.columns[4:].drop('GDP ($ per capita)')
  #I made an index with all the object columns
for i in a1:
    df[i]=df[i].str.replace(".","").str.replace(",",".")
    df[i]=df[i].astype('float64')  
#I used this method to change the variable type by replacing the commas by dots in order to be able to manipulate these values as quantitative variables
df.info()
df.head(8)
plt.subplots(figsize=(12,8))
sns.heatmap(df.drop(['Country','Region'], axis=1))
#this allows me to visualize where data is missing
sns.lmplot(x='Net migration',y='GDP ($ per capita)', data=df, hue = 'Region', height=12, fit_reg=False)
#The lmplot was used for plotting a scatterplot with hue. With this analysis we can see that there is a relation between GDP per capita and Migration
plt.subplots(figsize=(12,8))
sns.swarmplot('Net migration','Region', data=df)
#Analysis about net migration by region
df.corr()
plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(), annot=True)
sns.lmplot(x='Literacy (%)',y='GDP ($ per capita)', data=df, hue = 'Region', height=12, fit_reg=False,palette = 'bright')
sns.lmplot(x='Literacy (%)',y='Birthrate', data=df, hue = 'Region', height=12, fit_reg=False, palette = 'bright')
#I wanted to plot some choropleth maps here, but the kernel can't be connected to the internet
