#import libraries

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/world-population-trend/population_csv.csv')
data.head()
data.info()
data.describe()
data.nunique()
data['Country Name'].unique()
data[data['Country Name']=='India']
x = data[data['Country Name']=='India']['Year']

y = data[data['Country Name']=='India']['Value']
sns.jointplot(x=x, y=y, kind="hex", color="g");
data_18 = data[data['Year']== 2018]

data_18
#get the top 10 values and create a pie chart for them



data_val_10 = data_18.nlargest(10, ['Value']).reset_index()

data_val_10
data_val_50 = data_18.nlargest(50, ['Value']).reset_index()

data_val_50['Value'].plot(marker='.', alpha=0.5, figsize=(11, 9), subplots=True)
fig = px.pie(data_val_10[1:], values='Value', names='Country Name', title='World Population Pie Chart')

fig.show()
f,ax = plt.subplots(1,1, figsize=(15,7))



sns.barplot(x='Value', y='Country Name', data=data_val_10, ax=ax)

ax.set_title('Population by Countries')

ax.set_xlabel('Population')

ax.tick_params(labelsize=15)

plt.show()
with plt.style.context('dark_background'):

    plt.figure(figsize=(10, 8))



    plt.bar(range(10), data_val_10['Value'], alpha=0.8, align='center',

            label='World Population Chart by top 10 countries')

    plt.ylabel('Population')

    plt.xlabel('Countries')

    plt.legend(loc='best')

    plt.tight_layout()
#Kernel density estimation



sns.jointplot(x= x, y= y, kind="kde");
# Two-dimensional kernel density plot 



f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(x, y, ax=ax)

sns.rugplot(x, color="g", ax=ax)

sns.rugplot(y, vertical=True, ax=ax);
# Bivariate density



f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(x, y, cmap=cmap, n_levels=60, shade=True);