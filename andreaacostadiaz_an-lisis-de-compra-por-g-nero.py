import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt # visualizing data

import seaborn as sns 

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import seaborn as sns

df = pd.read_csv('../input/BlackFriday.csv')

df.shape
df.info()
df.describe()

df.describe()

df.head()
def plot(group,column,plot):

    ax=plt.figure()

    df.groupby(group)[column].sum().sort_values().plot(plot)

    

plot('Gender','Purchase','bar')

explode = (0.1,0)  

fig1, ax1 = plt.subplots(figsize=(8,4))

ax1.pie(df['Gender'].value_counts(), explode=explode,labels=['Hombre','Mujer'], autopct='%1.1f%%',

        shadow=True, startangle=100)

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
spent_byage = df.groupby(by='Age').sum()['Purchase']

plt.figure(figsize=(8,5))

sns.barplot(x=spent_byage.index,y=spent_byage.values, palette="Reds_d")

plt.title('Media de compras por edad')

plt.show()