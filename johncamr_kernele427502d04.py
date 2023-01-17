# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



from sklearn.preprocessing import MinMaxScaler

import seaborn           as sns   # visualizations

import matplotlib.pyplot as plt   # visualizations

from bokeh.io import output_notebook, show



import folium

import pylab as pl



import scipy.stats                # statistics

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_topsoccer0="../input/top250-00-19.csv"

df_topsoccer0 = pd.read_csv(data_topsoccer0)

df_topsoccer=pd.DataFrame(data=df_topsoccer0)

df_topsoccer.head()

print(df_topsoccer.head(3))

print(df_topsoccer.info())

print(df_topsoccer.columns)

print(df_topsoccer.dtypes)

print(df_topsoccer.tail(3))

type(df_topsoccer)
pd.options.display.float_format = '{:.2f}'.format

df_topsoccer.describe()
df_topsoccer.replace({'Age' : 0}, 24, inplace=True) # reempplazar las edades en 0 por la media 

df_topsoccer['Market_value'] = df_topsoccer['Market_value'].fillna(0) # llenamos los valores vacios a 0

df_topsoccer.replace({'Market_value' : 0}, df_topsoccer['Market_value'].mean(), inplace=True) # reempplazar losn 0 por la media 

df_topsoccer.describe()
df_topsoccer
Position_count=df_topsoccer.groupby ('Position')['Position'].count()

Age_count=df_topsoccer.groupby ('Age')['Age'].count()

TeamFrom_count=df_topsoccer.groupby ('Team_from')['Team_from'].count()

League_from_count=df_topsoccer.groupby ('League_from')['League_from'].count()

Season_count=df_topsoccer.groupby ('Season')['Season'].count()
Position_count
Age_count
TeamFrom_count
League_from_count
Season_count
# Visualization League_from vs Age

df = df_topsoccer.sort_values(['Age'],ascending=False).reset_index(drop=True)

plt.figure(figsize=(25,25))

sns.barplot(x=df["Age"],y=df["Position"])

plt.xlabel("Player Age",fontsize=15)

plt.ylabel("Position",fontsize=15)

plt.title("player Position by age",fontsize=15)

plt.show()
sns.set(style='darkgrid')

sns.countplot(x = 'Age',

              data = df_topsoccer,

              order = df_topsoccer['Age'].value_counts().index)

plt.show()
sns.set(style='darkgrid')

sns.countplot(y = 'Position',

              data = df_topsoccer,

              order = df_topsoccer['Position'].value_counts().index)

plt.show()
sns.set(style='darkgrid')

plt.figure(figsize=(15,25))

sns.countplot(y = 'League_from',

              data = df_topsoccer,

              order = df_topsoccer['League_from'].value_counts().index)

plt.show()
sns.set(style='darkgrid')

plt.figure(figsize=(15,25))

sns.countplot(y = 'League_to',

              data = df_topsoccer,

              order = df_topsoccer['League_to'].value_counts().index)

plt.show()
sns.set(style='darkgrid')

plt.figure(figsize=(15,25))

sns.countplot(y = 'Season',

              data = df_topsoccer,

              order = df_topsoccer['Season'].value_counts().index)

plt.show()
# Display the histogram to undestand the data



f, axes = plt.subplots(3, figsize=(15, 15))

sns.distplot( df_topsoccer["Age"], ax=axes[0])

sns.distplot( df_topsoccer["Transfer_fee"]/1000, ax=axes[1])

sns.distplot( df_topsoccer["Market_value"]/1000, ax=axes[2])

f, axes = plt.subplots(1, figsize=(10, 5))



sns.scatterplot(x="Transfer_fee", y="Market_value", data=df_topsoccer)

# tabla de contingencia en porcentajes relativos total

pd.crosstab(index=df_topsoccer['Position'], columns=df_topsoccer['League_from'],

            margins=True).apply(lambda r: r/len(df_topsoccer) *100,

                                axis=1)
# tabla de contingencia en porcentajes relativos total

pd.crosstab(index=df_topsoccer['Position'], columns=df_topsoccer['League_to'],

            margins=True).apply(lambda r: r/len(df_topsoccer) *100,

                                axis=1)
pd.crosstab(index=df_topsoccer['Season'], columns=df_topsoccer['Position'],

            margins=True).apply(lambda r: r/len(df_topsoccer) *100,

                                axis=1)
# tabla de contingencia en porcentajes relativos total

pd.crosstab(index=df_topsoccer['Season'], columns=df_topsoccer['Age'],

            margins=True).apply(lambda r: r/len(df_topsoccer) *100,

                                axis=1)
SumPosition = df_topsoccer.reset_index().groupby(('Position','League_from'))['Transfer_fee'].sum()

CountPosition= df_topsoccer.reset_index().groupby(('Position','League_from'))['Transfer_fee'].count()



MedPosition= SumPosition/CountPosition

df_topsoccer1=pd.DataFrame(data=MedPosition)
df_topsoccer1
df_topsoccer2 = pd.merge(df_topsoccer1, df_topsoccer, on='Position' and 'League_from')

df_topsoccer2 
print(df_topsoccer2.columns)
df_topsoccer2 = df_topsoccer2.rename(columns={'Transfer_fee_x': 'MediaXpos'})

df_topsoccer2 = df_topsoccer2.rename(columns={'Transfer_fee_y': 'Transfer_Value'})

df_topsoccer2['Rentable']= np.where(df_topsoccer2['Transfer_Value']>= df_topsoccer2['MediaXpos'],1,0)
sns.set(style='darkgrid')

plt.figure(figsize=(3,5))

sns.countplot(x = 'Rentable',

              data = df_topsoccer2,

              order = df_topsoccer2['Rentable'].value_counts().index)

plt.show()
f, axes = plt.subplots(1,1, figsize=(10, 5), sharex=True, sharey=True)



sns.scatterplot(x="Transfer_Value", y="MediaXpos", data=df_topsoccer2)

x1_bp = df_topsoccer2[(df_topsoccer2['Position']== 'Attacking Midfield') & (df_topsoccer2['Rentable'] == 1) & (df_topsoccer2['Season']=='2017-2018')]

x1_mp = df_topsoccer2[(df_topsoccer2['Position']== 'Attacking Midfield') & (df_topsoccer2['Rentable'] == 0) & (df_topsoccer2['Season']=='2017-2018')]



x2_bp = df_topsoccer2[(df_topsoccer2['Position']== 'Central Midfield') & (df_topsoccer2['Rentable'] == 1) & (df_topsoccer2['Season']=='2017-2018')]

x2_mp = df_topsoccer2[(df_topsoccer2['Position']== 'Central Midfield') & (df_topsoccer2['Rentable'] == 0) & (df_topsoccer2['Season']=='2017-2018')]



x3_bp = df_topsoccer2[(df_topsoccer2['Position']== 'Centre-Back') & (df_topsoccer2['Rentable'] == 1) & (df_topsoccer2['Season']=='2017-2018')]

x3_mp = df_topsoccer2[(df_topsoccer2['Position']== 'Centre-Back') & (df_topsoccer2['Rentable'] == 0) & (df_topsoccer2['Season']=='2017-2018')]
pl.figure(figsize = (20, 10))

pl.plot(x1_bp['Transfer_Value'], x1_bp['MediaXpos'], 'o', label="Attacking Midfield (Bien pago)");

pl.plot(x1_mp['Transfer_Value'], x1_mp['MediaXpos'], 'x', label="Attacking Midfield (Mal Pago)");

pl.plot(x2_bp['Transfer_Value'], x2_bp['MediaXpos'], 'o', label="Central Midfield (Bien pago)");

pl.plot(x2_mp['Transfer_Value'], x2_mp['MediaXpos'], 'x', label="Central Midfield (Mal Pago)");

pl.plot(x3_bp['Transfer_Value'], x3_bp['MediaXpos'], 'o', label="Centre-Back (Bien pago)");

pl.plot(x3_mp['Transfer_Value'], x3_mp['MediaXpos'], 'x', label="Centre-Back (Mal Pago)");

pl.xlabel('Valor de transferencia en el periodo 2017-2018')

pl.ylabel('Transferencia Media')

pl.legend(loc='best');