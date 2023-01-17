# importing libraries

%matplotlib inline 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy import stats as st

import seaborn as sns

import re

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import missingno as msno
# importing the dataset

df=pd.read_csv("countries of the world.csv", decimal = ',')
df.head()
df.describe()
top_gdp_countries = df.sort_values('GDP ($ per capita)',ascending=False)

# Look at top 20

top_gdp_countries[['GDP ($ per capita)','Country']].head(20)
fig, ax = plt.subplots(figsize=(16,6))

sns.barplot(x='Country', y='GDP ($ per capita)', data=top_gdp_countries.head(33), palette='Set1')

ax.set_xlabel(ax.get_xlabel(), labelpad=15)

ax.set_ylabel(ax.get_ylabel(), labelpad=30)

ax.xaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontsize(16)

plt.xticks(rotation=90)

plt.show()
fig, ax = plt.subplots(figsize=(16,6))

sns.barplot(x='Country', y='GDP ($ per capita)', data=top_gdp_countries.tail(33), palette='Set1')

ax.set_xlabel(ax.get_xlabel(), labelpad=15)

ax.set_ylabel(ax.get_ylabel(), labelpad=30)

ax.xaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontsize(16)

plt.xticks(rotation=90)

plt.show()
numeric_features = df.select_dtypes(include=[np.number])



numeric_features.columns
categorical_features = df.select_dtypes(include=[np.object])

categorical_features.columns
total = df.isnull().sum()[df.isnull().sum() != 0].sort_values(ascending = False)

percent = pd.Series(round(total/len(df)*100,2))

pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
print(np.isnan(df['GDP ($ per capita)']))
gdp_no_nan = df['GDP ($ per capita)'][~np.isnan(df['GDP ($ per capita)'])]
print(np.isnan(gdp_no_nan))
msno.matrix(df)
msno.heatmap(df)
msno.bar(df)
 
for col in df.columns.values:

    if df[col].isnull().sum() == 0:

        continue

    if (col != 'Country') & (col !='Region') & (col != 'Climate'):

        guess_values = df.groupby('Region')[col].median()

    else:

        guess_values = df.groupby('Region')[col].mode()

    for region in df['Region'].unique():

        df[col].loc[(df[col].isnull())&(df['Region']==region)] = guess_values[region]

    
df.isnull().sum()
df.isnull().sum().sum()
msno.matrix(df)
df.skew()
y = df['Population']

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)

y = df['Literacy (%)']

plt.figure(3); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)
df.kurt()
sns.distplot(df.skew(),color='blue',axlabel ='Skewness')
plt.figure(figsize = (12,8))

sns.distplot(df.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)

#plt.hist(train.kurt(),orientation = 'vertical',histtype = 'bar',label ='Kurtosis', color ='blue')

plt.show()
sns.boxplot(df['GDP ($ per capita)'])
sns.boxplot(np.array(df['Birthrate']))
sns.boxplot(np.array(df['Crops (%)']))
sns.boxplot(np.array(df['Infant mortality (per 1000 births)']))
sns.boxplot(np.array(df['Net migration']))
sns.boxplot(np.array(df['Deathrate']))
sns.boxplot(np.array(df['Literacy (%)']))
sns.boxplot(np.array(df['Agriculture']))
sns.boxplot(np.array(df['Phones (per 1000)']))
sns.boxplot(np.array(df['Service']))
plt.hist(df['GDP ($ per capita)'],orientation = 'vertical',histtype = 'bar', color ='blue')

plt.show()
y = df['GDP ($ per capita)']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)
numeric_features.corr()
plt.figure(figsize=(16,12))

sns.heatmap(data=df.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm')

plt.show()
# choose attributes which shows relation

x = df[['Net migration','GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Service','Infant mortality (per 1000 births)','Birthrate','Deathrate']]
# show corr of the same

sns.heatmap(x.corr(), annot=True, fmt='.2f',cmap='coolwarm')
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))

plt.subplots_adjust(hspace=0.4)



corr_to_gdp = pd.Series()

for col in df.columns.values[2:]:

    if (col!='GDP ($ per capita)'):

        corr_to_gdp[col] = df['GDP ($ per capita)'].corr(df[col])

abs_corr_to_gdp = corr_to_gdp.abs().sort_values(ascending=False)

corr_to_gdp = corr_to_gdp.loc[abs_corr_to_gdp.index]



for i in range(3):

    for j in range(3):

        sns.regplot(x=corr_to_gdp.index.values[i*3+j], y='GDP ($ per capita)', data=df,

                   ax=axes[i,j], fit_reg=False, marker='.')

        title = 'correlation='+str(corr_to_gdp[i*3+j])

        axes[i,j].set_title(title)

axes[1,2].set_xlim(0,102)

plt.show()
x.corr()
#checking distribution of literacy of the world

sns.distplot(np.array(df['Phones (per 1000)']))
#checking distribution of literacy of the world

sns.distplot(np.array(df['Service']))
sns.distplot(np.array(gdp_no_nan))
# choose attributes which shows relation and make the pairplots

x = df[['GDP ($ per capita)','Phones (per 1000)','Service']]
sns.pairplot(x)