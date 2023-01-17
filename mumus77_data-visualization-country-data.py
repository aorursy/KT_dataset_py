#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline 
import seaborn as sns
mpl.style.use('ggplot')


#Reading the file. I checked the file beforehand and saw that data is read with "," as decimal separator rather than default ".". So, I made sure that decimals are read as ",".
df = pd.read_csv('../input/countries of the world.csv', decimal=',')
#Checking the overall structure
df.head()
#Checking data field types
df.dtypes

#Checking if there are missing values
df.isnull().sum()
#As there are some missing values and there is already low number of data, instead of dropping them, I replaced missing values for each row, with mean of the region that specific country belongs to.
df['Service'] = df.groupby(['Region'])['Service'].transform(lambda x: x.fillna(x.mean()))
df['Industry'] = df.groupby(['Region'])['Industry'].transform(lambda x: x.fillna(x.mean()))
df['Agriculture'] = df.groupby(['Region'])['Agriculture'].transform(lambda x: x.fillna(x.mean()))
df['Deathrate'] = df.groupby(['Region'])['Deathrate'].transform(lambda x: x.fillna(x.mean()))
df['Birthrate'] = df.groupby(['Region'])['Birthrate'].transform(lambda x: x.fillna(x.mean()))
df['Climate'] = df.groupby(['Region'])['Climate'].transform(lambda x: x.fillna(x.mean()))
df['Other (%)'] = df.groupby(['Region'])['Other (%)'].transform(lambda x: x.fillna(x.mean()))
df['Crops (%)'] = df.groupby(['Region'])['Crops (%)'].transform(lambda x: x.fillna(x.mean()))
df['Arable (%)'] = df.groupby(['Region'])['Arable (%)'].transform(lambda x: x.fillna(x.mean()))
df['Literacy (%)'] = df.groupby(['Region'])['Literacy (%)'].transform(lambda x: x.fillna(x.mean()))
df['Phones (per 1000)'] = df.groupby(['Region'])['Phones (per 1000)'].transform(lambda x: x.fillna(x.mean()))
df['GDP ($ per capita)'] = df.groupby(['Region'])['GDP ($ per capita)'].transform(lambda x: x.fillna(x.mean()))
df['Infant mortality (per 1000 births)'] = df.groupby(['Region'])['Infant mortality (per 1000 births)'].transform(lambda x: x.fillna(x.mean()))
df['Net migration'] = df.groupby(['Region'])['Net migration'].transform(lambda x: x.fillna(x.mean()))

#Re-checking the missing values, if we actually filled the missing ones.
df.isnull().sum()
df.hist(bins=100, figsize=(20,20) , color = 'b')
x = df.groupby("Region", as_index=False)["Literacy (%)"].mean()
x.set_index("Region",drop=True,inplace=True)
ax = x.plot(kind="bar", title ="Literacy Rate by Region", figsize=(10, 8), legend=True, fontsize=10)
ax.set_xlabel("Region", fontsize=12)
ax.set_ylabel("Literacy (%)", fontsize=12)
plt.show()
x = df.groupby("Region", as_index=False)["GDP ($ per capita)"].mean()
x.set_index("Region",drop=True,inplace=True)
plt.figure(figsize=(50,50))
x.plot(kind = "pie",colormap = "jet", subplots = True, autopct='%1.1f%%', legend=False)
plt.title("% of Total GDP per Capita by Regions")
plt.figure(figsize=(10,6))
sns.distplot(df['Infant mortality (per 1000 births)'],color='red')
plt.title("Distribution of Infant Mortality Rate")
plt.show()
plt.figure(figsize=(12,7))
sns.heatmap(cbar=True,annot=True,data=df.corr()*100,cmap='Greens')
plt.title('% Correlation Matrix')
plt.show()
plt.figure(figsize=(20,6))
plt.subplots_adjust(hspace = .25)
plt.subplot(1,2,1)
plt.xlabel('GDP ($ per capita)',fontsize=12)
plt.ylabel('Phones (per 1000)',fontsize=12)
sns.stripplot(data=df,x='GDP ($ per capita)',y='Phones (per 1000)')

ax = sns.regplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=df)
ax = sns.regplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=df, color='green')
ax = sns.regplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=df, color='green', marker='+')
plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)
sns.set_style('whitegrid')
ax = sns.regplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=df, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='GDP ($ per capita)', ylabel='Phones (per 1000)')
ax.set_title('Regression Between Income per Capita & Number of Phones Purchased')

