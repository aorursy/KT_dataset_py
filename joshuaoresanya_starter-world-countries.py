import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns



from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import os # accessing directory structure
dataframe = pd.read_csv('../input/countries of the world.csv', sep= ',' , decimal=',' )

#(sep= ',' , decimal=',') is added to replace comma separated values with dot for decimal conversion.
dataframe.head()
dataframe.columns
dataframe.dtypes
df = dataframe.drop(['Area (sq. mi.)', 'Pop. Density (per sq. mi.)','Coastline (coast/area ratio)','Infant mortality (per 1000 births)',

               'Other (%)', 'Agriculture', 'Industry', 'Service', 'Arable (%)','Crops (%)','Climate' ], axis=1)

df
data = dataframe
data.info()
data.describe()
data.head()
data.index

data
print('Dataset has null values?')

data.isnull().values.any()
for col in data.columns:

    print(col  , (data[col].isnull().sum()/len(data[col])*100 ))
data = data.fillna(0)

for col in data.columns:

    print(col  , (data[col].isnull().sum()/len(data[col])*100 ))

#Recheck for missing values

print('Dataset has null values?')

data.isnull().values.any()
## to read specific columns

print(data[['Region','Country', 

                 'Population','Net migration','GDP ($ per capita)']])
  ##To read each row, the code below can be used. But I would comment it out as the result would be excessive. 

#print(data_drop.iloc[1:4])

#for index, row in data.iterrows():

 #   print(index,row)

                     
data.sort_values(by=['Region', 'Country','Net migration','GDP ($ per capita)'])
avg_population = sum(data.Population)/len(data.Population)

print("avg_population: ",avg_population)
data['Region'].unique()
len(data['Region'].unique())
data['Net migration'].mean()
data['Net migration'].min()
data['Net migration'].max()
data['Net migration'].std()
data['Net migration'].mode()
data['Net migration']
data['GDP ($ per capita)'].mean()
data['GDP ($ per capita)'].min()
data['GDP ($ per capita)'].max()
data['GDP ($ per capita)'].std()
data['GDP ($ per capita)'].mode()

region = data["Region"].value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=region.index,y=region.values)

plt.xticks(rotation=90)

plt.ylabel('Number of countries')

plt.xlabel('Region')

plt.title('Number of Countries by REGİON',color = 'red',fontsize=20)

plt.plot()
region = data["Region"].value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=region.index,y=region.values)

plt.xticks(rotation=90)

plt.ylabel('GDP ($ per capita)')

plt.xlabel('Region')

plt.title('GDP by REGİON',color = 'green',fontsize=20)

plt.plot()
#Box Plot

group = data.groupby("Region")

group.mean()



sns.boxplot(x=data["Region"],y=data["GDP ($ per capita)"],data=data, width=0.7,palette="Set2",fliersize=5)

plt.xticks(rotation=90)

plt.title("GDP BY REGİON",color="green")



plt.plot()
region = data["Region"].value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=region.index,y=region.values)

plt.xticks(rotation=90)

plt.ylabel('Net migration')

plt.xlabel('Region')

plt.title('Net migration by REGİON',color = 'blue',fontsize=20)

plt.plot()
group = data.groupby("Region")

group.mean()



sns.boxplot(x=data["Region"],y=data["Net migration"],data=data, width=0.7,palette="Set2",fliersize=5)

plt.xticks(rotation=90)

plt.title("Net migration BY REGİON",color="red")



plt.plot()
group = data.groupby("Region")

group.mean()



sns.boxplot(x=data["Region"],y=data["Population"],data=data, width=0.7,palette="Set2",fliersize=5)

plt.xticks(rotation=90)

plt.title("Population BY REGİON",color="red")



plt.plot()
region = data["Region"].value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=region.index,y=region.values)

plt.xticks(rotation=90)

plt.ylabel('Population')

plt.xlabel('Region')

plt.title('Net migration by REGİON',color = 'blue',fontsize=20)

plt.plot()
sns.set(style="whitegrid")

sns.violinplot(x="Literacy (%)",y="Phones (per 1000)",data=df,hue="Region",palette="PRGn")



plt.plot()
data.plot(kind='scatter', x='Literacy (%)', y='GDP ($ per capita)')
sns.stripplot(x="Region",y="Population",data=data,color="m")



plt.xticks(rotation=90)



plt.plot
plt.scatter(data["Birthrate"],data["Deathrate"],marker='^',facecolor='green')



plt.grid(True)



plt.xlabel('Birthrate')

plt.ylabel('Deathrate')



plt.title("Scatter Plot")



plt.legend(loc='upper left')



plt.show()
#Histogram showing the rate of net migration across regions

plt.hist(data["Net migration"], bins=10,density=True, facecolor='g', alpha=0.75)



plt.xlabel('Net migration')



plt.ylabel('Rate')



plt.title('Histogram Plot')



plt.grid(True)



plt.show()
explode = (0, 0.1, 0, 0,0,0,0)



sizes=[15,10,25,5,30,5,10]



labels="ASIA","EASTERN EUROPE","NORTHERN AFRICA","OCEANIA","WESTERN EUROPE","SUB-SAHARAN AFRICA","NORTHERN AMERICA"



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels,explode=explode, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
#correlation

data.corr()
#Visual representation in form of heatmap for correlated data

plt.figure(figsize=(16,12))

ax=plt.axes()

sns.heatmap(data=data.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax)

ax.set_title('Heatmap showing correlated values for the Dataset')

plt.show()
# By looking at the heatmap for a given dataset:

#We can say that following factors are positively correlated with Net migration:- r>0 for:-



#GDP_Per_Capita = (0.38) - highest postive correlation with Net migration



#Phones - (0.24)



#Deathrate = (0.04)



#Following values are inversely correlated with GDP per capita:- r<0 for:-



#Literacy (%) = (-0.02)



#Birthrate = (-0.06) - Highly negatively correlated



#Population = (0.00)



#Note that net migration has 0.00 corellation with population
x = data[['GDP ($ per capita)', 'Phones (per 1000)', 'Literacy (%)', 'Net migration', 'Birthrate','Deathrate', 'Population']]
# show corr of the same

plt.figure(figsize=(10,5))

ax=plt.axes()

sns.heatmap(x.corr(), annot=True,ax=ax)

ax.set_title('Heatmap showing correlated values for the Dataset')

plt.show()
x = data[['GDP ($ per capita)','Literacy (%)','Birthrate','Region']]



g=sns.pairplot(x, hue="Region", diag_kind='hist')

g.fig.suptitle('Pairplot showing GDP per capita, Literacy and Birthrate against Region',y=1.05)
data['Total_GDP ($)'] = data['GDP ($ per capita)'] * data['Population']

top_gdp_countries = data.sort_values('Total_GDP ($)',ascending=False).head(10)

other = pd.DataFrame({'Country':['Others'], 'Total_GDP ($)':[data['Total_GDP ($)'].sum() - top_gdp_countries['Total_GDP ($)'].sum()]})

gdps = pd.concat([top_gdp_countries[['Country','Total_GDP ($)']],other],ignore_index=True)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7),gridspec_kw = {'width_ratios':[2,1]})

sns.barplot(x='Country',y='Total_GDP ($)',data=gdps,ax=axes[0],palette='Set2')

axes[0].set_xlabel('Country',labelpad=30,fontsize=16)

axes[0].set_ylabel('Total_GDP',labelpad=30,fontsize=16)



colors = sns.color_palette("Set2", gdps.shape[0]).as_hex()

axes[1].pie(gdps['Total_GDP ($)'], labels=gdps['Country'],colors=colors,autopct='%1.1f%%',shadow=True)

axes[1].axis('equal')

plt.show()
Rank_total_gdp = data[['Country','Total_GDP ($)']].sort_values('Total_GDP ($)', ascending=False).reset_index()

Rank_gdp = data[['Country','GDP ($ per capita)']].sort_values('GDP ($ per capita)', ascending=False).reset_index()

Rank_total_gdp= pd.Series(Rank_total_gdp.index.values+1, index=Rank_total_gdp.Country)

Rank_gdp = pd.Series(Rank_gdp.index.values+1, index=Rank_gdp.Country)

Rank_change = (Rank_gdp-Rank_total_gdp).sort_values(ascending=False)

print('rank of total GDP - rank of GDP per capita:')

Rank_change.loc[top_gdp_countries.Country]
plt.figure(figsize=(16,12))

ax=plt.axes()

y=data[data.columns[2:]].apply(lambda x: x.corr(data['Total_GDP ($)']))

print(y)

sns.heatmap(data=data.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax)

ax.set_title('Heatmap showing correlated values for the Dataset with respect to total ')

plt.show()
#Conclusions

#The analysis was performed mostly with regards to correlation between Net migration and other factors or variables 

#such as Literacy, Population, Birthrate, Deathrate, etc.

#Net migration is correlated with many of the other variables not limited to literacy, birthrate, etc.

#Across the globe, Net migration is positively correlated with GDP per capita, Phones, and so on within each region. 

#Regions with more technological advancement, where people tend to buy more phones happen to have more Net migration. 

#Density distribution is mostly skewed for highly correlated factors. 

#Key findings leads us to know that Net migration is positively correlated with the factors such as GDP per capita.

#Population has no correlation with Net Migration.