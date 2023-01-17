# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importing dataset in csv to pandas data frame.

data = pd.read_csv('../input/Life Expectancy Data.csv', delimiter=',')

data.dataframeName = 'Life Expectancy Data.csv'
# Removing spaces from the column names.

data.columns = data.columns.str.replace(' ','')
df = data.copy()

dd = data.copy()

all = data.copy()

data.head()
# Pick required countries.

data = data.loc[data['Country'].isin(['Tonga','Monaco','Oman','Qatar','Finland','Nepal','India','Mexico','Poland'])]
# Drop rows with Null values.

data = data.dropna(axis=0)

data.head()
# Distributions of Immunization coverage for Polio, HepatitisB and Diphtheria

plt.figure(figsize=(8,14))

sns.set(style="ticks")



plt.subplot(3,1,1)

p1 = sns.distplot(data['Polio'],color='r')

p1.set_xlabel("Ploio",fontsize=14)

p1.set_ylabel("% Immunization Coverage",fontsize=14)

p1.set_title("Immunization for Polio, HepatitisB and Diphtheria",fontsize=14)



plt.subplot(3,1,2)

p2 = sns.distplot(data['HepatitisB'],color='b')

p2.set_xlabel("HepatitisB",fontsize=14)

p2.set_ylabel("% Immunization Coverage",fontsize=14)



plt.subplot(3,1,3)

p3 = sns.distplot(data['Diphtheria'],color='orange')

p3.set_xlabel("Diphtheria",fontsize=14)

p3.set_ylabel("% Immunization Coverage",fontsize=14)



plt.show()
# Drop rows with Null values from the main data frame.

all.isnull().sum()

all = all.dropna(axis=0)
# Verifying null values

all.isnull().sum()
columns_to_plot = all.columns[3:6].sort_values()

for column in columns_to_plot: 

    sns.kdeplot(all[column], shade=True)

    plt.title(column)

    plt.xlabel('Values')

    plt.ylabel('Density')

    plt.show()

    print(all[column].describe())
p1 = sns.lmplot(x='Lifeexpectancy',y='AdultMortality',data=data)

plt.title('Bivariate relationship')

plt.show()
p2 = sns.lmplot(x='Lifeexpectancy',y='infantdeaths',data=data)

plt.title('Bivariate relationship')

plt.show()
# Immunization coverage, Lifeexpectancy and AdultMortality across each country.

plt.figure(figsize=(18,18))

sns.set(style='whitegrid')



plt.subplot(3,2,1)

plot1 = sns.pointplot(data=data,x=data.Country,y=data['Lifeexpectancy'])

plot1.set_title("Lifeexpectancy across each country",fontsize=13)



plt.subplot(3,2,2)

plot2 = sns.barplot(data=data,x=data.Country,y=data['AdultMortality'],hue='Status')

plot2.set_title("AdultMortality across each country",fontsize=13)



plt.subplot(3,2,3)

plot3 = sns.boxplot(data=data,x=data.Country,y=data['HepatitisB'],hue='Status')

plot3.set_title("HepatitisB across each country",fontsize=13)



plt.subplot(3,2,4)

plot4 = sns.violinplot(data=data,x=data.Country,y=data['Polio'],hue='Status')

plot4.set_title("Polio across each country",fontsize=13)



plt.subplot(3,2,5)

plot5 = sns.stripplot(data=data,y=data.Country,x=data['Diphtheria'],hue='Status',dodge=True,jitter=True,alpha=1,zorder=1)

sns.pointplot(x=data['Diphtheria'],y=data.Country,hue='Status',data=data,dodge=.532,join=False,palette='dark',scale=.75,ci=None)

plot5.set_title("Diphtheria across each country",fontsize=13)



plt.show()
final = df[df['Country'].isin(['Norway','Italy','Singapore','Japan','Poland','Palau','Mexico','France'])]

final.dropna(inplace=True)

final.isnull().sum()
# Using Subplots

dims = (15,4)

fig, axs = plt.subplots(ncols=4,figsize=dims)

sns.distplot(final.Alcohol,ax=axs[0],color='r')

sns.distplot(final.AdultMortality,ax=axs[1],color='g')

sns.distplot(final.GDP,ax=axs[2])

sns.distplot(final.Population,ax=axs[3],color='orange')

plt.show()
dd.head()
# Find 10 countries with lowest percentange of immunization coverage for "HepatitisB"

dd = dd.groupby('Country').mean().nsmallest(10,'HepatitisB').reset_index()

dd
# Visualizing 10 countries with lowest percentange of immunization coverage for "HepatitisB" using barplots.

plt.figure(figsize=(14,5))

sns.set(style='dark')



plt.subplot(1,2,1)

plot1 = sns.barplot(data=dd,x=dd.Country,y=dd['HepatitisB'])

plot1.set_title("10 countries with low immunization for HepatitisB",fontsize=13)

plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90, ha="right", fontsize=12)



plt.subplot(1,2,2)

plot2 = sns.barplot(data=dd,x=dd.Country,y=dd['infantdeaths'])

plot2.set_title("infantdeaths across each country",fontsize=13)

plot2.set_xticklabels(plot2.get_xticklabels(), rotation=90, ha="right", fontsize=12)





#ax = plt.scatter(x='HepatitisB',y='infantdeaths',data=data)

plt.show()