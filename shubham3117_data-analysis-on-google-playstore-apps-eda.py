# Importing some librabries for reading the data, importing some packages and for EDA:
import os
import pandas as pd
import numpy as np
import seaborn as sns


from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import rcParams
apps = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
apps.head()
apps.describe()
# dimension of dataset
apps.shape
#Setting options to display all rows and columns

pd.options.display.max_columns=None
pd.options.display.max_rows=None
pd.options.display.width=None
apps.isnull().sum()
apps["Rating"].unique()
apps["Category"].value_counts().plot.barh(title="Count on Category distribution",figsize=(12,10))
apps['Category'].value_counts()/apps["Category"].count()*100
# Data to plot
labels =apps['Type'].value_counts(sort = True).index
sizes = apps['Type'].value_counts(sort = True)


colors = ["palegreen","blue"]
explode = (0.1,0)  # explode 1st slice
 
rcParams['figure.figsize'] = 8,10
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percent distribution of App in store',size = 25)
plt.show()
pd.crosstab(apps["Category"], apps["Type"]).sort_values(ascending=True, by='Category')
pd.crosstab(apps["Category"], apps["Type"]).plot.bar(figsize=(15,8))
#rest of the missing data
total = apps.isnull().sum().sort_values(ascending=False)
percent = (apps.isnull().sum()/apps.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(4)
apps.head()
# Contribution in null value by Cuurent Ver, Android ver, Content Rating & Type is hardly make .5% so it's better to drop null values of them. 
apps.dropna(how='any', inplace=True)
# final check:
apps.isnull().sum()
# Changed dimension:
apps.shape
apps['Rating'].describe()
apps['Content Rating'].value_counts()
sns.boxplot(apps["Rating"])
plt.figure(figsize=(14, 7))
ax = sns.boxplot(x="Rating", y="Content Rating", hue='Type', data=apps, palette="Set3", linewidth=2, saturation=0.75, width=0.8)
# Unrated rating has on 1 application so its better to drop that category for future simplification of calculation & model building:
apps = apps[apps['Content Rating'] != 'Unrated']
apps['Content Rating'].unique()
# Convert that category to numeric form for further calculations:
apps = pd.get_dummies(apps, columns= ["Content Rating"])
plt.figure(figsize=(14, 12))
ax = sns.boxplot(x="Rating", y="Category", data=apps, palette="Set3", linewidth=2, saturation=0.75, width=0.8)
apps['Price'] = apps['Price'].str.replace('$',"")
apps['Price'] = apps['Price'].apply(lambda x: float(x))
apps.Price.describe()
apps[apps['Price'] == 400]
plt.figure(figsize = (10,10))
sns.regplot(x="Price", y="Rating", color = 'darkorange',data=apps[apps['Reviews']<1000000]);
plt.title('Scatter plot Rating VS Price',size = 20)
# Convert Review from Object to Int
apps['Reviews'] = apps['Reviews'].apply(lambda x: int(x))
apps['Reviews'].describe()
# rating distibution 
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(apps.Reviews, color="Green", shade = True)
g.set_xlabel("Reviews")
g.set_ylabel("Frequency")
plt.title('Distribution of Reveiw',size = 20)
bins=[0,10000,100000,200000,300000,500000] 
group=['Low_Rcount','Average_Rcount','High_Rcount', 'Very high_Rcount', 'Trending_Rcount']
apps['Reviews_bins']=pd.cut(apps['Reviews'],bins,labels=group)
pd.crosstab(apps['Reviews_bins'],apps['Type'])

apps['Reviews_bins'].value_counts().plot.bar()
pd.crosstab(apps['Reviews_bins'],apps['Type']).plot.barh(figsize=(18,8))
# to check the interpretation we can see below chart:
plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'darkorange',data=apps[apps['Reviews']<1000000]);
plt.title('Rating VS Reveiws',size = 20)
apps['Installs'].unique()
apps["Installs"].value_counts().plot.bar(title= "User Count on Installation",figsize = (10,8))
# Removing punchuations & '+' sign from Intalls column:
apps.Installs = apps.Installs.apply(lambda x: x.replace(',',''))
apps.Installs = apps.Installs.apply(lambda x: x.replace('+',''))
apps.Installs = apps.Installs.apply(lambda x: int(x))
apps.Installs.unique()
Sorted_value = sorted(list(apps['Installs'].unique()))
apps['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace=True)
apps.Installs.head(3)
# As per our primary focus we can now check the crelation between rating & installs:
pd.crosstab(apps.Installs, apps.Rating)
# rather check with visualization:
plt.figure(figsize = (10,10))
sns.regplot(x="Installs", y="Rating", color = 'red',data=apps);
plt.title('Rating & Installs',size = 18)
apps.Genres.unique()
apps.Genres.value_counts().head()
apps.Genres.value_counts().tail(10)
apps['Genres'] = apps['Genres'].str.split(';').str[0]
apps.Genres.unique()
apps['Genres'].value_counts().tail()
apps['Genres'].replace('Music & Audio', 'Music',inplace = True)
apps[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().describe()
apps.tail(3)
apps.Size.unique()
# Convert kb to Mb so that we can scale up and calculate further modeling on 1 paramter only
k_indices = apps['Size'].loc[apps['Size'].str.contains('k')].index.tolist() 
# Convert that kb by divinding the size with 1024
converter = pd.DataFrame(apps.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x/1024).apply(lambda x: round(x, 3)).astype(str))
apps.loc[k_indices,'Size'] = converter
# Now we can see all the size is converted on the scale of MB.
converter.head()
apps.dtypes
# Type encoding
apps['Type'] = pd.get_dummies(apps['Type'])
apps.head()