# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from collections import Counter



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import cross_val_score
#Get data

train = pd.read_csv('../input/vgsales.csv')
#Taking a look at the data

train.describe()
fig,ax = plt.subplots(figsize=(8,5))

train['Genre'].value_counts(sort=False).plot(kind='bar',ax=ax,rot =90)

plt.title('Genre Distribution')

plt.xlabel('Genre')

plt.ylabel('Number of sales')
#Some genres have barely any sales. Looking at top 7 out of 12.

genre = Counter(train['Genre'].dropna().tolist()).most_common(7)

genre_name = [name[0] for name in genre]

genre_counts = [name[1] for name in genre]



fig,ax = plt.subplots(figsize=(8,5))

sns.barplot(x=genre_name,y=genre_counts,ax=ax)

plt.title('Top Seven Genres')

plt.xlabel('Genre')

plt.ylabel('Number of Games')
#We know there to be an even larger number of unique cases of platforms and publishers

#Yet again, just looking at top 7 for each

platforms = Counter(train['Platform'].dropna().tolist()).most_common(7)

platform_name = [name[0] for name in platforms]

platform_counts = [name[1] for name in platforms]



fig,ax = plt.subplots(figsize=(8,5))

sns.barplot(x=platform_name,y=platform_counts,ax=ax)

plt.title('Top Seven Platforms')

plt.xlabel('Platform')

plt.ylabel('Number of Games')
#And for Publisher

publisher = Counter(train['Publisher'].dropna().tolist()).most_common(7)

publisher_name = [name[0] for name in publisher]

publisher_counts = [name[1] for name in publisher]



fig,ax = plt.subplots(figsize=(8,5))

sns.barplot(x=publisher_name,y=publisher_counts,ax=ax)

plt.title('Top Seven Publishers')

plt.xlabel('Publishers')

plt.ylabel('Number of Games')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
min_year = int(train['Year'].dropna().min())

max_year = int(train['Year'].dropna().max())

sales = []

years = []

for year in range(min_year,max_year-3):

    sales.append(train[train['Year'] == year].dropna()['Global_Sales'].sum())

    years.append(year)



fig,ax = plt.subplots(figsize=(10,6))

sns.barplot(x = years,y = sales,ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('Sales per Year')

plt.ylabel('Global Sales')

plt.xlabel('Year')
#Looking at how sales are related to Our categorical features of Publisher, Genre, and Platform.

#Starting with Genre

#We dont really see a trend other than that the greatest variety of games happened around 2010

table_count = pd.pivot_table(train,values=['Global_Sales'],index=['Year'],columns=['Genre'],aggfunc='count',margins=False)



plt.figure(figsize=(19,16))

sns.heatmap(table_count['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)

plt.title('Count of games')
#We find that there are 579 unique publishers. We take only the top 25.

#We notice a probable outlier for Nintendo in 2006.

most_published = train.groupby('Publisher').Global_Sales.sum()

most_published.sort_values(ascending=False)[:25]

table_publishers =  pd.pivot_table(train[train.Publisher.isin(most_published.sort_values(ascending=False)[:25].index)],values=['Global_Sales'],index=['Year'],columns=['Publisher'],aggfunc='sum',margins=False)



plt.figure(figsize=(19,16))

sns.heatmap(table_publishers['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)

plt.title('Count of games')
len(set(train['Platform']))
#We find that there are 31 unique platforms.

#Looks good. We notice that (besides for the PC), games are released consecutive years for a given platform.

#Anomoly for the DS in 1985 (the DS wasn't released until 2004)

table_platforms = pd.pivot_table(train,values=['Global_Sales'],index=['Year'],columns=['Platform'],aggfunc='count',margins=False)



plt.figure(figsize=(19,16))

sns.heatmap(table_platforms['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)

plt.title('Count of games')
#Checking for Missing Data

train = train.drop(['Rank'], axis=1)



#Fixing Missing Data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(9)
#Since the games with missing information comprise such a small %, I choose to just drop them.

#Realistically, would just google and manually fill or write a script.

X_train = train.dropna()

X_train.isnull().sum().max()       #to check if any missing data remains
#Now looking to see if there are any correlations between our numerical features

#Not surprisingly, sales in different regions are correlated, but with little correlation to year.

corrmat = train.corr()

sns.heatmap(corrmat, vmax=.8, square=True);
#Looking for Outliers!

Global_Sales_Scaled = StandardScaler().fit_transform(X_train['Global_Sales'][:,np.newaxis]);

low_range = Global_Sales_Scaled[Global_Sales_Scaled[:,0].argsort()][:10]

high_range= Global_Sales_Scaled[Global_Sales_Scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#We find that the largest selling game (which happens to be Wii Sports) is probably an outlier.

#We simply drop it. This accounts for Nintendo's oddly high sales in 2006.

#The high sales are likely because Wii Sports came with every Wii console, causing its sales to equal Wii console sales.

X_train = X_train.drop(X_train[X_train['Global_Sales'] == 82.74].index)   #This is WiiSports, which comes with sales of the Nintendo Wii
#There was another anomaly with the DS in 1985.

#We drop that instance as well.

X_train = X_train.drop(X_train[X_train['Name'] == 'Strongest Tokyo University Shogi DS'].index)
#We also notice that there are some games with release dates in 2017 and later. These shouldn't be there

X_train = X_train.drop(X_train[X_train['Year'] >2016].index)
#Now we look for normality in our data

sns.distplot(X_train['Global_Sales'])

print("Skewness: %f" % X_train['Global_Sales'].skew())

print("Kurtosis: %f" % X_train['Global_Sales'].kurt())
res = stats.probplot(X_train['Global_Sales'], plot=plt)
#We attempt to normalize by taking the log

X_train['Global_Sales'] = np.log(X_train['Global_Sales'])

sns.distplot(X_train['Global_Sales'])
res = stats.probplot(X_train['Global_Sales'], plot=plt)
#We notice that the Sales in other regions are also not normally distributed, but will have to figure something out.

    #Reason is that some regions have 0 sales for some games

X_train.describe()
#Preparing to spot test models. 

Y_train = X_train['Global_Sales']

X_train = X_train.drop(['Global_Sales', 'Name'], axis=1)

X_train = pd.get_dummies(X_train)

X_train.shape, Y_train.shape
#Spot checking different algorithms

ridge = linear_model.Ridge()

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results_ridge = cross_val_score(ridge, X_train, Y_train, cv=cv)

results_ridge = results_ridge.mean()*100



B_ridge = linear_model.BayesianRidge()

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results_B_ridge = cross_val_score(B_ridge, X_train, Y_train, cv=cv)

results_B_ridge = results_B_ridge.mean()*100



huber = linear_model.HuberRegressor()

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results_huber = cross_val_score(huber, X_train, Y_train, cv=cv)

results_huber = results_huber.mean()*100



lasso = linear_model.Lasso(alpha=1e-4)

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results_lasso = cross_val_score(lasso, X_train, Y_train, cv=cv)

results_lasso = results_lasso.mean()*100



bag = BaggingRegressor()

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results_bag = cross_val_score(bag, X_train, Y_train, cv=cv)

results_bag = results_bag.mean()*100



forest = RandomForestRegressor()

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results_forest = cross_val_score(forest, X_train, Y_train, cv=cv)

results_forest = results_forest.mean()*100



ada = AdaBoostRegressor()

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results_ada = cross_val_score(ada, X_train, Y_train, cv=cv)

results_ada = results_ada.mean()*100



models = pd.DataFrame({

    'Model': ['Ridge', 'Bayesian Ridge', 'Huber', 'Lasso', 'Bagging', 'Random Forest', 'AdaBoost'],

    'Score': [results_ridge, results_B_ridge, results_huber, results_lasso, results_bag, results_forest, results_ada]})

print(models.sort_values(by='Score', ascending=False))