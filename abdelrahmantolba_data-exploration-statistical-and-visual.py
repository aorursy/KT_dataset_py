import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats
data = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')
data
# Know the size of the data uaing .shape function 

print('The size of the data is',data.shape[0],'rows and ', data.shape[1],'columns and the shape is ',data.shape)
data.columns
data.describe()
data.head()
data.tail(7)
# to see a specific row 

data.iloc[150]
# examinig punch of rows 

data.iloc[50:55]
# five random rows from the the data  

data.sample(5)
data['Legendary'].value_counts()
result = data['Legendary'].value_counts()

series = pd.Series(

    np.array([result[1],result[0]]),

    index = ['Legandary', 'NonLegendary']

                )

print(series)
# first we know what is the maximum attack value 

data['Attack'].max()

# now we query the whole record using the data we had from the previous 

data.query("Attack == 190")
# display all the legendary pocemons 

data.query('Legendary == True')
print('Maximum value of HP')

print(data['HP'].max())

print('----------------------')

print('Mimimum value of HP')

print(data['HP'].min())

# or u can easily use describe() function that we used before on that specific feature 

data['Attack'].describe()
# We can easily use describe() function that we used before on that specific feature 

data['Attack'].describe()
frame = data.groupby('Generation').mean()

frame
# we will plot the distripution of the attack feature using seaborn library 

sns.distplot(data['Attack']);
# look at the values of skewness and kurtosis of the prevoius graph 

print("Skewness: %f" % data['Attack'].skew())

print("Kurtosis: %f" % data['Attack'].kurt())
sns.distplot(data['HP']);
print("Skewness: %f" % data['HP'].skew())

print("Kurtosis: %f" % data['HP'].kurt())
var2 = 'Attack'

var1 = 'Total'

pd.concat([data[var2], data[var1]], axis=1)

data.plot.scatter(x=var1, y=var2, ylim=(0,200));
var1 = 'Generation'

var2 = 'Total'

pd.concat([data[var1], data[var2]], axis=1)

fig = sns.boxplot(x=var1, y=var2, data=data)
var1 = 'Speed'

var2 = 'Total'

pd.concat([data[var1], data[var2]], axis=1)

fig = sns.boxplot(x=var1, y=var2, data=data)
corrmat = data.corr()

corrmat
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=0.9, square=True);
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=0.9, square=True,annot=True);
sns.set()

cols = [ 'Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed' ,'Generation']

sns.pairplot(data[cols], size = 1.5) 

plt.show();
# see the count of every record in that col 

data['#'].value_counts()
# see the count of every record in the name columne 

data['Name'].value_counts()
sns.distplot(data['#'])
var = 'Attack'

var0 = '#'

pd.concat([data[var0], data[var]], axis=1)

data.plot.scatter(x=var, y=var0, ylim=(0,800),xlim=(0,180));
var = 'HP'

var0 = '#'

pd.concat([data[var0], data[var]], axis=1)

data.plot.scatter(x=var, y=var0, ylim=(0,800),xlim=(0,150));
data = data.drop('#',axis=1)

data.head()
data.columns
# see how many null value in every column 

All = data.isnull().sum().sort_values(ascending=False)

All
percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

percentage
missing_data = pd.concat([All, percentage], axis=1, keys=['All', 'percentage'])

missing_data.head(10)
New_data = data.drop('Type 2' , axis = 1)

New_data.head()
# deleting all the rows with at least one null value 

t1 = data.dropna()

All = t1.isnull().sum().sort_values(ascending=False)

All
#deleting all the rows that has all the values in them null  

t2 = data.dropna(how = 'all')
All = t2.isnull().sum().sort_values(ascending=False)

All
# filling the missing valuse in type 2 col witth a value of the previous row 

temp  = data['Type 2'].fillna(method='ffill')

temp
missing = temp.isnull().sum()

missing
sns.distplot(data['HP']);
sns.distplot(data['Defense'], fit=norm);

fig = plt.figure()

res = stats.probplot(data['Defense'], plot=plt)
#applying log transformation

data['Defense'] = np.log(data['Defense'])
sns.distplot(data['Defense'], fit=norm);

fig = plt.figure()

res = stats.probplot(data['Defense'], plot=plt)