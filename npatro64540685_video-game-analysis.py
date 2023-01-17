# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

#import researchpy as rp

import statsmodels.api as sm

from statsmodels.formula.api import ols

%matplotlib inline



df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')



df['Publisher'] = df['Publisher'].astype('category')

df['Genre'] = df['Genre'].astype('category')

df['Platform'] = df['Platform'].astype('category')
df[df.isnull().any(axis=1)]

df = df.dropna(axis=0, subset=['Year', 'Publisher', 'Platform', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
ref = pd.crosstab(df.Genre, df.Platform)

ref_total = ref.sum(axis = 1).sort_values(ascending = True)

sns.barplot(y = ref_total.index, x = ref_total.values, orient='h')
sns.heatmap(df.corr(), annot=True)
results = ols('Global_Sales ~ Year', data=df).fit()

results.summary()

aov_table = sm.stats.anova_lm(results, typ=2)

aov_table
list1 = list()

mylabels = list()

for genre in df.Genre.cat.categories:

    list1.append(df[df.Genre == genre].Year)

    mylabels.append(genre)

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(20,25)

h = plt.hist(list1, bins=30, stacked=True, rwidth=1, label=mylabels)

plt.title("Video Game Releases by Genre",fontsize=35, color="DarkBlue", fontname="Console")

plt.ylabel("Number of Releases", fontsize=35, color="Red")

plt.xlabel("Year", fontsize=35, color="Green")

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.legend(frameon=True,fancybox=True,shadow=True,framealpha=1,prop={'size':10})

plt.show()

list1 = list()

mylabels = list()

for platform in df.Platform.cat.categories:

    list1.append(df[df.Platform == platform].Year)

    mylabels.append(platform)

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(30,25)

h = plt.hist(list1, bins=30, stacked=True, rwidth=1, label=mylabels)

plt.title("Video Game Releases by Platform",fontsize=35, color="Red", fontname="Console")

plt.ylabel("Number of Releases", fontsize=35, color="Red")

plt.xlabel("Year", fontsize=35, color="Green")

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.legend(frameon=True,fancybox=True,shadow=True,framealpha=1,prop={'size':10})

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(15,20)

pvdf = df.pivot_table(values='Global_Sales',index='Platform',columns='Genre',  fill_value=0)

#pvdf[pvdf['Global_Sales'] != 0]

pf = sns.heatmap(pvdf, cmap = "coolwarm",linecolor='white',annot = True)
fig, ax = plt.subplots()

fig.set_size_inches(15,20)

pvdf = df.pivot_table(values='NA_Sales',index='Platform',columns='Genre',  fill_value=0)

pf = sns.heatmap(pvdf, cmap = "coolwarm",linecolor='white',annot = True)
fig, ax = plt.subplots()

fig.set_size_inches(15,20)

pvdf = df.pivot_table(values='EU_Sales',index='Platform',columns='Genre',  fill_value=0)

pf = sns.heatmap(pvdf, cmap = "coolwarm",linecolor='white',annot = True)
fig, ax = plt.subplots()

fig.set_size_inches(15,20)

pvdf = df.pivot_table(values='JP_Sales',index='Platform',columns='Genre',  fill_value=0)

pf = sns.heatmap(pvdf, cmap = "coolwarm",linecolor='white',annot = True, vmax=2, vmin=0)
g = sns.FacetGrid(df, col="Genre")

g = g.map(plt.scatter, "Year", "Global_Sales", edgecolor="w")

g = sns.FacetGrid(df, col="Genre")

g = g.map(plt.scatter, "Year", "NA_Sales", edgecolor="w")

g = sns.FacetGrid(df, col="Genre")

g = g.map(plt.scatter, "Year", "EU_Sales", edgecolor="w")

g = sns.FacetGrid(df, col="Genre")

g = g.map(plt.scatter, "Year", "JP_Sales", edgecolor="w")

g = sns.FacetGrid(df, col="Genre")

g = g.map(plt.scatter, "Year", "Other_Sales", edgecolor="w")
#Part 2. Building a regression model for forecasting sales of top 5 genres through the years 
df2 = df[df['Genre'] == 'Action']



df2 = df2.dropna(axis=0, subset=['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])

from sklearn.model_selection import train_test_split

X = df2[['Year']]

y = df2[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
sns.heatmap(df2.corr(), annot=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,y.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# Regression of Adventure Games

df2 = df[df['Genre'] == 'Adventure']



df2 = df2.dropna(axis=0, subset=['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])

sns.heatmap(df2.corr(), annot=True)
X = df2[['Year']]

y = df2[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,y.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
results = ols('JP_Sales ~ Year', data=df2).fit()

results.summary()
aov_table = sm.stats.anova_lm(results, typ=2)

aov_table
#Regression Analysis of Sports

df2 = df[df['Genre'] == 'Sports']



df2 = df2.dropna(axis=0, subset=['Year', 'Publisher', 'Platform', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])



X = df2[['Year']]

y = df2[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
lm = LinearRegression()
sns.heatmap(df2.corr(), annot=True)
lm.fit(X_train,y_train)

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,y.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#Regression Analysis of Role Playing Games

df2 = df[df['Genre'] == 'Role-Playing']



df2 = df2.dropna(axis=0, subset=['Year', 'Publisher', 'Platform', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])



X = df2[['Year']]

y = df2[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
sns.heatmap(df2.corr(), annot=True)
lm = LinearRegression()
lm.fit(X_train,y_train)

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,y.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#Regression Analysis of Misc

df2 = df[df['Genre'] == 'Misc']



df2 = df2.dropna(axis=0, subset=['Year', 'Publisher', 'Platform', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])



X = df2[['Year']]

y = df2[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
sns.heatmap(df2.corr(), annot=True)
lm = LinearRegression()
lm.fit(X_train,y_train)

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,y.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))