#Importing needed packages

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

sns.set()
#Importing the dataset

raw_data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
#Checking for any null values

raw_data.isnull().sum()
#Getting the names of the columns we have

print(raw_data.columns)
#Removing the HDI and country-year columns

no_na_data = raw_data[['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicides/100k pop', ' gdp_for_year ($) ', 'gdp_per_capita ($)', 'generation']]

no_na_data.head()
#Describing our dataset

no_na_data.describe()
#Checking the entries where suicides_no = 0 

no_na_data[no_na_data['suicides_no']==0]
#Creating a new dataframe 'clean_data' to work with 

clean_data = no_na_data.copy()
#Grouping our data by year

gp_year_data = clean_data.groupby('year', as_index=False).mean()



#Plotting the suicides rates by years 

fig, ax = plt.subplots(figsize=(12,4))

sns.lineplot(x='year', y='suicides/100k pop', color=sns.husl_palette(6)[5], data=gp_year_data, ax=ax)

plt.xlabel('Year')

plt.ylabel('Suicides/100k')

plt.title('Evolution of suicide rates\nthroughout the years', size=15)

plt.show()
#Grouping the data by country

gp_cnt_data = clean_data.groupby('country', as_index=False).sum()

top_ten = gp_cnt_data.nlargest(10, 'suicides_no').sort_values('suicides_no', ascending=False)



#Plotting the number of suicides according to the countries 

fig, ax = plt.subplots(figsize=(12,4))

sns.barplot(x='suicides_no', y='country', palette='husl', data=top_ten, ax=ax)

plt.xlabel('Suicides')

plt.ylabel('Country')

plt.title('Suicides according\nto the country', size=15)

ax.ticklabel_format(style='plain', axis='x')

plt.show()
#Grouping the data by generations

gp_gen_data = clean_data.groupby('generation', as_index=False).mean()



#Plotting the suicide rates according to the generations 

fig, ax = plt.subplots(figsize=(12,4))

sns.barplot(x='generation', y='suicides/100k pop', palette='husl', data=gp_gen_data, ax=ax, 

            order=['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials', 'Generation Z'])

plt.xlabel('Generation')

plt.ylabel('Suicides/100k')

plt.title('Suicide rates according\nto the generation', size=15)

plt.show()
#Grouping the data by age

gp_age_data = clean_data.groupby('age', as_index=False).mean()



#Plotting the suicide rates according to the age categories

fig, ax = plt.subplots(figsize=(12,4))

sns.barplot(x='age', y='suicides/100k pop', palette='husl', data=gp_age_data, ax=ax, 

           order=['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'])

plt.xlabel('Age categories')

plt.ylabel('Suicides/100k')

plt.title('Suicide rates according\nto the age categories', size=15)

plt.show()
#Grouping our data by generation and age 

gp_gen_age_data = clean_data.groupby(['generation', 'age'], as_index=False).mean()



#Making a list containing all the gens 

gens = ['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials', 'Generation Z']



#Creating the axis of the plots

plt.figure(figsize=(12,18))

ax1 = plt.subplot2grid((6,1),(0,0))

ax2 = plt.subplot2grid((6,1),(1,0))

ax3 = plt.subplot2grid((6,1),(2,0))

ax4 = plt.subplot2grid((6,1),(3,0))

ax5 = plt.subplot2grid((6,1),(4,0))

ax6 = plt.subplot2grid((6,1),(5,0))



#Making a list containing all the axes

axes = [ax1, ax2, ax3, ax4, ax5, ax6]



#Making a for loop to plot the needed plots 

for gen, ax in zip(gens, axes):

    sns.barplot(x='age', y='suicides/100k pop', palette='husl', 

                data=gp_gen_age_data[gp_gen_age_data['generation'] == gen],

                ax=ax, order=['5-14 years', '15-24 years', '25-34 years', '35-54 years', 

                          '55-74 years', '75+ years'])

    ax.set_xlabel('Age categories')

    ax.set_ylabel('Suicides/100k')

    ax.set_title(gen, size=15)

plt.tight_layout()
#Grouping the data by country

gp_cnt_data = clean_data.groupby('country', as_index=False).mean()



#Plotting the suicide rates according to the generations 

fig, ax = plt.subplots(figsize=(12,4))

sns.scatterplot(x='gdp_per_capita ($)', y='suicides/100k pop', color=sns.husl_palette(6)[4], data=gp_cnt_data, ax=ax)

plt.xlabel('GDP per capita')

plt.ylabel('Suicides/100k')

plt.title('Suicide rates according\nto the GDP per capita', size=15)

plt.show()
#Making bins and labels for the gdp_per_capita feature

bins = list(range(0, 160000, 20000))

labels = ['0-20,000', '20,000-40,000', '40,000-60,000', '60,000-80,000', '80,000-100,000', '100,000-120,000', '+120,000']

clean_data['gdp_per_capita_bins'] = pd.cut(clean_data['gdp_per_capita ($)'], bins=bins, labels=labels)



#Plotting the suicide rates according to the gbp per capita bins 

fig, ax = plt.subplots(figsize=(12,4))

sns.barplot(x='gdp_per_capita_bins', y='suicides/100k pop', palette='husl', data=clean_data, ax=ax)

plt.xlabel('GDP per capita')

plt.ylabel('Suicides/100k')

plt.title('Suicide rates according\nto the GDP per capita', size=15)

plt.show()
#Selecting the dependent and independent features

X = clean_data[['country', 'sex', 'population', 'age', 'gdp_per_capita ($)', 'generation']]

y = clean_data['suicides/100k pop']
#Transforming the categorical variables to dummy variables

X = pd.get_dummies(X, drop_first=True)
#Importing needed package for scaling

from sklearn.preprocessing import StandardScaler 
#Scaling our data 

sc = StandardScaler()

X[['population', 'gdp_per_capita ($)']] = sc.fit_transform(X[['population', 'gdp_per_capita ($)']])
#Importing needed package for splitting the dataset

from sklearn.model_selection import train_test_split
#Splitting the dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#Importing the Linear Regression algorithm 

from sklearn.linear_model import LinearRegression
#Initializing our Linear Regression

lr = LinearRegression()

lr.fit(X_train, y_train)
#Predicting the test values

lr_y_pred = lr.predict(X_test)
#Plotting the results

fig, ax = plt.subplots(figsize=(12,4))

sns.scatterplot(lr_y_pred, y_test, ax=ax, color=sns.husl_palette(10)[0])

sns.lineplot([0, 175], [0, 175], color=sns.husl_palette(10)[5], ax=ax)

plt.xlabel('Actual values')

plt.ylabel('Predicted values')

plt.title('Prediction evaluation (Linear Regression)', size=15)

plt.show()
#Importing the Decision Tree algorithm 

from sklearn.tree import DecisionTreeRegressor
#Initializing our Decision Tree

dt = DecisionTreeRegressor()

dt.fit(X_train, y_train)
#Predicting the test values

dt_y_pred = dt.predict(X_test)
#Plotting the results

fig, ax = plt.subplots(figsize=(12,4))

sns.scatterplot(dt_y_pred, y_test, ax=ax, color=sns.husl_palette(10)[0])

sns.lineplot([0, 175], [0, 175], color=sns.husl_palette(10)[5], ax=ax)

plt.xlabel('Actual values')

plt.ylabel('Predicted values')

plt.title('Prediction evaluation (Decision Tree)', size=15)

plt.show()