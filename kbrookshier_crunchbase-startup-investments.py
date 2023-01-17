# INTRODUCTION TO MINI PROJECT 2

# I have a public dataset from Crunchbase I will be analyzing.



# Crunchbase tracks startup investment funding and other corporate developments. This large dataset from the company covers 

# fundings round from seed investments through multi-series rounds.
# # RESEARCH QUESTIONS

# 1. How has funding changed over time?

# 2. Given a venture round and a date, what is the likeliest founding outcome?

# 3. How is funding clustered? (E.g. by region, type of company)   <--- Note: exploratory. Results are incomprehensible.

#                                                                  <--- Wanted to include, because (1) I explored this option

#                                                                  <--- and (2) I learned a ton.
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
# Import libraries

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import pandas as pd
# Set plotting

font = {'family' : 'normal',

        'size'   : 18}

plt.rc('font', **font)
# PART 0: DATA CLEANING
# Get data on companies

crunchbase = pd.read_csv("../input/crunchbase-data/crunchbase.csv")

crunchbase.head()
# Get data on funding rounds

rounds = pd.read_csv("../input/crunchbase-rounds/rounds.csv")

rounds.head()
# Check out columns in funding rounds

rounds.columns
# Remove unnecessary columns

rounds = rounds.drop(['company_permalink','company_category_list', 'funding_round_permalink', 

                     'funded_month', ' funded_quarter ','Unnamed: 16'], axis=1)



# Rename columns

# df.rename(columns={"A": "a", "B": "c"})

rounds = rounds.rename(columns=

    {'company_name': 'company'

     ,'company_market': 'market'

     ,'company_country_code':'country'

     ,'company_state_code':'state'

     ,'company_region':'region'

     ,'company_city':'city'

     , ' raised_amount_usd ': 'raised'

     , 'funded_at': 'date'})



# Drop lines without key data

rounds = rounds.dropna(subset=['funded_year', 'raised'])

rounds.shape



# Change string numbers into integers

rounds['funded_year'] = rounds['funded_year'].astype(int)

rounds = rounds[rounds.funded_year > 1999]



# Convert date from string to datetime

rounds['date'] = pd.to_datetime(rounds['date'])



# Change numbers into integers



# Remove whitespace

rounds['raised'] = rounds['raised'].str.strip()



# Remove commas

rounds['raised'] = rounds['raised'].str.replace(',', '')



# Replace null funding with zero

rounds['raised'] = rounds['raised'].str.replace('-', '0')



# Change to integers

rounds['raised'] = rounds['raised'].astype(int)
# Check largest rounds

rounds.sort_values(by=['raised'], ascending=False).head(10)
# RESEARCH QUESTION #1 How has funding changed over time?
# Look at funding types in dataset

rounds.groupby(['funding_round_type']).size().sort_values(ascending=False)



# The most common funding types are venture capital and seed financing.

# Based on this, the following analysis will only look at venture and seed data.
# Get an array for numpy

year = rounds[['funded_year']].values.reshape(-1,1)

raised = rounds[['raised']].values.reshape(-1,1)
# Venture capital

venture = rounds.loc[rounds['funding_round_type']=='venture']



# Get an array for numpy

year_v = venture[['funded_year']].values.reshape(-1,1)

raised_v = venture[['raised']].values.reshape(-1,1)



# Get linear regression

model = LinearRegression()

model.fit(year_v,raised_v)

pred_venture = model.predict(year_v)
# Plot venture capital data

plt.figure(figsize=(14,10))

plt.grid()



plt.scatter(year_v, raised_v, s=100, alpha=0.3)

plt.title('Venture capital raised by companies has increased over time')

plt.xlabel('Year')

plt.ylabel('Money')



# Plot the linear regression prediction

plt.plot(year_v, pred_venture, color = 'blue')
# Based on the model, funding should rise about $97,000 per year.

model.coef_
# An average company raising venture capital in 2020 should expect funding of $10.4 million.

money = 2020 * model.coef_ + model.intercept_

money
# Seed financing

seed = rounds.loc[rounds['funding_round_type']=='seed']



# Get an array for numpy

year_s = seed[['funded_year']].values.reshape(-1,1)

raised_s = seed[['raised']].values.reshape(-1,1)



# Get linear regression

model_seed = LinearRegression()

model_seed.fit(year_s,raised_s)

pred_seed = model_seed.predict(year_s)
# Plot the data

plt.figure(figsize=(14,10))

plt.grid()



plt.scatter(year_s, raised_s, s=100, alpha=0.3)

plt.title('Seed money raised by companies has decreased over time')

plt.xlabel('Year')

plt.ylabel('Money')



# Plot the linear regression prediction

plt.plot(year_s, pred_seed, color = 'red')
# Based on the model, funding should go down about $25,000 per year.

model_seed.coef_
# An average company raising a seed financing round in 2020 should expect funding of $450,000.

money_seed = 2020 * model_seed.coef_ + model_seed.intercept_

money_seed
# Plot the data

plt.figure(figsize=(14,10))

plt.grid()



plt.title('Venture capital is rising, while seed financing is falling')

plt.xlabel('Year')

plt.ylabel('Money')





# Plot the linear regression prediction

plt.plot(year_s, pred_seed, color = 'red')

plt.plot(year_v, pred_venture, color = 'blue')



# Ventere capital is rising over time, whereas seed financing is falling.
# RESEARCH QUESTION #2 Given a venture round and a date, what is the likeliest funding outcome?
# Get relevant data

funding = venture[['funded_year', 'raised','funding_round_code']]

funding = funding.dropna()

funding
funding_A = funding.loc[funding['funding_round_code']=='A']

funding_B = funding.loc[funding['funding_round_code']=='B']

funding_C = funding.loc[funding['funding_round_code']=='C']

funding_D = funding.loc[funding['funding_round_code']=='D']

funding_E = funding.loc[funding['funding_round_code']=='E']

funding_F = funding.loc[funding['funding_round_code']=='F']
# Plot the data

plt.figure(figsize=(14,10))

plt.grid()



plt.scatter(funding_A['funded_year'], funding_A['raised'], s=100, alpha=0.1, color='red')

plt.scatter(funding_B['funded_year'], funding_B['raised'], s=100, alpha=0.1, color='orange')

plt.scatter(funding_C['funded_year'], funding_C['raised'], s=100, alpha=0.1, color='yellow')

plt.title('Seed money')

plt.xlabel('Year')

plt.ylabel('Money')
# Graph the different funding rounds by frequency

funding.groupby(['funding_round_code']).size().sort_values(ascending=False).head(20).plot(kind='bar', figsize=(20,8))
# Replace funding round with an integer

funding = funding.replace({'A': 1, 'B': 2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8})

funding
X = funding[['funded_year','funding_round_code']]

Y = funding['raised']
# With sklearn

# regr = linear_model.LinearRegression()

regr = LinearRegression()

regr.fit(X, Y)



print('Intercept:', regr.intercept_)

print('Coefficients:', regr.coef_)
# Prediction for funding

print ('Predicted funding received for A round in 2018:', regr.predict([[2018,1]]))

print ('Predicted funding received for B round in 2018:', regr.predict([[2018,2]]))

print ('Predicted funding received for C round in 2018:', regr.predict([[2018,3]]))

print ('Predicted funding received for D round in 2018:', regr.predict([[2018,4]]))

print ('Predicted funding received for E round in 2018:', regr.predict([[2018,5]]))

print ()

print ('Predicted funding received for A round in 2020:', regr.predict([[2020,1]]))

print ('Predicted funding received for B round in 2020:', regr.predict([[2020,2]]))

print ('Predicted funding received for C round in 2020:', regr.predict([[2020,3]]))

print ('Predicted funding received for D round in 2020:', regr.predict([[2019,4]]))

print ('Predicted funding received for E round in 2020:', regr.predict([[2020,5]]))



# Given a year and round, the round code (A-H) has the largest impact on the amont of money raised.



# For example, in 2020 the expected funding for a series A round was $11.2 million

# The expected funding for a series B round was $18.2 million



# By contrast, in 2018 the expected funding for a series A round was $10.2 million

# The expected funding for a series B round was $17.2 million
# RESEARCH QUESTION #3 How is funding clustered? (E.g. by region, type of company)



# PLEASE NOTE

# The data I was using was categorical data

# I tried out Kmeans clustering, which produced an output (but not an output I can make sense of)

# I also tried a linear regression, using status (dead, operating, acquired) as the predicted value. Something happened, 

# and I also can't make sense of it



# So, you can read this. But this section was more about me failing than succeeding.



# Learnings:

# Numerical data is WAY easier to work with than nominal data

# I have a lot more to learn, but this has still been my first experience with modeling, and it rocked!
# Import clustering

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model
# Get relevant data

df = crunchbase[[' market ','state_code','status']]

df = df.rename(columns={' market ': 'market', 'state_code':'state'})

df = df.dropna()

df.head(10)
# Check out most common markets

df.groupby(['market']).size().sort_values(ascending=False).head()
# Get rid of market that have only 1 entry

df = df[df.market.duplicated(keep=False)]
# Prep data for clustering



# Make columns into categories

df['market'] = df['market'].astype('category')

df['state'] = df['state'].astype('category')

df['status'] = df['status'].astype('category')



# Transform string data into numerical category data

cat_columns = df.select_dtypes(['category']).columns

df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
# Check out kmeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(df) 

kmeans.cluster_centers_
# Make a prediction

kmeans.predict([[123, 1, 25]])
# Predict outcome (company died, still operating, acquired) given market type and state

y = df[['status']]

df = df[['market','state']]
# Create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
# Fit model

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
# Make predictions!

predictions[0:5]
# The line / model

plt.scatter(y_test, predictions)

plt.xlabel('Values')

plt.ylabel('Predictions')



# PLEASE NOTE: yes, this is plotted.

# I have no idea what it means
# How good is the model?

print('Score:', model.score(X_test, y_test))



# The model is *terrible*! 
# # CONCLUSIONS



# 1. How has funding changed over time?

# Venture capital funding rounds have increased over time (about $97,000 per year); an average company raising 

# venture capital in 2020 should expect funding of $10.4 million. Seed financing has dropped (about $25,000 per year);

# an average company raising seed financing in 2020 should expect funding of $450,000.



# 2. Given a venture round and a date, what is the likeliest founding outcome?

# Given a year and round, the round code (A-H) has the largest impact on the amont of money raised.

# For example, in 2020 the expected funding for a series A round was $11.2 million

# The expected funding for a series B round was $18.2 million

# By contrast, in 2018 the expected funding for a series A round was $10.2 million

# The expected funding for a series B round was $17.2 million



# 3. How is funding clustered? (E.g. by region, type of company)

# I had results, but they were incomprehensible (and, by extension, inconclusive). I included these, because 

# (A) I explored this option, (2) I learned a ton in the process wrangling this data, and (3) show you where

# my current limitations are. Despite this, I have a clear area where I would like to progress in the future!