# Installing basic packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import matplotlib.ticker as ticker  #to change the display of units used in graphs



%matplotlib inline



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))
# uploading both the 'train' and 'test' data files into dataframes



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
print('train_df shape:', train_df.shape)

print('test_df shape:', test_df.shape)
# Train data: investigating all columns and missing values from each



print(train_df.info(verbose=True))
# Test data: investigating all columns and missing values from each



print(test_df.info(verbose=True))
train_df.describe(include='all')
# setting the 'id' as index

train_df.set_index('id', inplace=True)

test_df.set_index('id', inplace=True)



# changing types a float for budget and revenue

train_df = train_df.astype({"budget":'float64', "revenue":'float64'})

test_df = test_df.astype({"budget":'float64'})



# adjusting display of large numbers using commas and showing only 2 decimal points

pd.set_option('display.float_format', '{:,.2f}'.format)
# check if it worked

print('Train data')

print(train_df.describe())

print("")

print('Test data')

print(test_df.describe())
sns.pairplot(train_df)
#train_df['belongs_to_collection'].apply(lambda x: len(str(x)))

for x in train_df['belongs_to_collection'].apply(lambda x: len(str(x))):

    if x > 3:

        train_df['in_collection'] = 1

    else: 

        train_df['in_collection'] = 0



train_df['in_collection']



#train_df['belongs_to_collection'] = x = 1 if len(str(x))>3 else x = 0 for train_df['belongs_to_collection']
train_df['in_collection']
budget_ax = train_df['budget'].plot(kind='hist', bins= 100, figsize=(15,3))

budget_ax.set_title('Budget histogram')

budget_ax.xaxis.set_major_formatter(ticker.EngFormatter())
# Counting how many zero values there are per column

(train_df == 0).sum(axis=0)
# analysing the rows with budget value > 0

train_df[train_df.budget>0].describe()
# analysing the rows where: [ 0 < budget value < 1,000 ]

train_df[(train_df.budget<1000) & (train_df.budget>0)].describe()
train_df[(train_df.budget>1000)].describe()
budget2_ax = train_df.budget[train_df.budget>1000].plot(kind='hist', bins= 100, figsize=(15,3))

budget2_ax.set_title('Histogram: Movies with budget > $1,000')

budget2_ax.xaxis.set_major_formatter(ticker.EngFormatter())
# Visualise using a box plot.

train_df.budget.plot(kind='box');
# Counting how many zero values there are per column in the test_df

(test_df == 0).sum(axis=0)
test_df.describe()
budget3_ax = test_df.budget[test_df.budget>1000].plot(kind='hist', bins= 100, figsize=(15,3))

budget3_ax.set_title('Histogram: Test Movies with budget > $1,000')

budget3_ax.xaxis.set_major_formatter(ticker.EngFormatter())
test_df[(test_df.budget>1000)].describe()
# Analysing the relationship between 'budget' and 'revenue' 



fig, axe = plt.subplots(figsize=(8, 5))    

cool_chart = sns.scatterplot(ax=axe, data=train_df, x='budget',y='revenue',marker='o',

                             s=100,palette="magma", alpha=0.3)



cool_chart.xaxis.set_major_formatter(ticker.EngFormatter())

cool_chart.yaxis.set_major_formatter(ticker.EngFormatter())

plt.show()


fig, axe = plt.subplots(figsize=(8, 5))    

cool_chart = sns.scatterplot(ax=axe,data=train_df[train_df.budget >0],

                             x='budget', y='revenue', marker='o', s=100, palette="magma", alpha=0.3)



# zooming-in but setting limits on x=axis up to 100M and y-axis up to 200M

axe.set(xlim = (0,100000000), ylim = (0, 200000000))



cool_chart.xaxis.set_major_formatter(ticker.EngFormatter())

cool_chart.yaxis.set_major_formatter(ticker.EngFormatter())

plt.show()


fig, axe = plt.subplots(figsize=(8, 5))    

cool_chart = sns.scatterplot(ax=axe, data=train_df[train_df.budget >0],x='budget',y='revenue',

                             marker='o',alpha=0.3,s=100,palette="magma")



# zooming-in but setting limits on x=axis up to 100M and y-axis up to 200M

axe.set(xlim = (0,40000000), ylim = (0, 200000000))



cool_chart.xaxis.set_major_formatter(ticker.EngFormatter())

cool_chart.yaxis.set_major_formatter(ticker.EngFormatter())

plt.show()
train_df.release_date.head()
# creating a date function so we can apply it to the train and test datasets



def date_features(df):

# Changing the date column to an apropriate pandas date format

    df['release_date'] = pd.to_datetime(df['release_date'])



# Creating the new columns

    df['release_year'] = df['release_date'].dt.year

    df['release_month'] = df['release_date'].dt.month

    df['release_day'] = df['release_date'].dt.day

    df['release_quarter'] = df['release_date'].dt.quarter

    df['release_dayofweek'] = df['release_date'].dt.dayofweek   # where Mon=0 and Sun=6



    

    

    df.loc[df['release_year'] > 2019,'release_year']=df.loc[df['release_year']>2019,'release_year'].apply(lambda x: x - 100)

    

    

    return df
# applying the date function

train_df = date_features(train_df)



# checking if it worked

train_df.sort_values(by='release_year',ascending=False).head(10)[['title','revenue','release_year', 'release_quarter','release_month','release_day','release_dayofweek']]
train_df['release_year'].plot(kind='hist', bins= 100, figsize=(10,3))
train_df['release_year'].value_counts().sort_values()
sns.jointplot(x="release_year", y="revenue", data=train_df, color="g", alpha=0.3);

sns.jointplot(x="release_quarter", y="revenue", data=train_df, height=7, ratio=4, color="gray", alpha=0.3)
sns.jointplot(x="release_month", y="revenue", data=train_df, height=7, ratio=4, color="r", alpha=0.3)
# Visualise using a box plot.

train_df.boxplot(column='revenue', by='release_month', figsize=(9, 6));
sns.jointplot(x="release_dayofweek", y="revenue", data=train_df, height=7, ratio=4, color="b", alpha=0.3)
# Visualise using a box plot.

train_df.boxplot(column='revenue', by='release_dayofweek', figsize=(9, 6));
#runtime analysis vs revenue



fig, ax = plt.subplots(figsize=(8, 5))    

palette = sns.color_palette("bright", 6)

cool_chart = sns.scatterplot(ax=ax, data=train_df, x='runtime', y='revenue', marker='o', s=100, palette="magma")

cool_chart.legend(bbox_to_anchor=(1, 1), ncol=1)

#ax.set(xlim = (50000,250000))



cool_chart.xaxis.set_major_formatter(ticker.EngFormatter())

cool_chart.yaxis.set_major_formatter(ticker.EngFormatter())

plt.show()
train_df['status'].value_counts()
test_df['status'].value_counts()
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics
# separating the features and getting them ready:

X = train_df.drop(['revenue'])

# separating the variable to predict and getting it ready:

y = train_df['revenue']



# Creating subsets from the data for model training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Instantiating a regressor

regressor = LinearRegression()  



# Fitting the model with the train data

regressor.fit(X_train, y_train)



# Making a prediction



# Evaluating the score of the prediction


