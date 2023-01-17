#import packages needed

#bring in data and clean



import statsmodels.formula.api as smf

import seaborn as sns

import pandas as pd

import numpy as np

import seaborn

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import linear_model, model_selection, metrics





data = pd.read_csv('../input/diamonds.csv')

df = data.loc[data['carat'] <= 3]

df.set_index('Unnamed: 0')

df.head()

#Visualize correlation of numerical values



plt.figure(figsize=(10, 10))

corr = df.corr()

sns.heatmap(corr*100, cmap="YlGn", annot= True, fmt=".0f")
plt.figure(figsize=(16, 6))

plt.scatter(df['carat'],df['price'])



z = np.polyfit(df['carat'], df['price'], 1)

p = np.poly1d(z)

plt.plot(df['carat'],p(df['carat']),"r--")
#Visualize potential correlation of categorical values

#Clarity

plt.figure(figsize=(16, 10))

#a = sns.scatterplot( x='carat', y='price', data=df, fit_reg=False, hue='clarity', legend=False)

a = sns.scatterplot( x='carat', y='price', data=df, hue='clarity',edgecolor='None',palette='ocean')

a.legend()

#plt.legend(loc='lower right')
#Color

plt.figure(figsize=(16, 10))

#a = sns.scatterplot( x='carat', y='price', data=df, fit_reg=False, hue='clarity', legend=False)

a = sns.scatterplot( x='carat', y='price', data=df, hue='color',edgecolor='None')

a.legend()
#Cut

plt.figure(figsize=(16, 10))

#a = sns.scatterplot( x='carat', y='price', data=df, fit_reg=False, hue='clarity', legend=False)

a = sns.scatterplot( x='carat', y='price', data=df, hue='cut',edgecolor='None')

a.legend()
#Split data into train and test data sets

train, test = model_selection.train_test_split(df)
#Create regression models

all_reg = smf.ols("price ~ carat + clarity + color + cut", data=df).fit()

train_reg = smf.ols("price ~ carat + clarity + color + cut", data=train).fit()

test_reg = smf.ols("price ~ carat + clarity + color + cut", data=test).fit()
#Plot and compare actual vs predicted price of diamonds from training set

plt.figure(figsize=(16, 10))

plt.ylim([0,train['price'].max()])



plt.scatter(train['carat'], train['price'])

plt.scatter(train['carat'], train_reg.predict(train),color='#00CC00',label='Train Data')



z = np.polyfit(train['carat'], train['price'], 1)

p = np.poly1d(z)

plt.plot(train['carat'],p(train['carat']),"r--")
#Analyze summary of training model

train_reg.summary()
#Plot and compare actual vs predicted price of diamonds from testing set

plt.figure(figsize=(16, 10))

plt.ylim([0,train['price'].max()])



plt.scatter(test['carat'], test['price'])

plt.scatter(test['carat'], test_reg.predict(test),color='#00CC00',label='Train Data')



z = np.polyfit(test['carat'], test['price'], 1)

p = np.poly1d(z)

plt.plot(test['carat'],p(test['carat']),"r--")
#Analyze summary of test model

test_reg.summary()
#Is my $8k diamond over priced?

new_df = pd.DataFrame(columns=['carat', 'cut', 'color', 'clarity'], index=[0])

new_df.loc[0] = pd.Series({'carat': 1, 'cut': 'Good', 'color': 'J', 'clarity': 'SI1'})

new_df['carat'] = new_df['carat'].astype('float')

all_reg.predict(new_df)

plt.figure(figsize=(16, 10))

plt.ylim([0,df['price'].max()])



plt.scatter(df['carat'], df['price'])

plt.scatter(df['carat'], all_reg.predict(df),color='#00CC00',label='Data')



z = np.polyfit(df['carat'], df['price'], 1)

p = np.poly1d(z)

plt.plot(df['carat'],p(df['carat']),"r--")

plt.scatter(new_df['carat'],all_reg.predict(new_df),s=200)