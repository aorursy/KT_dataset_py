# importing necessary libraries

import numpy as np

import pandas as pd

import os

import sklearn

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt



# reading the required files

tr_trns = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

tr_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

tst_trns = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

tst_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
# We'll need to join on the TransactionID field to get a single source of data

tr_trns['TransactionID'].head()

tr_id['TransactionID'].head()





# Limiting the number of columns for the first 5 questions of the exercise

list(tr_trns.columns)

cols1 = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD',

         'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2',

         'dist1', 'dist2', 'isFraud']

trns_sub = tr_trns[cols1]





list(tr_id.columns)

cols2 = ['TransactionID', 'DeviceType', 'DeviceInfo']

id_sub = tr_id[cols2]





# Now joining the two tables

trns_join = trns_sub.merge(id_sub, on = 'TransactionID', how = 'left')

trns_join.dtypes

desc = trns_join.describe()

trns_join.isna().sum()          # How many NAs in each feature
# 1. Plotting the distribution of some features

# First separating the dataset into fradulent and nonfradulent

fraud_dat = trns_join[trns_join['isFraud'] == 1]

nonfraud_dat = trns_join[trns_join['isFraud'] == 0]



# shape of both datasets

print(fraud_dat.shape)

print(nonfraud_dat.shape)
## distribution of some features in fraud_dat

# a. TransactionDT

print(fraud_dat['TransactionDT'].describe())

plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)

plt.hist(fraud_dat['TransactionDT'])

plt.title('TransactionDT in fraudulent data')



# a.2.

plt.subplot(1, 2, 2)

plt.hist(nonfraud_dat['TransactionDT'])

plt.title('TransactionDT in the non-fraudulent data')



# Finding the minimum and maximum time

a = np.min(trns_join['TransactionDT'])

b = np.max(trns_join['TransactionDT'])

(b-a)/86400

# So, the transactions have been recorded over a span of about 182 days
# b. TransactionAmt

print(fraud_dat['TransactionAmt'].describe())

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)

plt.hist(fraud_dat['TransactionAmt'], color='orange')

plt.title('Distribution of TransactionAmt in Fraudulent data')

print(fraud_dat['TransactionAmt'].quantile(0.9))

len(fraud_dat[fraud_dat['TransactionAmt'] < 1000])/len(fraud_dat['TransactionAmt'])



# Plotting for less than $500 amount

tmp = fraud_dat[fraud_dat['TransactionAmt'] < 500]

plt.subplot(1, 4, 2)

plt.hist(tmp['TransactionAmt'], color='orange')

plt.title('Distribution for TransactionAmt < $500')

# We can also check the distribution of the fraudulent transaction amount

# during the different time frames.

plt.subplot(1, 4, 3)

plt.scatter(tmp['TransactionAmt'], tmp['TransactionDT'], color='orange')

plt.title('TransactionAmt vs Transaction Time')



# b.2. 

plt.subplot(1, 4, 4)

plt.hist(nonfraud_dat['TransactionAmt'], color='orange')

plt.title('Distribution of TransactionAmt in non-Fraudulent data')

print(nonfraud_dat['TransactionAmt'].quantile(0.9))
# c. ProductCD (a categorical variable)

print(fraud_dat['ProductCD'].describe())

plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)

fraud_dat['ProductCD'].value_counts().plot(kind='bar', color = 'purple')

plt.title('ProductCD distribution in fraudulent')



# c.2.

plt.subplot(1, 2, 2)

nonfraud_dat['ProductCD'].value_counts().plot(kind='bar', color = 'purple')

plt.title('ProductCD distribution in non-fraudulent')
# d. card4 and card6 are two more categorical variables

plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)

fraud_dat['card4'].value_counts().plot(kind='bar')

plt.title('card4 distribution in fraudulent')



plt.subplot(2, 2, 2)

nonfraud_dat['card4'].value_counts().plot(kind='bar')

plt.title('card4 distribution in non-fraudulent')



# card6 plots

plt.subplot(2, 2, 3)

fraud_dat['card6'].value_counts().plot(kind='bar')

plt.title('card6 distribution in fraudulent')



plt.subplot(2, 2, 4)

nonfraud_dat['card6'].value_counts().plot(kind='bar')

plt.title('card6 distribution in non-fraudulent')
# e. P_emaildomain and R_emaildomain (more categorical variables)

plt.figure(figsize=(18, 15))

plt.subplot(2, 2, 1)

fraud_dat['P_emaildomain'].value_counts().plot(kind='bar')

plt.title('Purchaser emaildomain in fraudulent')

# a number of email domains have been used for frauds, however, the most

# popular ones stands out here as well. Frauds committed using gmail domain

# are more in number than all the other domains combined.

plt.subplot(2, 2, 2)

nonfraud_dat['P_emaildomain'].value_counts().plot(kind='bar')

plt.title('Purchaser emaildomain in non-fraudulent')



# R_emaildomain

plt.subplot(2, 2, 3)

fraud_dat['R_emaildomain'].value_counts().plot(kind='bar')

plt.title('Recipient emaildomain in fraudulent')



plt.subplot(2, 2, 4)

nonfraud_dat['R_emaildomain'].value_counts().plot(kind='bar')

plt.title('Recipient emaildomain in non-fraudulent')



# analyzing the email domains of the purchaser and recipient email domain

a = set(fraud_dat['P_emaildomain'])

b = set(fraud_dat['R_emaildomain'])

c = a.difference(b)

print(c)

d = b.difference(a)

print(d)

# so, the recipient email domain is actually a subset of the purchaser email

# domain. There is no secret domain that the recipient uses but not being

# used by the purchasers. 
# f. addr1 and addr2 - both are address codes and hence categorical (i.e.

# mathematical operations shouldn't be performed on them)

plt.figure(figsize=(18, 15))

plt.subplot(2, 2, 1)

fraud_dat['addr1'].value_counts().plot(kind='pie')

plt.title('addr1 distribution in fraudulent')

# Using the pie chart, we are able to easily see the most frequently 

# occurring region codes

plt.subplot(2, 2, 2)

fraud_dat['addr2'].value_counts().plot(kind='pie')

plt.title('addr2 distribution in fraudulent')

# the country code most frequently appearing is '87', in fact almost all

# fraudulent transactions are from that country. Can contrast with the 

# whole dataset, if '87' still has the highest number of transactions by 

# same proportion

plt.subplot(2, 2, 3)

nonfraud_dat['addr1'].value_counts().plot(kind='pie')

plt.title('addr1 distribution in non-fraudulent')

plt.subplot(2, 2, 4)

nonfraud_dat['addr2'].value_counts().plot(kind='pie')

plt.title('addr2 distribution in non-fraudulent')

# The distribution is similar to nonfraudulent transactions, just a few 

# addresses have shifted positions which became more prominent in fraudulent

# transactions
# g. dist1 and dist2 - both are continuous, however contain a large number

# of missing values. Could be missing due to privacy or other legal reasons

# we can still plot the rest of the 'good' records and see distribution

plt.figure(figsize=(18, 10))

plt.subplot(2, 2, 1)

plt.hist(fraud_dat['dist1'].dropna(), alpha=0.5)

plt.title('dist1 distribution in fraudulent')

# Again a skewed distribution meaning most 'good' records lie within the 

# range of (0, 1000) of dist1

plt.subplot(2, 2, 2)

plt.hist(nonfraud_dat['dist1'].dropna(), alpha=0.5)

plt.title('dist1 distribution in non-fraudulent')

# The distribution is almost identical

plt.subplot(2, 2, 3)

plt.hist(fraud_dat['dist2'].dropna(), alpha=0.5, color='red')

plt.title('dist2 distribution in fraudulent')

plt.subplot(2, 2, 4)

plt.hist(nonfraud_dat['dist2'].dropna(), alpha=0.5, color='red')

plt.title('dist2 distribution in non-fraudulent')

# both look the same as 'dist1'
# h. DeviceType - a potentially useful categorical variable

plt.figure(figsize=(18, 5))

plt.subplot(1, 2, 1)

print(fraud_dat['DeviceType'].value_counts())

fraud_dat['DeviceType'].value_counts().plot(kind='bar', color='yellow')

plt.title('DeviceType distribution for fraudulent')

# both the deviceType users are fairly equally distributed

plt.subplot(1, 2, 2)

nonfraud_dat['DeviceType'].value_counts().plot(kind='bar')

plt.title('DeviceType distribution for non-fraudulent')

# For nonfraudulent, more desktop devices are used as compared to mobile
# i. DeviceInfo - another potentially useful categorical variable

print(fraud_dat['DeviceInfo'].value_counts())

# There are a lot of categories that are hard to visualize, still I would

# take an attempt

plt.figure(figsize=(18, 5))

plt.subplot(1, 2, 1)

fraud_dat['DeviceInfo'].value_counts().plot(kind='pie')

plt.title('DeviceInfo distribution in fraudulent')

# Although a mess, it's clear that Windows and iOS platforms have the largest

# share, taking more than 50% of total records

plt.subplot(1, 2, 2)

nonfraud_dat['DeviceInfo'].value_counts().plot(kind='pie')

plt.title('DeviceInfo distribution in non-fraudulent')

# not much difference between both distribution
# 2. Analyzing the frequency distribution of transactions by time for the 

# most frequent code

# first the most frequent country code



df = trns_join

df['addr2'].value_counts()

# 87 country code has the highest number of transactions

#df = df.dropna(subset = ['addr2'])

#df['addr2'] = df['addr2'].astype(int)

df = df[df['addr2'] == 87]

# now dataframe is just composed of transactions from that country

# we can take a look at the distribution of TransactionDT feature

df['TransactionDT'].head(10)

# we find the time of the place by dividing with 86400

df['Time'] = df['TransactionDT']%86400

df['Time'].head()

# now dividing that with 60*60 or 3600 to get in hours

df['Time'] = df['Time']/3600

plt.figure(figsize=(18, 5))

plt.subplot(1, 2, 1)

plt.hist(df['Time'], color = 'green', alpha = 0.7)

plt.title('Distribution of Time variable')



# We can go one step further to see if the fraudulent transactions also

# follow the same pattern

plt.subplot(1, 2, 2)

plt.hist(df['Time'][df['isFraud'] == 1], color = 'teal', alpha = 0.7)

plt.title('Distribution of Time variable for fraudulent transactions')

# It also follows the same pattern
# 3. Analyze the ProductCD variable



df = trns_join

df.columns

df['ProductCD'].value_counts()

df = df[['ProductCD', 'TransactionAmt']]

df.boxplot(by='ProductCD')

# Here, we can see the Product code W has a couple of points that have very

# high transaction amount value. Those products are the most expensive, which

# belong to the product code W. However, most of the products with the code

# W are still far lower in the amount. We can try to create the boxplot again

# without the obvious outliers

df[df['TransactionAmt'] < 25000].boxplot(by='ProductCD')

# It again gives us the same picture that there are a lot of outliers 

# especially in product code W. We may rather take mean/median of transaction

# amount for each product code and compare. 

# Actually median could be preferred as it's not impacted by the outliers

# but it may not capture the whole idea of most or least expensive products

df.groupby('ProductCD').agg('mean')

# Using mean, we see that products with code 'R' and 'W' have the most 

# expensive products. While code 'C' and 'S' have the cheapest products

df.groupby('ProductCD').agg('median')

# Using median, it's obvious that the 'R' has the most expensive products

# while 'C' products are the cheapest
# 4. Plot time of day vs transaction amount

df = trns_join

df['Time'] = df['TransactionDT']%86400

df['Time'].head()

# now dividing that with 60*60 or 3600 to get in hours

df['Time'] = df['Time']/3600

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

plt.scatter(df['Time'], df['TransactionAmt'])

plt.title('Time vs TransactionAmt')

# There is a clear outlier - to view the distribution clearly, we will remove

# the outlier and replot

df = df[df['TransactionAmt'] < 25000]

plt.subplot(1, 2, 2)

plt.scatter(df['Time'], df['TransactionAmt'])

plt.title('With outlier removed')

z = np.polyfit(df['Time'], df['TransactionAmt'], 1)

p = np.poly1d(z)

plt.plot(df['Time'], p(df['Time']), 'm-')

plt.title('With the trend line')

plt.show()

# Not an interesting correlation can be observed here. The trendline is

# also quite constant. Let's find out the correlation coefficients

df['Time'].corr(df['TransactionAmt'], method = 'pearson')

# 0.04538

df['Time'].corr(df['TransactionAmt'], method = 'spearman')

# 0.03808

# The above numbers imply a very weak correlation between the two variables
# 5. Any interesting plot

# Let's identify the top 5 regions with highest proportion of frauds in 

# the country with most frequent transactions ('87')



df = df[df['addr2'] == 87]

df['addr1'].value_counts()

df1 = df[['addr1', 'isFraud']]

x = df1.groupby(['addr1']).agg('count')

x['addr1'] = x.index

df2 = df1[df1['isFraud'] == 1]

y = df2.groupby(['addr1']).agg('count')

y.rename(columns = {'isFraud':'Fraud'}, inplace = True)

y['addr1'] = y.index

# now join both x and y

x = x.reset_index(drop=True)

y = y.reset_index(drop=True)

xy = x.merge(y, on = 'addr1', how = 'left')

xy['Fraud'][xy['Fraud'].isna()] = 0

# now find the proportion of frauds

xy['fraud_ratio'] = xy['Fraud']/xy['isFraud']

xy = xy.sort_values('fraud_ratio', ascending = False)

xy['addr1'] =xy['addr1'].astype(int).astype(str)

plt.bar(xy['addr1'][:5], xy['fraud_ratio'][:5], color = 'orange')



# Region code 260 has the highest fraud occurence with 33%, however

# it reflects just 2 frauds out of 6 total transactions. Other top 10

# highest fraudulent regions we can see in tabular format:

cols = xy.columns.tolist()

cols = [cols[1]] + [cols[0]] + cols[2:]

xy = xy[cols]

xy.rename(columns = {'isFraud':'Total'}, inplace = True)

xy = xy.reset_index(drop=True)

xy.head(10)