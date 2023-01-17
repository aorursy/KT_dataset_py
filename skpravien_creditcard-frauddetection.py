import pandas as pd

import numpy as np 

import tensorflow as tf

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
df = pd.read_csv("../input/creditcard.csv")

df.head()
df.describe()
df.isnull().sum()
print ("Fraud")

print (df.Time[df.Class == 1].describe())

print ()

print ("Normal")

print (df.Time[df.Class == 0].describe())
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 50



ax1.hist(df.Time[df.Class == 1], bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Time[df.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Time (in Seconds)')

plt.ylabel('Number of Transactions')

plt.show()
print ("Fraud")

print (df.Amount[df.Class == 1].describe())

print ()

print ("Normal")

print (df.Amount[df.Class == 0].describe())
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 30



ax1.hist(df.Amount[df.Class == 1], bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Amount[df.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.yscale('log')

plt.show()
df['Amount_max_fraud'] = 1

df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))



ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])

ax1.set_title('Fraud')



ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])

ax2.set_title('Normal')



plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()
#Select only the anonymized features.

v_features = df.ix[:,1:29].columns

plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(df[v_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(df[cn][df.Class == 1], bins=50)

    sns.distplot(df[cn][df.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of feature: ' + str(cn))

plt.show()