# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





from sklearn import model_selection

from sklearn import linear_model

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')

df.head(10)
df.describe()
print(df.shape)

df.tail(10)
fraud = df[df.Class==1]

print(fraud.shape)

fraud.head(10)
#visualize time, what is it? is it epoch?

df.Time.describe()

# are there any relation to time that fraud happens at
plt.figure(figsize=(15,15))

sns.heatmap(fraud.corr(),annot=True,cmap='coolwarm',fmt='.1f')
non_fraud = df[df.Class!=1]

plt.figure(figsize=(15,15))

sns.heatmap(non_fraud.corr(),annot=True,cmap='coolwarm',fmt='.1f')
# get 500 of random non-fraud data points

non_fraud_sample = non_fraud.sample(500)

non_fraud_sample.head(10)

non_fraud_sample.describe()

sample_df = pd.concat([non_fraud_sample, fraud])

sample_df.describe()
# split into train and test of 400 train and test

train, test = model_selection.train_test_split(sample_df, train_size=800)

print(train.shape, test.shape)
# supervised learning try different ml algorithms from sklearn

target_label = 'Class'

train_x = train.loc[:,'Time':'Amount']

train_y = train.loc[:,target_label]

test_x = test.loc[:,'Time':'Amount']

test_y = test.loc[:,target_label]
# try logistic regression since the target variable is categorical

ln = linear_model.LogisticRegression()

reg = ln.fit(train_x, train_y)

score = reg.score(train_x, train_y)

# coeff_df = pd.DataFrame(reg.coef_, train_x.columns, columns=['Coefficient'])

print(score)

reg.coef_
# predict

predict_y = ln.predict(test_x)

df = pd.DataFrame({'Actual': test_y, 'Predicted': predict_y})

df.head(10)
# print various metrics

metrics.accuracy_score(test_y, predict_y)

# get f1 score

# get confusion matrix and calc precision and recall
# plot auc
#try unsupervised learning
# maybe deep learning??