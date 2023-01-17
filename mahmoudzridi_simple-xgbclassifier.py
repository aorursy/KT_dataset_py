# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve

from sklearn.metrics import average_precision_score

from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance, to_graphviz





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/abstract-data-set-for-credit-card-fraud-detection/creditcardcsvpresent.csv')

df.head()
df.loc[df['Is declined']=='N','Is declined'] = 0

df.loc[df['Is declined']=='Y','Is declined'] = 1

df['Is declined'].unique()
df.loc[df['isForeignTransaction']=='N','isForeignTransaction'] = 0

df.loc[df['isForeignTransaction']=='Y','isForeignTransaction'] = 1

df['isForeignTransaction'].unique()
df.loc[df['isHighRiskCountry']=='N','isHighRiskCountry'] = 0

df.loc[df['isHighRiskCountry']=='Y','isHighRiskCountry'] = 1

df['isHighRiskCountry'].unique()
df.loc[df['isFradulent']=='N','isFradulent'] = 0

df.loc[df['isFradulent']=='Y','isFradulent'] = 1

df['isFradulent'].unique()
df.head()
print(df.shape)

df.isnull().sum()
df.dropna(axis=1,inplace=True)

df.head()
df = df.rename(columns={'Average Amount/transaction/day' : 'AverageAmount_transaction_day'})

df = df.rename(columns={'Is declined' : 'IsDeclined'})

df = df.rename(columns={'Total Number of declines/day' : 'TotalNumberOfDeclines_day'})

df = df.rename(columns={'6-month_chbk_freq' : '6_month_chbk_freq'})
print('YES NO             isFradulent')

print(df[df['isFradulent']==1].isFradulent.sum(), df[df['isFradulent']==0].isFradulent.count(),'count')

print(df[df['isFradulent']==1].isHighRiskCountry.sum(), df[df['isFradulent']==0].isHighRiskCountry.sum(),'isHighRiskCountry sum')

print(df[df['isFradulent']==1].isForeignTransaction.sum(), df[df['isFradulent']==0].isForeignTransaction.sum(),'isForeignTransaction sum')

print(df[df['isFradulent']==1].IsDeclined.sum(), df[df['isFradulent']==0].IsDeclined.sum(),'IsDeclined sum')

print(df[df['isFradulent']==1].TotalNumberOfDeclines_day.mean(), df[df['isFradulent']==0].TotalNumberOfDeclines_day.mean(),'TotalNumberOfDeclines_day mean')
avgf = df[df['isFradulent']==1].Transaction_amount/df.AverageAmount_transaction_day

avgl = df[df['isFradulent']==0].Transaction_amount/df.AverageAmount_transaction_day

print('average of transaction amount per AverageAmount_transaction_day')

print('Fraud :',avgf.mean())

print('legal :',avgl.mean())
df['avgs'] = df.Transaction_amount/df.AverageAmount_transaction_day

df.head()
Y = df['isFradulent']

X = df.drop(['Merchant_id','isFradulent'],axis=1)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2)
weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)

probabilities = clf.fit(trainX, trainY).predict_proba(testX)

print(average_precision_score(testY, probabilities[:, 1]))
'''

The figure below shows that the new feature avgs that

we created isthe most relevant feature for the model.

The features are ordered based on the number of 

samples affected by splits on those features.

'''

fig = plt.figure(figsize = (14, 9))

ax = fig.add_subplot(111)



colours = plt.cm.Set1(np.linspace(0, 1, 9))



ax = plot_importance(clf, height = 1, color = colours, grid = False, \

                     show_values = False, importance_type = 'cover', ax = ax);

for axis in ['top','bottom','left','right']:

            ax.spines[axis].set_linewidth(2)

        

ax.set_xlabel('importance score', size = 16);

ax.set_ylabel('features', size = 16);

ax.set_yticklabels(ax.get_yticklabels(), size = 12);

ax.set_title('Ordering of features by importance to the model learnt', size = 20);