# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


df = pd.read_csv('../input/behavioral-risk-factor-surveillance-system/2015.csv').sample(10000, random_state = 50)

df.head()
df['_RFHLTH'].value_counts()
df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df['_RFHLTH'].value_counts()
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df['_RFHLTH'].value_counts()
df = df.rename(columns = {'_RFHLTH': 'Label'})
df.shape
percentOfData = df.count()*100/9980
percentOfData.where(percentOfData<50).dropna()
badFeatures = percentOfData.where(percentOfData<50).dropna()
# Remove columns with missing values

df = df.drop(columns = badFeatures.index.to_list())
# Remove all non float data

df = df.select_dtypes(include=['float64'])
#Removing few more columns

df = df.drop(columns=['SEX','_STATE','FMONTH','SEQNO','DISPCODE','MARITAL','EDUCA','POORHLTH', 'PHYSHLTH', 'GENHLTH', 'HLTHPLN1', 'MENTHLTH'])
from IPython.display import HTML

HTML(pd.DataFrame(df.dtypes).to_html())
df.head()
from sklearn.model_selection import train_test_split



# Extract the labels

#labels = np.array(df.pop('Label'))



# 30% examples in test data

train, test, train_labels, test_labels = train_test_split(df, df['Label'], test_size = 0.3, 

                                                          random_state = 50)
# Imputation of missing values

train = train.fillna(train.mean())

test = test.fillna(test.mean())
train.columns
sns.distplot(train['Label'], kde=False)
train.shape
test.shape
# Train tree

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(random_state=50, max_depth=60)

tree.fit(train, train_labels)

print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')
# Make probability predictions

train_probs = tree.predict_proba(train)[:, 1]

probs = tree.predict_proba(test)[:, 1]



train_predictions = tree.predict(train)

predictions = tree.predict(test)
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve



print(f'Train ROC AUC Score: {roc_auc_score(train_labels, train_probs)}')

print(f'Test ROC AUC  Score: {roc_auc_score(test_labels, probs)}')
print(f'Baseline ROC AUC: {roc_auc_score(test_labels, [1 for _ in range(len(test_labels))])}')