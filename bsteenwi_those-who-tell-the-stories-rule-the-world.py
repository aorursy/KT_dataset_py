import pandas as pd

from sklearn import preprocessing

import random

import numpy

import warnings

import ruleset as rs





warnings.filterwarnings('ignore')



random.seed(42)

numpy.random.seed(42)
df = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

#example data

df[['Q1','Q2','Q3','Q4','Q5','Q6','Q7', 'Q8']].head()
df['Q24_Part_4'].value_counts().sort_index(ascending=False)
# load dataset, remove first row (=question text)

df = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv').iloc[1:]

# replace the underscore in the dataset, just for to get a better output

df.columns = df.columns.str.replace('_', '')

# unknown for all NaN values, we will discriminate between 'Bayesian approach' & 'Unknown' in column Q24_Part_4

df.fillna('unknown', inplace=True)



# X: training set, 

# !!!! we use a predefined set of columns to speed up the execution time !!!! (normally use .drop(['Q24Part4', , axis=1))

X = df[['Q9Part5', 'Q18OTHERTEXT', 'Q19', 'Q24Part5', 'Q18Part2', 'Q8', 'Q24Part9', 'Q29Part10']]

# change dtype of columns to string (categorical)

for col in X.columns:

    X[col] = X[col].apply(str)



# encode label

le = preprocessing.LabelEncoder()

# we want rules specifying the 'Bayesian approach' community

y = 1-le.fit_transform(df.Q24Part4)



# let's mine some rules, limit to max 5 conjunctive rules of max length 5

# https://github.com/zli37/bayesianRuleSet

model = rs.BayesianRuleSet(method='forest', max_iter=10000, maxlen=5, max_rules=5)

model.fit(X, y)