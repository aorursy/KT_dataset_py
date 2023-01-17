# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import fbeta_score, accuracy_score

from sklearn.model_selection import train_test_split



from IPython.display import display

from time import time

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

filepath = '/kaggle/input/udacity-mlcharity-competition/'

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv(filepath+'/census.csv')

data.head(5)
subs_df=pd.read_csv('../input/udacity-mlcharity-competition/example_submission.csv')

test_census_df= pd.read_csv(filepath+'/test_census.csv')

test_census_df.head(5)
vis = sns.boxplot(data=data, x=data['income'], y=data['age'])

fig = vis.get_figure()
# TODO: Total number of records

n_records = data.shape[0]



# TODO: Number of records where individual's income is more than $50,000

n_greater_50k = data[data['income'] == '>50K'].shape[0]



# TODO: Number of records where individual's income is at most $50,000

n_at_most_50k = data[data['income']  == '<=50K'].shape[0]



# TODO: Percentage of individuals whose income is more than $50,000

greater_percent = float(n_greater_50k)*100/n_records



# Print the results

print("Total number of records: {}".format(n_records))

print("Individuals making more than $50,000: {}".format(n_greater_50k))

print("Individuals making at most $50,000: {}".format(n_at_most_50k))

print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
income_raw = data['income']

features_raw = data.drop('income', axis = 1)



plt.figure(figsize=(17,7))

plt.subplot(2,2,1)

sns.distplot(a=data['capital-gain'],hist=True,kde=False, color='r')

plt.legend()

plt.subplot(2,2,2)

sns.distplot(a=data['capital-loss'],hist=True,kde=False, color='blue')

plt.legend()
skewed = ['capital-gain', 'capital-loss']

features_log_transform = pd.DataFrame(data=features_raw)

features_log_transform[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

subs_df.to_csv('sample_submission.csv',index=False)

print('Successful Submission!!')