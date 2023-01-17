# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/HR_comma_sep.csv')



df.rename(columns={'average_montly_hours':'average_monthly_hours','sales':'department'},inplace=True)



df.describe()



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')



## Attrition by departments

#for col in df.columns:

#    plot = sns.factorplot(x=col, y='left', kind='bar', data=df)

#    plot.set_xticklabels(rotation=45, horizontalalignment='right');

    

X = df.drop('left', axis=1)

y = df['left']

X.drop(['department','salary'], axis=1, inplace=True)



# One-hot encoding

salary_dummy = pd.get_dummies(df['salary'])

department_dummy = pd.get_dummies(df['department'])



import sklearn.model_selection as sklmdl



X_train, X_test, y_train, y_test = sklmdl.train_test_split(X, y, test_size=0.3)



import sklearn.preprocessing as sklprepro

stdsc = sklprepro.StandardScaler()



# transform our training features

X_train_std = stdsc.fit_transform(X_train)

# transform the testing features in the same way

X_test_std = stdsc.transform(X_test)



# Cross validation

from sklearn.model_selection import ShuffleSplit



cv = ShuffleSplit(n_splits=20, test_size=0.3)





reduced_features = ['satisfaction_level', 'time_spend_company', 

                    'number_project', 'average_monthly_hours', 'last_evaluation']

X2_train = X_train[reduced_features]

X2_test = X_test[reduced_features]

stdsc2 = sklprepro.StandardScaler()

X2_train_std = stdsc2.fit_transform(X2_train)

X2_test_std = stdsc2.transform(X2_test)



# Model #4: K-mean clustering

import sklearn.cluster as sklclstr



# Fit entire dataset. Reduced features (top 5 from RF importance scores); scaled.

X2 = X[reduced_features]

X2_std = stdsc2.fit_transform(X2)



# Inertia vs. # of clusters

x1 = []

y1 = []

for n in range(2,11):

    km = sklclstr.KMeans(n_clusters=n, random_state=7)

    km.fit(X2_std)

    x1.append(n)

    y1.append(km.inertia_)

plt.scatter(x1, y1)

plt.plot(x1, y1);





km = sklclstr.KMeans(n_clusters=7, n_init=20, random_state=7)

km.fit(X2_std)

columns = {str(x): stdsc2.inverse_transform(km.cluster_centers_[x]) for x in range(0,len(km.cluster_centers_))}

pd.DataFrame(columns, index=X2.columns)



# Percentage of employees left for each cluster. Helps identify which cluster to direct our focus.

kmpredict = pd.DataFrame(data=df['left'])

kmpredict['cluster'] = km.labels_

kmpredict.groupby('cluster').mean()
