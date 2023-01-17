# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
# Since we already have numpy and pandas loaded, we will need to load the others that we are going to need
detectfall = pd.read_csv(r"../input/falldeteciton.csv")
# Let us look at the data now

print(detectfall.head())



# If you prefer another amount than the default, you can customized it

print(detectfall.head(3))
# You can see I have provided two different printing and got two different answers. This is how you can customize how many rows you would like to see
# Let us now take a further look at the data and inspect the distribution

detectfall.shape
detectfall.describe()
# Let us check to see if there are any null values. It is good to check for this

detectfall.isna().sum()
cols = ['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']
# Let's look at some plots now

fig = plt.figure(figsize = (10, 20)) # (Breite, Lange)

for i in range (0, len(cols)):

    fig.add_subplot(len(cols), 1, i+1)

    sns.distplot(detectfall[cols[i]]);
# Boxplot

sns.boxplot(data = detectfall)
# It looks like we have some heavy outliers in the EEG column. Since it is only a few value, we need to cut them out of the distribution. Also, SL has some outliers that need to be removed

detectfall = detectfall[(detectfall['EEG'] < detectfall['EEG'].quantile(0.999) ) 

& (detectfall['EEG'] > detectfall['EEG'].quantile(0.001))]



# For SL

detectfall = detectfall[(detectfall['SL'] < detectfall['SL'].quantile(0.999) ) 

& (detectfall['SL'] > detectfall['SL'].quantile(0.001))]
# Let us look at another boxplot of detectfall

sns.boxplot(data = detectfall)
# Before we start with the regression, we should take a time to look at what a picture of the data looks like. 

sns.lmplot('TIME', 'HR', data = detectfall,

          palette='Set1', fit_reg=False, scatter_kws={"s": 70})
# We could as well do a joint plot with seaborn to gather both histogram and scatter plot. This is one of the thing I like with seaborn. It gives you more option than matplotlib.

sns.jointplot('TIME', 'HR', detectfall)
# Another great feature that comes with seaborn is the heatmap plot. It allows you to see a plot of the correlation that each attribute has with each other. So here, I am sure you can see that the darker an attribute is in comparison with another, the least like there is any correlation between these two attributes. Just as in statistics, the closer the probability is to 1, the more there is a correlation; if it is closer to 0, the least likelyhood there is a correlation. So, this heatmap from seaborn just basically answered a few questions we may have had. This is very cool!!!

sns.heatmap(detectfall.corr())
from sklearn.model_selection import train_test_split



# Classifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



# Metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
# It is time to split the data and create a training and testing set

target = detectfall['ACTIVITY']

attribute = detectfall[['TIME','SL','EEG','BP','HR','CIRCLUATION']]



# Training and test sets

X_train, X_test, y_train, y_test = train_test_split(attribute, target, test_size = 0.3, random_state = 101)
# Our first Classifiers testing is the Logistic Regression

LogR = LogisticRegression()

LogR.fit (X_train, y_train)

pred = LogR.predict(X_test)

acc = accuracy_score(y_test, pred)

acc
# If you take a look at the answer given, we see that we cannot do a logistic regression because the target value is not between 0 and 1. 
# Now we are going to do a decision tree

dtcfall = DecisionTreeClassifier()

dtcfall.fit(X_train, y_train)

dtcfallpred = dtcfall.predict(X_test)

dtcfallacc = accuracy_score(y_test, dtcfallpred)

dtcfallacc
# Let us move on to the Random Forest Classifier

rfcfall = RandomForestClassifier()

rfcfall.fit(X_train, y_train)

rfcfallpred = rfcfall.predict(X_test)

rfcfallacc = accuracy_score(rfcfallpred, y_test)

rfcfallacc
# Try the KNN classifier

knnfall = KNeighborsClassifier()

knnfall.fit(X_train, y_train)

knnfallpred = knnfall.predict(X_test)

knnfallacc = accuracy_score(knnfallpred, y_test)

knnfallacc
# Our final classifier is the Support Vector Machine

svcfall = SVC(gamma='auto')

svcfall.fit(X_train, y_train)

svcfallpred = svcfall.predict(X_test)

svcfallacc = accuracy_score(svcfallpred,y_test)

svcfallacc
# Grid Search Parameter Tuning

alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])



# Create and fit a ridge regression model, testing each alpha

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

model = Ridge()

grid = GridSearchCV(estimator=model, param_grid=dict(alpha = alphas))

grid.fit(attribute, target)

print(grid)
# Summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)
# Let us try and randomized search

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform as sp_rand

# Prepare a uniform distribution to sample for the alpha parameter

paramgrid = {'alpha': sp_rand()}



# Create and fit a ridge regression model, testing random alpha values

model = Ridge()

rsearch = RandomizedSearchCV(estimator = model, param_distributions= paramgrid, n_iter=1000)

rsearch.fit(attribute, target)

print(rsearch)
# Summarize the results of the random parameter search

print(rsearch.best_score_)

print(rsearch.best_estimator_)
# We need to make this regression simpler; and the way to do it is to keep the attribute that contribute more to the falling of elderly people

plt.scatter('ACTIVITY', 'EEG', c = 'red', data = detectfall)
plt.scatter('ACTIVITY', 'TIME', c = 'blue', data = detectfall)
plt.scatter('ACTIVITY', 'SL', c = 'black', data = detectfall)
plt.scatter('ACTIVITY', 'HR', c = 'yellow', data = detectfall)
plt.scatter('ACTIVITY', 'BP', c = 'green', data = detectfall)
plt.scatter('ACTIVITY', 'CIRCLUATION', c = 'purple', data = detectfall)