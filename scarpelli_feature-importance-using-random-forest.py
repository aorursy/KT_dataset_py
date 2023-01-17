# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Imports

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/numeric sequence.csv")

#Print to standard output, and see the results in the "log" section below after running your script

#print("\n\nTop of the training data:")

#print(train.head())



#print("\n\nSummary statistics of training data")

#print(train.describe())
train = train.values

new_format = np.zeros((2400,15)) # Creating the array with 14 (features) + 1 (binding/ label) columns

for i in range(14):

    new_format[0::,i] = train[0::,2*i] * 2 + train[0::,2*i+1] # new mapping 



new_format[0::,14] = train[0::,28] # Binding sites
limit = 2400 * 9/10 # We're going to use 90% of the data set for training and 10% for test (to get the scores)

# and repeat 50 times (each time we use a random permutation of the data set)



# The data is sorted (1200 binding sites, then 1200 non binding sites), let's do a random permutation on rows

data_set = np.random.permutation(new_format)

n_trees = 50 # number of trees in each forest (ie n_estimators)

n_test = 50



scores = np.zeros(n_test) # for each run we will calculate the score on the test set

features_importances = np.zeros((n_test,14)) # and we'll store the importance of each features



for i in range(0,n_test):

    data_set = np.random.permutation(data_set) # random permutation of rows

    train_set = data_set[0:limit,0::] # the first 90% of rows --> training set

    test_set = data_set[limit::,0::] # the remaining 10% --> test set

    forest = RandomForestClassifier(n_estimators= n_trees)

    forest = forest.fit( train_set[0::,0:14], train_set[0::,14])

    scores[i] = forest.score(test_set[0::, 0:14], test_set[0::,14])

    features_importances[i,0::] = forest.feature_importances_

print("The maximum score is %f" % max(scores))

print("The minimum score is %f" %min(scores))

print("The mean of the scores is %f" %np.mean(scores))

print("The median of the socres is %f" %np.median(scores))

print("The variance of the scores is %f" %np.var(scores))

print("Considering the data set is 50% label 1, 50% label2, the random classifier's score would be 50%")

delta = 100* (np.mean(scores) - 0.5)

print("In average this classifier is %f points better" %delta)

m_scores = np.mean(scores) * np.ones(n_test)
# Let's plot to vizualize the scores :

plt.plot(range(0,n_test),scores,'o')

plt.plot(range(0,n_test),m_scores)
labels = ['Base 1', 'Base 2', 'Base 3','Base 4','Base 5','Base 6','Base 7', 'Base 8', 'Base 9', 'Base 10', 'Base 11', 'Base 12', 'Base 13', 'Base 14']

x = range(0,n_test)

m = 1/14 * np.ones(n_test) # If every feature had the same importance it would be 1/14

for feat_imp, label in zip(np.transpose(features_importances), labels):

    plt.plot(x, feat_imp, label=label)

plt.plot(x,m,label='mean')

plt.legend()

plt.show()
features_importances_m = np.mean(features_importances,axis=0)

y = range(1,15)

m2 = 1/14 * np.ones(15)

plt.plot(y,features_importances_m,'ro',m2,'g')