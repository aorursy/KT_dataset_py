# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # cool plots

import matplotlib.pyplot as plt # regular plots





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/data.csv')

dataset.describe()



y = dataset.diagnosis



# for now, only look at columns that have the mean suffix. otherwise our predictors will be highly correlated. 

cols = [col for col in dataset.columns.values if 'mean' in col]

x = dataset[cols]



dataset['e'] = 1 # hack to make violinplot work. 
f, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,10))

for i, col in enumerate(x.columns):

    #plt.figure(i, figsize=(2,2))

    sns.violinplot(data=dataset[[col, "diagnosis", "e"]], x="e", y=col, split=True, hue="diagnosis", ax=axes[i // 4][i % 4])

plt.tight_layout()
# correlation matrix

corr = x.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
# remove perimeter_mean, area_mean as these are (obviously) highly correlated with radius_mean

# we just need to keep one of the three. my choice of radius_mean is arbitrary

cleaned_x = x.drop(columns=["perimeter_mean", "area_mean"])



from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix



# Support Vector Machine

clf = svm.SVC(gamma='scale', kernel='poly')

scores = cross_val_score(clf, cleaned_x, y, cv=10)

print("Accuracy: {}".format(scores.mean()))



clf.fit(cleaned_x, y)



print(confusion_matrix(y, clf.predict(cleaned_x)))

# Based on the violin plots above, these features looks _somewhat_ normal.

# Let's try quadratic discriminant analysis because the covariances 

# are clearly different wrt diagnosis. 

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



qda = QuadraticDiscriminantAnalysis()



scores_qda = cross_val_score(qda, cleaned_x, y, cv=10)

print("Accuracy: {}".format(scores_qda.mean()))



qda.fit(cleaned_x, y)

print(confusion_matrix(y, qda.predict(cleaned_x)))

# how confident is the model wrt wrong predictions?

wrong_predictions_confidence = [tup for tup in zip(qda.predict(cleaned_x), y, [max(v) for v in qda.predict_proba(cleaned_x)]) if tup[0] != tup[1]]

wrong_predictions_confidence