# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy



from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV, GridSearchCV

from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC, SVC
submit = pd.read_csv('../input/kernel27f23e847c/svc_pred.csv')

submit.index.name = "ImageId"

submit.index += 1

submit[['Label']].to_csv('svc_pred_indexed.csv')