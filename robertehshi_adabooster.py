# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import log, exp
import pylab as pl
from random import random


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv', index_col='ID')

feature_cols = list(training_data.columns[1:-1])
target_col = training_data.columns[-1]
#from sklearn.decomposition import PCA
#pca = PCA(n_components=100, whiten=True).fit(training_data[feature_cols]) # 14 PCs explain virtually all the variance

# Print the components and the amount of variance in the data contained in each dimension
#print ('\n', pca.explained_variance_ratio_)
#print(sum(pca.explained_variance_ratio_))
group_size = training_data[training_data['TARGET']==1].shape[0]

training_data = training_data.reindex(np.random.permutation(training_data.index))

balanced_data = pd.concat([training_data[training_data['TARGET']==1][:group_size], 
                           training_data[training_data['TARGET']==0][:group_size]])

y_train = balanced_data[target_col]
X_train = balanced_data[feature_cols]
#y_train = pd.DataFrame(balanced_data[target_col])
#X_train = pd.DataFrame(pca.transform(balanced_data[feature_cols]), index=y_train.index)

#X_test = pd.DataFrame(pca.transform(test_data[feature_cols]), index=test_data.index)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

booster = AdaBoostClassifier(n_estimators=151)
booster.fit(X_train,y_train)

results = pd.DataFrame({'TARGET':booster.predict(test_data)}, index=test_data.index)
print(results)

results.to_csv('submission.csv')
