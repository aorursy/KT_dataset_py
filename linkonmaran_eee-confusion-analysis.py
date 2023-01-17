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
from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

import matplotlib
# Reading CSV file

conf = pd.read_csv("../input/EEG data.csv",delimiter = ",")

X = conf[[2,3,4,5,6,7,8,9,10,11,12]]

Y = conf[[14]]
# Feature Selection based on LinearSVC

Y_new = Y.values.ravel()

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y_new)

model = SelectFromModel(lsvc, prefit=True)

bl = model.get_support()

X_new = model.transform(X)
# idx gives index of significant features. 

idx = [i for i in range(len(bl)) if bl[i]]

#[2]: Attention (Proprietary measure of mental focus)

#[3]: Mediation (Proprietary measure of calmness)

#[4]: Raw (Raw EEG signal)
# Random Forest Classifier

clsfr = RandomForestClassifier()

r,c = Y.shape

Y_re = Y.values.reshape(r) 
# Cross Validation

scores = cross_val_score(clsfr, X_new, Y_re,cv =5)
# Average of Accuracy scores (~55%)

mean = sum(scores)/len(scores)