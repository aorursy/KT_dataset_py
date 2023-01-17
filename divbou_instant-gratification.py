# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

from sklearn import metrics

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# Any results you write to the current directory are saved as output.
X = train.drop(['target','id'],axis=1)

y = train.target
y.head()

train.dtypes

train.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,

                                                    random_state=0)
model = QuadraticDiscriminantAnalysis(priors=0.5, reg_param=0.5)
model.fit(X_train,y_train)
prediction = model.predict(X_test)

sub = pd.concat([test.id, pd.DataFrame(prediction)], axis=1, ignore_index=True)

sub.columns = ['id','target']
metrics.roc_auc_score(y_test, prediction)
sub.to_csv('test_submission.csv', index=False, header=True)
sub.loc[1:5,:]