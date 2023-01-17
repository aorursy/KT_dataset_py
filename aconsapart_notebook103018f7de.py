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

import numpy

import xgboost

from numpy import loadtxt

from sklearn import cross_validation

from sklearn.metrics import accuracy_score

import xgboost as xg
dataset = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

#Checking for missing data

NAs = pd.concat([dataset.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])

NAs[NAs.sum(axis=1) > 0]
# Create linear regression object

regr = xg.XBGRegressor()