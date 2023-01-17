# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sklearn

from sklearn.model_selection import StratifiedKFold 

import matplotlib as plt

%matplotlib inline

import warnings

from collections import Counter

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



#Read the data



#load the data into dataframes

submission=pd.read_csv('../input/sample_submission.csv')

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')



#Explore the data

train.shape

test.shape



train.head()

test.head()



train.describe()

test.describe()



train.info()

test.info()





train.isnull().sum()

test.isnull().sum()

#There are no null values. But there are -1s which we can convert to a proper value later



#Define target and ID columns:

target = 'target'

IDcol = ['id']

not_reqd =[]





predictors = [x for x in train.columns if x not in [target]+IDcol+not_reqd]







from sklearn.ensemble import RandomForestRegressor

alg1 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)

alg1.fit(train[predictors],train[target])









# Any results you write to the current directory are saved as output.