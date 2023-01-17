import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb1

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.preprocessing import Imputer

from sklearn import preprocessing

from datetime import datetime

import datetime

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import *

from sklearn.svm import SVC# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from xgboost import XGBClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.