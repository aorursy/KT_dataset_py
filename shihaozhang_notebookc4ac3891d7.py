import numpy as np 

import pandas as pd 

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb



# Any results you write to the current directory are saved as output.
people = pd.read_csv("../input/people.csv")

act_train = pd.read_csv("../input/act_train.csv")

act_test = pd.read_csv("../input/act_test.csv")
print("People.shape is {}".format(people.shape))

print("act_train.shape is {}".format(act_train.shape))

print("act_test.shape is {}".format(act_test.shape))
train = pd.merge(people, act_train, how='inner', on='people_id',suffixes=('_p', '_act'))

print("train data set's dimension is {}".format(train.shape))
dtrain = xgb.DMatrix('../data/agaricus.txt.train')