# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/test.csv")

df.head()

df.describe()
dtrain = xgb.DMatrix('../input/test.csv',header=True)

dtest = xgb.DMatrix('../input/train.csv',header=True)

# specify parameters via map

param = {"num_class": 2, "objective" : "multi:softprob" }

num_round = 2

bst = xgb.train(param, dtrain, num_round)

# make prediction

preds = bst.predict(dtest)
print(preds)