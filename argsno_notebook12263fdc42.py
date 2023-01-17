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
# Reading Data

df = pd.read_csv("../input/train.csv")
df.describe()
# Specify parameters via map

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}

num_round = 2

bst = xgb.train(param, dtrain, num_round)
# make prediction

preds = bst.predict(dtest)