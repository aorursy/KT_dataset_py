# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor as GBR

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

DATA_PATH = "../input/"
def load_data(housing_path=DATA_PATH):
    csv_path = os.path.join(DATA_PATH, "train_V2.csv")
    return pd.read_csv(csv_path,skipinitialspace = True)


pubg_data = load_data()

pubg_data = pubg_data.fillna(0)




pre_data = pubg_data.drop(["Id","groupId","matchId","matchType"],axis = 1)

pre_train_set, pre_test_set = train_test_split(pre_data,test_size=0.2,random_state=8)

#scaler = MinMaxScaler(feature_range=(0, 10))

train_set = pre_train_set.drop("winPlacePerc",axis = 1).values
train_label =pre_train_set["winPlacePerc"].values

test_set = pre_test_set.drop("winPlacePerc",axis = 1).values
test_label = pre_test_set["winPlacePerc"].values

#train_prepared = scaler.fit_transform(train_set)

lasso_reg = linear_model.Lasso(alpha = 0.1)
rid_reg = linear_model.Ridge(fit_intercept = True)

#score = cross_val_score(gbr_reg, train_set, train_label,cv=5)
param_grid = {'alpha':[50,100,500,1000]}
grid_search = GridSearchCV(rid_reg, param_grid, cv=5)
grid_search.fit(train_set,train_label)

print("best alpha param")
print(grid_search.best_params_)

finalmode = grid_search.best_estimator_

game_test = pd.read_csv('../input/test_V2.csv', index_col='Id')
pre_test = game_test.drop(["groupId","matchId","matchType"],axis = 1).values

predict_test = finalmode.predict(pre_test)

dfpredictions = pd.DataFrame(predict_test, index=game_test.index).rename(columns={0:'winPlacePerc'})


print(dfpredictions.head(15))

dfpredictions.to_csv('submission.csv', header=True)




# Any results you write to the current directory are saved as output.
