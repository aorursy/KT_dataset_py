# Use one-hot encoding for categorical variables
# Take the log on sales price
# Use decision Trees, and set parameters as 1
# only use columns 
# ['Id','LotArea', 'OverallQual','OverallCond','YearBuilt','TotRmsAbvGrd','GarageCars','WoodDeckSF',
# 'PoolArea','SalePrice']

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
columns_to_use = ['Id', 'LotArea', 'OverallQual','OverallCond','YearBuilt',
                  'TotRmsAbvGrd','GarageCars','WoodDeckSF','PoolArea','SalePrice']
columns_in_test = columns_to_use.copy()
columns_in_test.remove("SalePrice")
columns_in_test
df = pd.read_csv("../input/train.csv", usecols=columns_to_use)
df.set_index('Id', inplace=True)
pd.options.display.max_rows=5
indexes = [1, 79, 347, 810, 811, 1179]
df = df.loc[indexes]
df
# pull data into target (y) and predictors (X)
train_y = np.log(df.SalePrice)
# predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = df.drop(['SalePrice'], 1)
from sklearn import tree
my_model = tree.DecisionTreeRegressor(random_state=42, max_depth=1)
my_model.fit(train_X, train_y)
import graphviz 
dot_data = tree.export_graphviz(my_model, out_file=None, feature_names=train_X.columns, filled=True) 
graph = graphviz.Source(dot_data) 
graph
# graph.render("housing")
