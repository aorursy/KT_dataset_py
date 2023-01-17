import pandas as pd

import numpy

from sklearn import linear_model



# init

train_2d = pd.read_csv("../input/train.csv")

pred_2d = pd.read_csv("../input/test.csv")

x_train_2d, y_train_1d, x_pred_2d = train_2d.loc[:, "MSSubClass":"SaleCondition"], train_2d["SalePrice"], pred_2d.loc[:, "MSSubClass":"SaleCondition"]

x_2d = pd.concat((x_train_2d, x_pred_2d)); x_2d = x_2d.reset_index(); 
# example

cur_x_train_2d = x_train_2d.copy()

cur_x_train_2d['GrLivArea'] = numpy.log1p(cur_x_train_2d['GrLivArea'])
# example

x_2d["YrSold"] = 2010 - x_2d["YrSold"]
# example

x_2d['LotShape'] = x_2d['LotShape'].replace({'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1})
# example

x_2d, y_train_1d = x_2d.drop([1298]), y_train_1d.drop([1298])
# add some filler values for the sake of the example

linear_y_pred_1d = numpy.ones(100); xgb_y_pred_1d = numpy.ones(100)

# note that sometime it's better to do 0.6, 0.4 or 0.7 0.3. depends on relative strength of models.

avg_pred_1d = (linear_y_pred_1d * 0.5 + xgb_y_pred_1d * 0.5)
# my eventual lasso parametrs

regr = linear_model.Lasso(max_iter=1e6, alpha=5e-4)