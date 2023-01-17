import numpy as np

import pandas as pd

from sklearn import model_selection, datasets, linear_model, metrics

from sklearn.preprocessing import scale

from sklearn.linear_model import Lasso

from xgboost import XGBRegressor

from sklearn.externals import joblib

from sklearn import preprocessing





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/train.csv')

test = pd.read_csv('/kaggle/input/test.csv')

sub = pd.read_csv('/kaggle/input/sample_submission.csv')

data.select_dtypes(object).keys()
labelEncoder = preprocessing.LabelEncoder()



for i in data.select_dtypes(object).keys():

    data[i] = labelEncoder.fit_transform(data[i])

    test[i] = labelEncoder.transform(test[i])
X = scale(data.drop(columns = ['Energy_consumption','Id']))

y = data["Energy_consumption"]

data_col = data.drop(columns = ['Energy_consumption','Id']).columns

lasso_clasificator = Lasso(alpha=2)#11 - nado

lasso_clasificator.fit(X,y)

inform_data =[]

coef = []

for i,item in enumerate(lasso_clasificator.coef_):

    if item != abs(0):

        inform_data.append(data_col[i])

        coef.append(item)



print("Количество информативных признаков - ", len(inform_data))

xg_reg = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.15, learning_rate = 0.001,

                max_depth = 5, n_estimators = 70000,subsample=0.5)

xg_reg.fit(data[inform_data], data['Energy_consumption'] )
xbg_scoring = model_selection.cross_val_score(xg_reg,data[inform_data], data['Energy_consumption'], scoring = 'neg_mean_squared_error', cv = 5)

xbg_scoring.mean()
pred = xg_reg.predict(test[inform_data])

sub['Energy_consumption'] = pred

sub.to_csv('/kaggle/working/subEND.csv')