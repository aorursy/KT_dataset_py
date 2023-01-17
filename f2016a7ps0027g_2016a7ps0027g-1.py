import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import train_test_split



%matplotlib inline
train, test = pd.read_csv("../input/train.csv"), pd.read_csv("../input/test.csv")



data = pd.concat((train, test)).reset_index(drop=True)
data.head()
def convert_to_str(year):

    return str(year)

  

data['year'] = data['year'].apply(convert_to_str)



data = pd.get_dummies(data)

data.head()
cols_to_norm = ['4046','4225', '4770', 'Large Bags','Small Bags','Total Bags','Total Volume','XLarge Bags']



normalised_data = data[cols_to_norm].apply(lambda x: (x-np.mean(x))/ (max(x) - min(x)))

normalised_data.head()



data['4046'] = normalised_data['4046']

data['4225'] = normalised_data['4225']

data['4770'] = normalised_data['4770']

data['Large Bags'] = normalised_data['Large Bags']

data['Small Bags'] = normalised_data['Small Bags']

data['Total Bags'] = normalised_data['Total Bags']

data['Total Volume'] = normalised_data['Total Volume']

data['XLarge Bags'] = normalised_data['XLarge Bags']

data.head()
X = data.drop('AveragePrice',axis = 1)

y = train['AveragePrice']

X_train = X[:10000]

X_test = X[10000:]

y_train = y
X_train_t, X_val, y_train_t, y_val = train_test_split(X_train,y_train,test_size=0.1,random_state=13)
X_train.head()
X_test.columns
from xgboost import XGBRegressor as XGB

from sklearn.feature_selection import RFE



xgb = XGB(n_estimators=1000, learning_rate=0.01,reg_lambda = 3, subsample=0.50,colsample_bytree=1, max_depth=21,random_state = 42)

rfe = RFE(xgb, 7)             

rfe = rfe.fit(X_train_t, y_train_t)

print(rfe.support_)           

print(rfe.ranking_)
X_train.columns
pred = rfe.predict(X_val)

print(mse(pred,y_val))
drop_cols = X_train.columns[rfe.support_==False]

X_train.drop(drop_cols,axis=1,inplace = True)

X_test.drop(drop_cols,axis=1,inplace = True)

print(drop_cols)
xgb = XGB(n_estimators=5000, learning_rate=0.01,reg_lambda = 3, subsample=0.50,colsample_bytree=1, max_depth=21,random_state = 42)

xgb.fit(X_train,y_train)

pred = xgb.predict(X_test)

#my_submission = pd.DataFrame({'id':pd.read_csv(path+'test.csv').id,'AveragePrice':pred})
#my_submission.head()
#cols = my_submission.columns.tolist()

#cols = cols[-1:] + cols[:-1]

#my_submission = my_submission[cols]
#my_submission.head()
#my_submission.to_csv(path+'submission.csv',index=False)