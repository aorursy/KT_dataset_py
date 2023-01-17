# Source - https://www.datacamp.com/community/tutorials/xgboost-in-python



from sklearn.datasets import load_boston

boston = load_boston()

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
print(boston.DESCR)
dat = pd.DataFrame(boston.data)

dat.columns = boston.feature_names

dat['Price'] = boston.target

dat.describe()
dat.head()
#Missing values

tot = dat.isnull().sum()

per = (dat.isnull().sum()/dat.isnull().count())*100

missing = pd.concat([tot, per], axis = 1, keys= ['# missing', '% missing'])

missing.transpose()
# Correlation 

plt.figure(figsize=(7,7))

cor = dat.corr()

sns.heatmap(cor, linewidths=.1,cmap="Reds", annot= True)



# Tax and Rad are highly correlated
# convert data to DMatrix - a format that is optimized to run xgb on

x,y = dat.iloc[:,:-1],dat.iloc[:,-1]

dat_m = xgb.DMatrix(data = x, label = y)
# Test train split

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(x_train,y_train)

preds = xg_reg.predict(x_test)
#RMSE value for basic xg boosting

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))
#setting up parameters



params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}



cv_results = xgb.cv(dtrain=dat_m, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results.tail()
x_r = xgb.train(params=params, dtrain=dat_m, num_boost_round=10)
xgb.plot_importance(xg_reg)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()