# Predicting Apparent Temperature from other variables on dataset



import numpy as np 

import pandas as pd 



# read data

df = pd.read_csv("/kaggle/input/szeged-weather/weatherHistory.csv")



# visualize first rows

df.head()
# analyse data

df.describe()
# plotting apparent temperature versus wind speed and humidity

from matplotlib import pyplot as plt



x = np.array(df[['Wind Speed (km/h)']])

y = np.array(df[['Apparent Temperature (C)']])

colors = np.array(df[['Humidity']])

plt.scatter(x, y, c=colors, alpha=0.5)

plt.rcParams['figure.figsize'] = [8, 8]

plt.show()
# define target

y = df[['Apparent Temperature (C)']]

y.head()
# data without target and other variables to be excluded from the model

X = df.loc[:, ~df.columns.isin(['Apparent Temperature (C)','Formatted Date','Daily Summary'])]

X.head()
# convert categorical variables

X = pd.get_dummies(X, prefix_sep='_', drop_first=False)

X.columns
# split into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train.columns
# creating xgboost matrices

import xgboost as xgb

train_matrix = xgb.DMatrix(X_train, label=y_train)

test_matrix = xgb.DMatrix(X_test, label=y_test)
# defining XGB parameters

xgb_reg = xgb.XGBRegressor(

    objective ='reg:squarederror',

    colsample_bytree = 0.8,

    learning_rate = 0.01,

    max_depth = 3,

    subsample = 0.5,

    gamma = 0,

    min_child_weight = 2,

    #alpha = 10,

    n_estimators = 1500,

    seed = 123

)

xgb_reg
# train the model

eval_set = [(X_train, y_train), (X_test, y_test)]

xgb_reg.fit(X_train, y_train, eval_metric = "rmse", eval_set=eval_set, verbose=False)
# making predictions

y_pred = xgb_reg.predict(X_test)

preds = [round(value) for value in y_pred]
# evaluate predictions

from sklearn import metrics



rmse = np.sqrt(metrics.mean_squared_error(np.array(y_test), preds))

print("RMSE: %f" % (rmse))

r2 = metrics.r2_score(np.array(y_test), preds)

print("R2: %.4f" % (r2))
# actuals versus predicted

from matplotlib import pyplot as plt



x = np.array(y_test)

y = preds

plt.scatter(x, y)

plt.rcParams['figure.figsize'] = [8, 8]

plt.show()
# plot feature importance

from matplotlib import pylab as plt



xgb.plot_importance(xgb_reg)

plt.rcParams['figure.figsize'] = [8, 8]

plt.show()
# retrieve performance metrics

results = xgb_reg.evals_result()

epochs = len(results['validation_0']['rmse'])

x_axis = range(0, epochs)



# plot RMSE

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['rmse'], label='Train')

ax.plot(x_axis, results['validation_1']['rmse'], label='Test')

ax.legend()

plt.ylabel('RMSE')

plt.title('XGBoost RMSE')

plt.rcParams['figure.figsize'] = [8, 8]

plt.show()