import numpy as np

from sklearn import metrics



def mse(y_true, y_pred):

    return metrics.mean_squared_error(y_true, y_pred)



def rmse(y_true, y_pred):

    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))



def mae(y_true, y_pred):

    return metrics.mean_absolute_error(y_true, y_pred)



def mape(y_true, y_pred):

    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100



def smape(y_true, y_pred):

    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100



def rmsle(y_true, y_pred):

    return np.sqrt(metrics.mean_squared_log_error([p if p>=0 else 0 for p in y_pred],[t if t>=0 else 0 for t in y_true]))



y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])

y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])



# MSE

print(mse(y_true, y_pred)) # 8.107142857142858

# RMSE

print(rmse(y_true, y_pred)) # 2.847304489713536

# MAE

print(mae(y_true, y_pred)) # 1.9285714285714286

# MAPE

print(mape(y_true, y_pred)) # 76.07142857142858

# SMAPE

print(smape(y_true, y_pred)) # 57.76942355889724

# RMSLE

print(rmsle(y_true, y_pred))
##### for xgboost demo rmsle

def rmsle_4_xgboost(y_pred, dtrain):

    y_true = dtrain.get_label()

    return 'error', rmsle(y_true, y_pred)