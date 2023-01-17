"""Loading libraries and stuff"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Preprocessing and metrics"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""Regressors"""
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
% matplotlib inline
data = pd.read_csv('../input/train.csv', nrows=100000)
data.describe()
data.isnull().sum()
print('Size with nulls (train): %d' % len(data))
data = data.dropna(how = 'any', axis = 'rows')
print('Size without nulls (train): %d' % len(data))
k = 0
for i in data.fare_amount:
    if i < 0:
        k += 1
print("Number of fake fares: ", k)
print('Length of original data: %d' % len(data))
data = data[data.fare_amount>=0]
print('Lengh of new data: %d' % len(data))
data[data.fare_amount<70].fare_amount.hist(bins=70, figsize=(14,3))
plt.xlabel('Fare $USD')
plt.title("Histogram of fare's distribution");
data['lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_train  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'lat_change',
               'lon_change']
Y = data.fare_amount.as_matrix()
X = data[features_train]
Y = np.reshape(Y,(Y.shape[0],1))
X.head()
min_max_scaler = preprocessing.MinMaxScaler()
Y = min_max_scaler.fit_transform(Y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=42)
#np.save('Xtrain.npy',Xtrain)
#np.save('Xtest.npy', Xtest)
#np.save('Ytrain.npy', Ytrain)
#np.save('Ytest.npy', Ytest)
class Regressors:
    def __init__(self, Xtrain, Xtest, Ytrain, Ytest):
        """Initializing"""
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.Ytrain = Ytrain
        self.Ytest = Ytest

    def def_xgboost(self, estimators):
        """Defining the XGBoosting regressor"""
        xgb_ = xgb.XGBRegressor(objective ='reg:linear', learning_rate=0.01, max_depth=3, n_estimators=estimators)
        xgb_.fit(self.Xtrain, self.Ytrain)
        pred = xgb_.predict(self.Xtest)
        
        return pred

    def def_RandomForestRegressor(self, estimators):
        """Defining the Random Forest Regeressor"""
        rfr_ = RandomForestRegressor(n_estimators=estimators, max_depth=3)
        rfr_.fit(self.Xtrain, self.Ytrain)
        pred = rfr_.predict(self.Xtest)

        return pred

    def def_GradientBoostingRegressor(self, estimators):
        """Defining the Gradient Boosting Regressor"""
        gbr_ = GradientBoostingRegressor(n_estimators=estimators, max_depth=3)
        gbr_.fit(self.Xtrain, self.Ytrain)
        pred = gbr_.predict(self.Xtest)

        return pred

    def def_AdaBoostRegressor(self, estimators):
        """Defining Ada Boosting Regressor"""
        abr_ = AdaBoostRegressor(n_estimators=estimators)
        abr_.fit(self.Xtrain, self.Ytrain)
        pred = abr_.predict(self.Xtest)

        return pred
def def_metrics(ypred):
    mae = mean_absolute_error(Ytest, ypred)
    mse = mean_squared_error(Ytest, ypred)

    return mae, mse

def plot_performance(plot_name, loss_mae, loss_mse):
    steps = np.arange(50, 500, 50)
    plt.style.use('ggplot')
    plt.title(plot_name)
    plt.plot(steps, loss_mae, linewidth=3, label="MAE")
    plt.plot(steps, loss_mse, linewidth=3, label="MSE")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Number of estimators")
    plt.show()
"""Initializing the class"""
model = Regressors(Xtrain, Xtest, Ytrain, Ytest)
plot_name="XGBoosting Regressor"
loss_mae, loss_mse = [], []
print(plot_name)
for est in range(50,500,50):
    print("Number of estimators: %d" % est)
    mae, mse = def_metrics(model.def_xgboost(estimators = est))
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)
plot_name="Random Forest Regressor"
loss_mae, loss_mse = [], []
print(plot_name)
for est in range(50,500,50):
    print("Number of estimators: %d" % est)
    mae, mse = def_metrics(model.def_xgboost(estimators = est))
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)
plot_name="Gradient Boosting Regressor"
loss_mae, loss_mse = [], []
print(plot_name)
for est in range(50,500,50):
    print("Number of estimators: %d" % est)
    mae, mse = def_metrics(model.def_xgboost(estimators = est))
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)
plot_name="Ada Boost Regressor"
loss_mae, loss_mse= [], []
print(plot_name)
for est in range(50,500,50):
    print("Number of estimators: %d" % est)
    mae, mse = def_metrics(model.def_xgboost(estimators = est))
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)
