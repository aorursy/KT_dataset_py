%matplotlib inline
from IPython.display import clear_output, Image, display
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir("../input/")
train_full = pd.read_csv("train.csv",sep=",",index_col=0)
train = train_full.drop("SalePrice",axis=1)
sales = train_full["SalePrice"]

test = pd.read_csv("test.csv",sep=",",index_col=0)
display(train.head())
display(test.head())
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(9,7))

ax1.hist(train_full["SalePrice"].values)
ax1.set_title("Sales Price")
ax2.hist(train_full["LotArea"].values)
ax2.set_title("LotArea")
ax3.hist(train_full["LotFrontage"].dropna().values)
ax3.set_title("LotFrontage")
ax4.hist(train_full["1stFlrSF"].values)
ax4.set_title("First Floor Area")
f.suptitle("Non-normalized variables")
#f.subplots_adjust(hspace=.3)
print("Original training shape: {}".format(train.shape))
print("Original test shape: {}".format(test.shape))
sales = np.log1p(train_full["SalePrice"])
all_data = pd.concat((train,test))
#display(all_data.head())
#print(all_data.shape)
numeric = all_data.dtypes[all_data.dtypes != "object"].index
all_data[numeric] = np.log1p(all_data[numeric])
all_data[numeric] = all_data[numeric].fillna(all_data[numeric].mean())
all_data=pd.get_dummies(all_data)
train = all_data.iloc[:train.shape[0],:]
test = all_data.iloc[train.shape[0]:,:]
print("New training shape: {}".format(train.shape))
print("New test shape: {}".format(test.shape))
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(9,7))

ax1.hist(sales.values)
ax1.set_title("Sales Price")
ax2.hist(train["LotArea"].values)
ax2.set_title("LotArea")
ax3.hist(train["LotFrontage"].dropna().values)
ax3.set_title("LotFrontage")
ax4.hist(train["1stFlrSF"].values)
ax4.set_title("First Floor Area")
f.suptitle("Normalized variables")
#f.subplots_adjust(hspace=.3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train,sales,test_size=.2,random_state=123, shuffle=True)
print(X_train.shape)
print(X_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def base_model():
    model = Sequential()
    model.add(Dense(200, input_dim=288, kernel_initializer='normal' ,activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model
def base_model():

    model = Sequential()
    model.add(Dense(200, input_dim=288, kernel_initializer='normal' ,activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model
model = KerasRegressor(build_fn=base_model, verbose=0)
batch_size = [5, 20, 40, 100]
epochs = [10, 50, 100,300]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring='neg_mean_squared_error',n_jobs=2)
grid_result = grid.fit(X_train.values, y_train.values)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
def base_model(l1=200,l2=100):
    model = Sequential()
    model.add(Dense(l1, input_dim=288, kernel_initializer='normal' ,activation='relu'))
    model.add(Dense(l2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='Adamax')
    return model

model = KerasRegressor(build_fn=base_model, verbose=0, epochs=10, batch_size=5)
l1 = [60, 200, 500]
l2 = [100, 250 ,400]
param_grid = dict(l1=l1, l2=l2)
grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring='neg_mean_squared_error')
grid_result = grid.fit(X_train.values, y_train.values)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
def base_model():
    alpha=.00001
    model = Sequential()
    model.add(Dense(500, input_dim=288, kernel_initializer='normal' ,activation='relu'))
    model.add(Dense(250, kernel_initializer='normal',activation='relu',))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='Adamax')
    return model
reg = KerasRegressor(build_fn=base_model, epochs=10, batch_size=20,verbose=1,validation_split=0.2)
kfold = KFold(n_splits=5, random_state=43)
results = np.sqrt(-1*cross_val_score(reg, X_train.values, y_train.values,scoring= "neg_mean_squared_error", cv=kfold))
print("Training RMSE mean and std from CV: {} {}".format(results.mean(),results.std()))
reg.fit(X_train.values, y_train.values)
prediction=reg.predict(X_test.values)
result = np.sqrt(mean_squared_error(y_test,prediction))
print("Testing RMSE: {}".format(result))
test_preds = np.expm1(reg.predict(test.values))
test_submission = pd.DataFrame({'id':test.index.values,"SalePrice" : test_preds})
#test_submission.loc[:,["id","SalePrice"]].to_csv("test_sale.csv", index = False)


