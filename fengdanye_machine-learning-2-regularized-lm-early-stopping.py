import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import os
print(os.listdir("../input"))
plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)
# open the training dataset
dataTrain = pd.read_csv('../input/train.csv')
dataTrain.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
dataTrain = dataTrain.select_dtypes(include=numerics)
dataTrain.head()
dataTrain = dataTrain[['GarageArea','SalePrice']]
dataTrain.head()
dataTrain.isnull().values.any()
# Take a look at the data. 
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', linestyle = '')
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.show()
# format training data
xTrain = dataTrain['GarageArea'].values.reshape(-1,1) # as_matrix is deprecated since version 0.23.0
yTrain = dataTrain['SalePrice'].values.reshape(-1,1)
xTrain
# Transform the input features
Poly = PolynomialFeatures(degree = 10, include_bias = False)
xTrainPoly = Poly.fit_transform(xTrain)
from sklearn.preprocessing import StandardScaler
# standardization
scaler = StandardScaler()
xTrainPolyStan = scaler.fit_transform(xTrainPoly)
scaler.scale_, scaler.mean_
# linear regression
reg = LinearRegression()
reg.fit(xTrainPolyStan, yTrain)

# predict
xFit = np.linspace(0,1500,num=200).reshape(-1,1)
xFitPoly = Poly.transform(xFit)
xFitPolyStan = scaler.transform(xFitPoly)
yFit = reg.predict(xFitPolyStan)

# plot
plt.plot(xFit,yFit, lw=3, color='r', zorder = 2)
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.show()
from sklearn.linear_model import Ridge
i=0
ls = ['-','--',':']
color = ['r','g','orange']

for a in [0,2,2000]:
    ridgeReg = Ridge(alpha=a)
    ridgeReg.fit(xTrainPolyStan, yTrain)

    # predict
    xFit = np.linspace(0,1500,num=200).reshape(-1,1)
    xFitPoly = Poly.transform(xFit)
    xFitPolyStan = scaler.transform(xFitPoly)
    yFit = ridgeReg.predict(xFitPolyStan)
    
    # plot
    plt.plot(xFit,yFit, lw=3, color=color[i], zorder = 2, label= "alpha = " + str(a),linestyle=ls[i])
    i = i + 1
    
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()
from sklearn.linear_model import Lasso
i=0
ls = ['-','--',':']
color = ['r','g','orange']

for a in [0.1,1,10]:
    lassoReg = Lasso(alpha=a)
    lassoReg.fit(xTrainPolyStan, yTrain)

    # predict
    xFit = np.linspace(0,1500,num=200).reshape(-1,1)
    xFitPoly = Poly.transform(xFit)
    xFitPolyStan = scaler.transform(xFitPoly)
    yFit = lassoReg.predict(xFitPolyStan)
    
    # plot
    plt.plot(xFit,yFit, lw=3, color=color[i], zorder = 2, label= "alpha = " + str(a),linestyle=ls[i])
    i = i + 1
    
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(loss='squared_loss', penalty='l1', alpha=0.1)
yTrain = yTrain.ravel() # format required by sgd
sgd.fit(xTrainPolyStan, yTrain)

# predict
xFit = np.linspace(0,1500,num=200).reshape(-1,1)
xFitPoly = Poly.transform(xFit)
xFitPolyStan = scaler.transform(xFitPoly)
yFit = sgd.predict(xFitPolyStan)

plt.plot(xFit,yFit, lw=3, color='r', zorder = 2, label= "alpha = 0.1",linestyle='-')
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()
from sklearn.linear_model import ElasticNet
yTrain = yTrain.reshape(-1,1)
elasticReg = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
elasticReg.fit(xTrainPolyStan, yTrain)

# predict
xFit = np.linspace(0,1500,num=200).reshape(-1,1)
xFitPoly = Poly.transform(xFit)
xFitPolyStan = scaler.transform(xFitPoly)
yFit = elasticReg.predict(xFitPolyStan)

plt.plot(xFit,yFit, lw=3, color='r', zorder = 2, label= "alpha = 0.1",linestyle='-')
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()
from sklearn.metrics import mean_squared_error
dataTrain = pd.read_csv('../input/train.csv')
dataTrain.head()
# Obtain training data
xTrain = dataTrain[['OverallQual','LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'OpenPorchSF']].values
yTrain = dataTrain['SalePrice'].values.reshape(-1,1)
# Transform the data
poly2 = PolynomialFeatures(degree = 4, include_bias = False)
xTrainPoly = poly2.fit_transform(xTrain)
scaler = StandardScaler()
xTrainPolyStan = scaler.fit_transform(xTrainPoly)

# Fit the data
elasticReg = ElasticNet(alpha = 0.1, l1_ratio = 0.85)
elasticReg.fit(xTrainPolyStan, yTrain)

# evaluate performance on training set
yTrainHat = elasticReg.predict(xTrainPolyStan)

# calculate rmse based on log(price)
mse = mean_squared_error(np.log(yTrain), np.log(yTrainHat))
rmse = np.sqrt(mse)
print(rmse)
x = np.linspace(0,800000,num=1000)
plt.plot(yTrainHat, yTrain,marker='o', linestyle = '', zorder = 1, color='b')
plt.plot(x, x, linestyle = '-',color='red',zorder=2,lw=3)
plt.xlabel('predicted sale price (dollars)', fontsize = 18)
plt.ylabel('actual sale price (dollars)', fontsize = 18)
plt.show()
dataTest = pd.read_csv('../input/test.csv')
dataTest.head()
dataTest = dataTest[['Id','OverallQual','LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea','OpenPorchSF']]
dataTest.isnull().any()
# fill the nans with respective means.
dictMs = {'TotalBsmtSF':dataTest['TotalBsmtSF'].mean(skipna=True),
          'GarageArea':dataTest['GarageArea'].mean(skipna=True)}
dataTest = dataTest.fillna(value=dictMs)
dataTest.isnull().any()
xTest = dataTest[['OverallQual','LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'OpenPorchSF']].values
xTestPoly = poly2.transform(xTest)
xTestPolyStan = scaler.transform(xTestPoly)
yTestHat = elasticReg.predict(xTestPolyStan)
sub = pd.DataFrame()
sub['Id'] = dataTest['Id']
sub['SalePrice'] = yTestHat
sub.to_csv('submission.csv',index=False)
# Transform and standardize the data
poly2 = PolynomialFeatures(degree = 4, include_bias = False)
xTrainPoly = poly2.fit_transform(xTrain)
scaler = StandardScaler()
xTrainPolyStan = scaler.fit_transform(xTrainPoly)
yTrainLog = np.log(yTrain)
# shuffle data and split into training set and validation set
from sklearn.utils import shuffle
xShuffled, yShuffled = shuffle(xTrainPolyStan, yTrainLog)

train_ratio = 0.8
mTrain = np.int(len(xShuffled[:,0])*train_ratio) # 1168
print("Training sample size is: ", mTrain)

X_train_stan = xShuffled[0:mTrain]
Y_train = yShuffled[0:mTrain].ravel()
X_val_stan = xShuffled[mTrain:]
Y_val = yShuffled[mTrain:].ravel()
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sgdReg = SGDRegressor(n_iter=1, warm_start = True, penalty=None, learning_rate = 'constant', eta0=0.00001)

mse_val_min = float("inf")
best_epoch = None
best_model = None
rmse_train = []
rmse_val = []

n_no_change = 0

for epoch in range(1,100000):
    sgdReg.fit(X_train_stan, Y_train)
    Y_train_predict = sgdReg.predict(X_train_stan)
    train_error = mean_squared_error(Y_train_predict,Y_train)
    rmse_train.append(np.sqrt(train_error))
    Y_val_predict = sgdReg.predict(X_val_stan)
    val_error = mean_squared_error(Y_val_predict, Y_val)
    rmse_val.append(np.sqrt(val_error))
    
    if val_error < mse_val_min:
        n_no_change = 0
        mse_val_min = val_error
        best_epoch = epoch
        best_model = deepcopy(sgdReg)
    else:
        n_no_change = n_no_change + 1
    
    if n_no_change >= 1000:
        print('Time to stop!')
        print('num epoch =', epoch)
        print('best epoch = ', best_epoch)
        break
# plot rmse
fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))
plt.subplots_adjust(wspace=0.5)

ax[0].plot(rmse_train, label = 'training')
ax[0].plot(rmse_val, label = 'validation')
ax[0].set_xlabel('epoch', fontsize = 18)
ax[0].set_ylabel('RMSE', fontsize = 18)
ax[0].legend(fontsize=14)

ax[1].plot(rmse_train[best_epoch-3:best_epoch+2], label = 'training')
ax[1].plot(rmse_val[best_epoch-3:best_epoch+2], label = 'validation')
ax[1].set_xlabel('epoch', fontsize = 18)
ax[1].set_ylabel('RMSE', fontsize = 18)
ax[1].set_xticks([0,1,2,3,4])
xticklabels = [str(e) for e in range(best_epoch-2,best_epoch+3)]
ax[1].set_xticklabels(xticklabels)
ax[1].plot(2,rmse_val[best_epoch-1],marker='o',color='r')
ax[1].text(2,rmse_val[best_epoch-1]-0.001,'minimum',color='r',fontsize=14)
ax[1].legend(fontsize=14)
plt.show()
# total rmse on the train + validation sets
yTrainHatLog = best_model.predict(xTrainPolyStan)
print(np.sqrt(mean_squared_error(yTrainHatLog,yTrainLog)))
# plot the train + validation sets' predicted sale price vs actual sale price
plt.plot(np.exp(yTrainHatLog),np.exp(yTrainLog), marker = 'o', linestyle='', color = 'b')
x = np.linspace(0,800000,num=1000)
plt.plot(x, x, linestyle = '-',color='red',zorder=2,lw=3)
plt.xlabel('predicted sale price (dollars)', fontsize = 18)
plt.ylabel('actual sale price (dollars)', fontsize = 18)
plt.show()
from sklearn.model_selection import KFold
xShuffled, yShuffled = shuffle(xTrainPolyStan, yTrainLog)

sgdReg = SGDRegressor(n_iter=1, warm_start = True, penalty=None, learning_rate = 'constant', eta0=0.00001)

round_num = 0
best_epoch = None
best_model = None
rmse_train = []
rmse_val = []

kf = KFold(n_splits=5)

for train_index, val_index in kf.split(xShuffled):
    round_num = round_num + 1
    print("Round #", round_num)
    X_train_stan, X_val_stan = xShuffled[train_index], xShuffled[val_index]
    Y_train, Y_val = yShuffled[train_index].ravel(), yShuffled[val_index].ravel()
    
    print("Running...")
    mse_val_min = float("inf")
    n_no_change = 0

    for epoch in range(1,100000):
        sgdReg.fit(X_train_stan, Y_train)
        Y_train_predict = sgdReg.predict(X_train_stan)
        train_error = mean_squared_error(Y_train_predict,Y_train)
        rmse_train.append(np.sqrt(train_error))
        Y_val_predict = sgdReg.predict(X_val_stan)
        val_error = mean_squared_error(Y_val_predict, Y_val)
        rmse_val.append(np.sqrt(val_error))
    
        if val_error < mse_val_min:
            n_no_change = 0
            mse_val_min = val_error
            best_epoch = epoch
            best_model = deepcopy(sgdReg)
        else:
            n_no_change = n_no_change + 1
    
        if n_no_change >= 1000:
            print('Time to stop!')
            print('num epoch =', epoch)
            print('best epoch = ', best_epoch,', from round #', round_num)
            break
fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,4))
plt.subplots_adjust(wspace=0.5)

ax.plot(rmse_train, label = 'training')
ax.plot(rmse_val, label = 'validation')
ax.set_xlabel('epoch', fontsize = 18)
ax.set_ylabel('RMSE', fontsize = 18)
ax.legend(fontsize=14)
plt.show()
# total rmse on the train + validation sets
yTrainHatLog = best_model.predict(xTrainPolyStan)
print(np.sqrt(mean_squared_error(yTrainHatLog,yTrainLog)))
# plot the train + validation sets' predicted sale price vs actual sale price
plt.plot(np.exp(yTrainHatLog),np.exp(yTrainLog), marker = 'o', linestyle='', color = 'b')
x = np.linspace(0,800000,num=1000)
plt.plot(x, x, linestyle = '-',color='red',zorder=2,lw=3)
plt.xlabel('predicted sale price (dollars)', fontsize = 18)
plt.ylabel('actual sale price (dollars)', fontsize = 18)
plt.show()
# test set (data preprocessing - see Elastic Net part)
yTestHatLog = best_model.predict(xTestPolyStan)
sub = pd.DataFrame()
sub['Id'] = dataTest['Id']
sub['SalePrice'] = np.exp(yTestHatLog)
sub.to_csv('submission2.csv',index=False)
