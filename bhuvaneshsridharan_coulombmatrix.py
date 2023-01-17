# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from scipy.io import loadmat
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data = loadmat("../input/qm7-atomization-energy/qm7.mat")
crossvalid_no = 1
no_of_train = 100
test_data_indices = raw_data['P'][crossvalid_no].flatten()
train_data_indices = np.array(list(set(raw_data['P'].flatten()) - set(test_data_indices)))[:no_of_train]
print(test_data_indices.shape)
print(train_data_indices.shape)
train_data_X = raw_data['X'][train_data_indices]

train_data_Y = raw_data['T'][0][train_data_indices]
test_data_X = raw_data['X'][test_data_indices].reshape(1433,23*23)
test_data_Y = raw_data['T'][0,test_data_indices]
train_X_reshaped = train_data_X.reshape(len(train_data_X),23*23)
reg_mod = KernelRidge(alpha = 1e-8, kernel = "laplacian", gamma=1/4000)
reg_mod.fit(train_X_reshaped, train_data_Y)
prediction = reg_mod.predict(test_data_X)
mean_absolute_error(prediction, test_data_Y)
trainVsMae = []
for train_size in range(500, 6000, 250): #iterating through different train sizes
    if train_size > 5732: train_size = 5732
    mae = np.array([])
    
    for validation_no in range(5): #different crossvalidation
        
        #Randomly making test and train data indices
        
        test_data_indices = raw_data['P'][validation_no].flatten()
        train_data_indices = np.array(list(set(raw_data['P'].flatten()) - set(test_data_indices)))[:train_size]
        
        train_data_X = raw_data['X'][train_data_indices]
        train_data_Y = raw_data['T'][0][train_data_indices]
        
        test_data_X = raw_data['X'][test_data_indices].reshape(1433,23*23)
        test_data_Y = raw_data['T'][0,test_data_indices]
        
        train_X_reshaped = train_data_X.reshape(len(train_data_X),23*23)
        
        reg_mod = KernelRidge(alpha = 1e-8, kernel = "laplacian", gamma=1/4000)
        reg_mod.fit(train_X_reshaped, train_data_Y)
        predictions = reg_mod.predict(test_data_X)
        mae = np.append(mae, mean_absolute_error(test_data_Y, predictions))
        print(" Train Size : {:n} , valid: {:n}  MAE: {:f}".format(train_size, validation_no, mae[-1]))
    trainVsMae = np.append(trainVsMae, np.average(mae))
    print("Net mae for train_size : {:n} = {:f}".format(train_size, trainVsMae[-1]))
        
plt.xlabel("Training Size")
plt.ylabel("MAE")
plt.title("Training Size vs MAE")
plt.plot([500+i*250 for i in range(len(trainVsMae)-2)], trainVsMae[2:], "b--^")
plt.show()
trainVsMae
print("MAE error for this method(Laplacian Kernel): {:f}".format(trainVsMae[-1]))
trainVsMaeGaussian = []
for train_size in range(500, 6000, 250): #iterating through different train sizes
    if train_size > 5732: train_size = 5732
    mae = np.array([])
    
    for validation_no in range(5): #different crossvalidation
        
        #Randomly making test and train data indices
        
        test_data_indices = raw_data['P'][validation_no].flatten()
        train_data_indices = np.array(list(set(raw_data['P'].flatten()) - set(test_data_indices)))[:train_size]
        
        train_data_X = raw_data['X'][train_data_indices]
        train_data_Y = raw_data['T'][0][train_data_indices]
        
        test_data_X = raw_data['X'][test_data_indices].reshape(1433,23*23)
        test_data_Y = raw_data['T'][0,test_data_indices]
        
        train_X_reshaped = train_data_X.reshape(len(train_data_X),23*23)
        
        reg_mod = KernelRidge(alpha = 1e-3, kernel = "rbf", gamma=0.0001389)
        reg_mod.fit(train_X_reshaped, train_data_Y)
        predictions = reg_mod.predict(test_data_X)
        mae = np.append(mae, mean_absolute_error(test_data_Y, predictions))
        print(" Train Size : {:n} , valid: {:n}  MAE: {:f}".format(train_size, validation_no, mae[-1]))
    trainVsMaeGaussian = np.append(trainVsMaeGaussian, np.average(mae))
    print("Net mae for train_size : {:n} = {:f}".format(train_size, trainVsMaeGaussian[-1]))
        
plt.xlabel("Training Size")
plt.ylabel("MAE")
plt.title("Training Size vs MAE")
plt.plot([500+i*250 for i in range(len(trainVsMaeGaussian)-2)], trainVsMaeGaussian[2:], "r--^", label="Gaussian Kernel")
plt.plot([500+i*250 for i in range(len(trainVsMae)-2)], trainVsMae[2:], "b--^", label="Laplacian Kernel")
plt.legend()
plt.show()
print("MAE error for this method(Laplacian Kernel): {:f}".format(trainVsMaeGaussian[-1]))