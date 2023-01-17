# standard packages
import os
import warnings
warnings.filterwarnings('ignore') 
import pickle
import random

# third-party packages
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Import data - Korean Finance Data
org_data = pd.read_csv("../input/vkospi/options_KR.csv")
# Import data - Volatility index around the world
CBOE_Volatility = pd.read_csv("../input/volatility-index-around-the-world/CBOE Volatility Index Historical Data.csv")
HSI_Volatility  = pd.read_csv("../input/volatility-index-around-the-world/HSI Volatility Historical Data.csv")
Nikkei_Volatility = pd.read_csv("../input/volatility-index-around-the-world/Nikkei Volatility Historical Data.csv")

CBOE_Volatility = CBOE_Volatility.loc[:,['Date','Price']]
HSI_Volatility = HSI_Volatility.loc[:,['Date','Price']]
Nikkei_Volatility = Nikkei_Volatility.loc[:,['Date','Price']]
# Convert date data
org_data.Date = pd.to_datetime(org_data.Date)
CBOE_Volatility.Date = pd.to_datetime(CBOE_Volatility.Date)
HSI_Volatility.Date = pd.to_datetime(HSI_Volatility.Date)
Nikkei_Volatility.Date = pd.to_datetime(Nikkei_Volatility.Date)
# Reformat data, because it is time series data.
#
# VKSOPI of Day n is predicted from variabels of Day n-1,
# except for 'Day_till_expiration' and 'Day_of_a_week' since they can be known in the prediction.
#
# What we want our df to be:
# Date | VKOSPI | ~ | Day_till_expiration | Day_of_a_week
# n      n       n-1  n                     n

pivot = ['Date','VKOSPI','Day_till_expiration','Day_of_a_week']

# copy the original data
full_data = org_data.copy()

# shift the pivot data
full_data[pivot] = full_data[pivot].shift(periods=-1)

# drop the last row
full_data = full_data.drop(full_data.index[-1])
# Formatting foreign volatility index as input data

def correspond_foreign_vol(date,data):
    """find the price of the date that is most recent to the given 'date' in the 'data'"""
    while True:
        date = date - pd.Timedelta('1 Day')  # go back one day 
        result_series = data['Price'].loc[data['Date']==date]  # find row('price' specifically) in the 'data' that matches with the 'date'
        if not result_series.empty:  # if not empty (which means there is a row that matches with the 'date')
            result_series = result_series.reset_index()  # reset index
            result_value = result_series['Price'][0]  # and bring the value
            return result_value

# Apply function
full_data['CBOE'] = full_data['Date'].apply(correspond_foreign_vol,data=CBOE_Volatility)
full_data['HSI'] = full_data['Date'].apply(correspond_foreign_vol,data=HSI_Volatility)
full_data['Nikkei'] = full_data['Date'].apply(correspond_foreign_vol,data=Nikkei_Volatility)
# Estimate 'historical volatility' using GARCH model

import scipy.optimize as optimize 

# Calculate return
KOSPI200_yesterday = org_data['KOSPI200'].shift(periods=1)  # Series made up of shifted KOSPI200
return_array = (org_data['KOSPI200']-KOSPI200_yesterday) / KOSPI200_yesterday  # Calculate return
# Set VKOSPI as an initial variance for garch

# 'Variance' represents (simga^2) for 1 day.
# 'VKSOPI' is an index of (simga) for 1 year expressed in percentage(%).
# Therefore, when assuming 252 trading days per year, 
# VKOSPI = sqrt(Variance * 252) * 100

initial_vkospi = full_data['VKOSPI'][1] 
initial_variance = initial_vkospi*initial_vkospi/2520000 
# GARCH model
def garch_forward(return_rate,variance,coefficients):
    ''' data type: float, float, 1d array(length=3)
    
    calculate variance of the next day based on GARCH
    '''
    # Coefficients
    alpha,beta,omega = coefficients
    # Calculate
    return omega + alpha*return_rate*return_rate + beta*variance
# Function for optimization

def garch_for_optimization(array): 
    ''' data type: 1d array(length=3)
    
    Maximazing the probability of the data occuring, assuming that 
    the return of asset(KOSPI200 in this case) is normal with zero mean.
    The variable 'probability' here is the logarithm of probability explained above. 
    The optimzer seaches over alpha,beta,and omega to maximize the sum of probability.
    '''
    # Coeffcients
    alpha,beta,omega = array
    
    # Variables
    sum_probability = 0
    variance = initial_variance
    
    for i in range(1,return_array_train.shape[0]):  # exclude the first value because it's nan.
        return_rate = return_array_train[i]  # set return rate
        # in case something goes wrong
        if variance<=0:
            print("Negative variance")
            break
        
        # calculate probability for optimization
        probability = -np.log(variance) - return_rate * return_rate / variance
        sum_probability += probability
        
        # calculate next day's variance 
        variance = garch_forward(return_rate,variance,array)
   
    return -sum_probability # because probability needs to be maximized
# Executing this cell will definitely take some time!!! (load the pickle instead)
# Optimize
#
# Used NYU V-Lab's estimate (in Feb 15,2020) for initial guess. 
# Why not use V-Lab's estimate? Because the coefficients can be different for different time period.
#
# Bounds on coefficients
# 0<=alpha<=1
# 0<=beta<=1
# 0<=omega
    
bounds = optimize.Bounds([0,0,0],[1,1,np.inf])
initial_guess = [0.14,0.76,2.97]  # V-Lab's estimate

# Trust-constr performed the best 
optimize_res_trust = optimize.minimize(garch_for_optimization,initial_guess,method='trust-constr',bounds=bounds)

# Pickle the result since it takes too much time
with open('../input/optimize_res_trust.pkl','wb') as file:
    pickle.dump(optimize_res_trust,file)
# Load the pickle: optimize_res_trust
with open('/kaggle/input/optimize-res-trust/optimize_res_trust.pkl','rb') as file:
    optimize_res_trust = pickle.load(file)
print('optimization result:')
print('alpha =',optimize_res_trust.x[0])
print('beta =',optimize_res_trust.x[1])
print('omega =',optimize_res_trust.x[2])
# Estimate historical volatility with estimated coefficients
#
# 'Variance' represents (simga^2) for 1 day.
# 'VKSOPI' is an index of (simga) for 1 year expressed in percentage(%).
# Therefore, when assuming 252 trading days per year, 
# VKOSPI = sqrt(Variance * 252) * 100

variance_array = np.zeros(full_data.shape[0],)  # Create array to store variance

# Values to be pre-assigned
variance_array[0] = np.nan  # Historical volatility cannot be estimated for the first one in full_data (06/03/2009)
variance_array[1] = initial_variance 

# Calculate historical volatilities using GARCH
for i in range(2,full_data.shape[0]):
    variance_array[i]=garch_forward(return_array[i-1],variance_array[i-1],optimize_res_trust.x)
    
# Adjust value to compare with VKOSPI (detail mentioned at the very top)  
historical_volatility = np.sqrt(variance_array * 252) * 100

# Add to the dataset
full_data['Historical Volatility'] = historical_volatility
full_data = full_data.dropna()  # Drop NA (because of nan in historical volatility)
# Before visualizing, few matplotlib settings
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['axes.labelpad'] = 5.5  # space between axis and label

mpl.rcParams['legend.fontsize'] = 'x-large'

# and seaborn settings
sns.set_style('darkgrid')
# Compare with VKOSPI (Visualization)
pd.plotting.register_matplotlib_converters()  # because of compatilbility issue with pd.Timestamp and matplotlib

fig, ax = plt.figure(figsize=(20,7)), plt.axes()
sns.lineplot(data=full_data,x='Date',y='VKOSPI',label='VKOSPI',ax=ax)
sns.lineplot(data=full_data,x='Date',y='Historical Volatility',label='Historical Volatility',ax=ax)
plt.ylabel('Volatility')
plt.show()
# Calculate error
abs_garch = abs(full_data['VKOSPI']-full_data['Historical Volatility'])
square_garch = (full_data['VKOSPI']-full_data['Historical Volatility'])**2

print('Total set')
print('MAE:',abs_garch.mean())
print('RMSE:',square_garch.mean()**0.5)

print('\nTest set ([-516:])')
print('MAE:',abs_garch[-516:].mean())
print('RMSE:',square_garch[-516:].mean()**0.5)
### Split data into train data and test data
split_ratio = 0.8

data = full_data.set_index('Date')
len_data = full_data.shape[0]

train_org = full_data.iloc[:int(len_data*split_ratio),]
test_org = full_data.iloc[int(len_data*split_ratio):,]

print('train set',train_org.shape)
print('test set',test_org.shape)
# 1. Day of a week
barplot = sns.barplot(x='Day_of_a_week',y='VKOSPI',data=train_org,ci=95,order=['Mon','Tue','Wed','Thu','Fri']) 
barplot.set_ylim((16,18))
plt.show()
# 2. Foreign volatility indices
corr = train_org.loc[:,['VKOSPI','CBOE','HSI','Nikkei']].corr()

# heatmap
mask = np.zeros_like(corr)
for i in range(mask.shape[0]):
    mask[i,i]=True
sns.heatmap(corr,annot=True,mask=mask,cmap='coolwarm',vmin=-1,vmax=1)
plt.show()
# lmplot 
sns.lmplot(x='Day_till_expiration',y='VKOSPI',data=train_org)
plt.show()

# Too bizarre when VKOSPI is small. Mean?
print('Too bizarre when VKOSPI is small. Mean?')
mean_by_dtexp = train_org.groupby('Day_till_expiration').mean()['VKOSPI']
sns.scatterplot(x=mean_by_dtexp.index,y=mean_by_dtexp.values)
plt.ylabel('Mean of VKOSPI')
plt.show()

# Reasonable to suspect that there aren't much data with very high day_till_expiration
print('Reasonable to suspect that there aren\'t much data with very high day_till_expiration')
len_by_dtexp = train_org['Day_till_expiration'].value_counts()
plt.figure(figsize=(15,5))
plt.xlabel('Day_till_expiration')
plt.ylabel('Frequency of data')
sns.barplot(x=len_by_dtexp.index,y=len_by_dtexp.values)
plt.show()

print('The number of data seems insufficient when Day_till_expiration is high')
# lmplot only with reasonable ones
def slice_and_anaylze(slice_at):
    # slice dataframe till the given parameter(slice_at)
    print('lmplot only with reasonable ones (till {})'.format(slice_at))
    mean_by_dtexp = full_data.groupby('Day_till_expiration').mean()['VKOSPI']
    mean_by_dtexp = mean_by_dtexp[:slice_at]  # slice
    mean_by_dtexp = pd.DataFrame({'Day_till_expiration':mean_by_dtexp.index,
                                  'Mean_of_VKOSPI':mean_by_dtexp.values})  # Convert series to dataframe (for sns plot)
    
    # lmplot
    sns.lmplot(x='Day_till_expiration',y='Mean_of_VKOSPI',data=mean_by_dtexp)
    plt.show()

    # correlation test (pearson)
    corr_coef = stats.pearsonr(mean_by_dtexp['Day_till_expiration'],mean_by_dtexp['Mean_of_VKOSPI'])
    print('Pearson correlation test')
    print('='*50)
    print('correlation coefficient:',corr_coef[0])
    print('2-tailed p-value:', corr_coef[1])
    
# 24 vs 19...
slice_and_anaylze(19)
slice_and_anaylze(24)
# Plot each market variable with VKOSPI

# Columns to plot 
plot_columns = ['KOSPI200','Open_interest','For_KOSPI_Netbuying_Amount','For_Future_Netbuying_Quantity',
                'For_Call_Netbuying_Quantity','For_Put_Netbuying_Quantity','Indiv_Future_Netbuying_Quantity',
                'Indiv_Call_Netbuying_Quantity','Indiv_Put_Netbuying_Quantity','PCRatio']

# plot
fig, axes = plt.subplots(10,2,figsize=(25,75))
for i,col in enumerate(plot_columns):
    sns.regplot(col,'VKOSPI',data=train_org, ax=axes[i,0]) # regression plot
    sns.kdeplot(train_org[col],train_org['VKOSPI'],shade=True, ax=axes[i,1]) # kernel density estimate plot
  
# Correlation matrix
corr = train_org[plot_columns].corr()

mask = np.zeros_like(corr)
for i in range(mask.shape[0]):
    mask[i,i]=True
plt.figure(figsize=(12,5))
sns.heatmap(corr,annot=True,mask=mask,cmap='coolwarm',vmin=-1,vmax=1)
plt.show()
# Plot variables with high correlation coefficient

def cor_reg_kde(x,y,data):
    # Correlation coefficient
    print('correlation coefficient:',corr.loc[x,y])
    
    # Subplot adjust
    plt.subplots_adjust(wspace=0.5)
    
    # Plot (regression and kernel density)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    sns.regplot(x,y,data=data,ax=ax1)
    sns.kdeplot(data[x],data[y],shade=True,ax=ax2)
    
cor_reg_kde('Indiv_Future_Netbuying_Quantity','For_Future_Netbuying_Quantity',train_org)
cor_reg_kde('Indiv_Put_Netbuying_Quantity','Open_interest',train_org)
cor_reg_kde('Indiv_Put_Netbuying_Quantity','Indiv_Call_Netbuying_Quantity',train_org)
cor_reg_kde('Indiv_Call_Netbuying_Quantity','Open_interest',train_org)
# Do not execute
# pickle the final dataset
with open('/kaggle/working/train_org.pkl','wb') as file:
    pickle.dump(train_org,file)
with open('/kaggle/working/test_org.pkl','wb') as file:
    pickle.dump(test_org,file)
# Unnecessary if the cells above are executed
# load pickle
with open('/kaggle/input/dataset/train_org.pkl','rb') as file:
    train_org = pickle.load(file)
with open('/kaggle/input/dataset/test_org.pkl','rb') as file:
    test_org = pickle.load(file)

# Standardize (Normalize)

# Keep the original
train = train_org.copy()
test = test_org.copy()

# Variables that has to be standardized
to_standardize = ['VKOSPI', 'KOSPI200', 'Open_interest',
                  'For_KOSPI_Netbuying_Amount', 'For_Future_Netbuying_Quantity',
                  'For_Call_Netbuying_Quantity', 'For_Put_Netbuying_Quantity',
                  'Indiv_Future_Netbuying_Quantity', 'Indiv_Call_Netbuying_Quantity',
                  'Indiv_Put_Netbuying_Quantity', 'PCRatio', 'Day_till_expiration',
                  'CBOE','Historical Volatility'
                 ]

# Standardization is based on the mean and std of train data
mean_train = train[to_standardize].mean()
std_train = train[to_standardize].std()

# Standardize
train[to_standardize] = (train[to_standardize]-mean_train)/std_train
test[to_standardize] = (test[to_standardize]-mean_train)/std_train
# Data for neural network
# Input variables

# Ver 1
# 'EVERYTHING'
input_var = ['KOSPI200', 'Open_interest',
             'For_KOSPI_Netbuying_Amount', 'For_Future_Netbuying_Quantity',
             'For_Call_Netbuying_Quantity', 'For_Put_Netbuying_Quantity',
             'Indiv_Future_Netbuying_Quantity','Indiv_Call_Netbuying_Quantity',
             'Indiv_Put_Netbuying_Quantity', 'PCRatio', 
             'Day_till_expiration', 'CBOE', 
             'Historical Volatility']  
input_ver = '1, EVERYTHING'
print('total',len(input_var))
# Ver 2
# 'Conservative'
input_var = ['KOSPI200', 'Open_interest',
             'Day_till_expiration', 'CBOE', 
             'Historical Volatility'] 
input_ver = '2, Conservative'
print('total',len(input_var))
# Ver 3
# 'Foreigners-believer'
input_var = ['KOSPI200', 'Open_interest',
             'For_KOSPI_Netbuying_Amount', 'For_Future_Netbuying_Quantity',
             'For_Call_Netbuying_Quantity', 'For_Put_Netbuying_Quantity',
             'PCRatio', 'Day_till_expiration', 'CBOE', 
             'Historical Volatility']
input_ver = '3, Foreigners-believer'
print('total',len(input_var))
# Data only including input variables and target data
x_train = train[input_var]
y_train = train['VKOSPI']
x_test = test[input_var]
y_test = test['VKOSPI']
print(input_ver)
### NN Class
class NN:
    def __init__(self,hid_nodes,activation1,activation2,epochs,loss,metrics):
        '''Set hyperparameters'''
        self.HID_NODES = hid_nodes
        self.ACTIVATION1 = activation1
        self.ACTIVATION2 = activation2
        self.EPOCHS = epochs
        self.LOSS = loss
        self.METRICS = metrics
        
    def train(self):
        '''Train a model'''
        # Build
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=self.HID_NODES,
                                     activation=self.ACTIVATION1,
                                     input_shape=(x_train.shape[1],)))
        model.add(keras.layers.Dense(units=1,
                                     activation=self.ACTIVATION2))
        
        
        # Compile
        model.compile(optimizer='rmsprop',
                      loss=self.LOSS,
                      metrics=self.METRICS)
        
        # Fit
        self.history = model.fit(x_train,y_train,
                                 batch_size=32,
                                 epochs=self.EPOCHS,
                                 validation_split=0.2,
                                 verbose=0).history
        
        # Model into variable
        self.model = model
    
    def calculate_error(self):
        '''Calculate errors: loss and RMSE of train set and validation set'''
        # Scale-back functions
        def fun_mae(x): return x*std_train['VKOSPI']
        def fun_rmse(x): return (x*std_train['VKOSPI']*std_train['VKOSPI'])**0.5
        
        # Scale back
        if (self.LOSS == 'MAE') & (self.METRICS == ['MSE']) :
            self.loss = list(map(fun_mae,self.history['loss']))
            self.metric = list(map(fun_rmse,self.history['MSE']))
            self.val_loss = list(map(fun_mae,self.history['val_loss']))
            self.val_metric = list(map(fun_rmse,self.history['val_MSE']))
            
        elif (self.LOSS == 'MSE') & (self.METRICS == ['MAE']):
            self.loss = list(map(fun_rmse,self.history['loss']))
            self.metric = list(map(fun_mae,self.history['MAE']))
            self.val_loss = list(map(fun_rmse,self.history['val_loss']))
            self.val_metric = list(map(fun_mae,self.history['val_MAE']))
        
        else:
            raise Exception('Loss is neither MSE nor MAE. Cannot scale back properly!! :(')
                
    def plot_training(self):
        '''Plot the change in error during training'''
        # Warning Note
        print('Note that RMSE will be represented instead of MSE.')
        
        
        fig,ax = plt.figure(figsize=(14,7)), plt.axes()
        x=np.linspace(1,self.EPOCHS,num=self.EPOCHS)

        # Plot
        ax.plot(x, self.loss, label='loss', marker='.',markersize=20,color='b')
        ax.plot(x, self.metric, label='metric', marker='.',markersize=20,ls='--',color='r')
        ax.plot(x, self.val_loss, label='val loss', marker='*',markersize=20,color='g')
        ax.plot(x, self.val_metric, label='val metric', marker='.',markersize=20,ls='--',color='y')

        # Setting
        plt.ylim(0,)
        plt.legend(numpoints=3,markerscale=0.5)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.xticks(x)
        
        plt.show()
        
    def predict(self):
        self.mae,self.mse = self.model.evaluate(x_test,y_test,verbose=0)  # note that these errors are not scaled back.
        self.prediction = self.model.predict(x_test)
### Train NN
tf.random.set_seed(18)  # to ensure reproducibility
nn_1 = NN(hid_nodes=10,
             activation1='tanh',
             activation2='linear',
             epochs=4,
             loss='MAE',
             metrics=['MSE'])   # RMSE will be represented instead.

nn_1.train()
nn_1.calculate_error()
nn_1.plot_training()

print('MAE in train set:', nn_1.loss[-1])
print('RMSE in train set:', nn_1.metric[-1])
print('MAE in validation set:', nn_1.val_loss[-1])
print('RMSE in validation set:', nn_1.val_metric[-1])
## RNN(GRU) Class
class RNN(NN):  # inherit nn 
    ''' Most parts are the same to NN class. 
    However, 'x' need to be reshaped to be fed into RNN.
    Variables: 'x_train_rnn','x_test_rnn' is the alternative used in RNN.
    '''
    def __init__(self,units,epochs,method,loss,metrics):
        self.UNITS = units
        self.EPOCHS = epochs
        self.METHOD = method
        self.LOSS = loss
        self.METRICS = metrics
        
    
    def train(self):
        # Build
        model = keras.Sequential()
        if self.METHOD == 'LSTM':
            model.add(keras.layers.LSTM(units=self.UNITS,
                                        input_shape=(1,x_train.shape[1]))) # input_shape is 2D, (time steps, features)
        elif self.METHOD == 'GRU':
            model.add(keras.layers.GRU(units=self.UNITS,
                                       input_shape=(1,x_train.shape[1]))) # input_shape is 2D, (time steps, features)
  
        model.add(keras.layers.Dense(3,activation='tanh'))
        model.add(keras.layers.Dense(1,activation='linear'))
        
        # Compile
        model.compile(optimizer = 'rmsprop',
                      loss = self.LOSS,
                      metrics = self.METRICS)
        
        # Fit
        self.history = model.fit(x_train_rnn,y_train,
                                 batch_size=32,
                                 epochs=self.EPOCHS,
                                 validation_split=0.2,
                                 verbose=0).history
        
        # Model into variable
        self.model = model

    def predict(self):
        self.mae,self.mse = self.model.evaluate(x_test_rnn,y_test,verbose=0)  # note that these errors are not scaled back.
        self.prediction = self.model.predict(x_test_rnn)
        
        
# Train RNN
tf.random.set_seed(20)

# Data reshape for rnn
x_train_rnn = x_train.to_numpy().reshape((x_train.shape[0],1,x_train.shape[1]))

rnn_1 = RNN(units=6,
            epochs=5,
            method='GRU',
            loss='MAE',
            metrics=['MSE'])  # RMSE will be represented instead.

rnn_1.train()
rnn_1.calculate_error()
rnn_1.plot_training()

print('MAE in train set:', rnn_1.loss[-1])
print('RMSE in train set:', rnn_1.metric[-1])
print('MAE in validation set:', rnn_1.val_loss[-1])
print('RMSE in validation set:', rnn_1.val_metric[-1])
# Train RNN, having MSE as loss function
tf.random.set_seed(17)

# Data reshape for rnn
x_train_rnn = x_train.to_numpy().reshape((x_train.shape[0],1,x_train.shape[1]))

rnn_1 = RNN(units=5,
            epochs=15,
            method='GRU',
            loss = 'MSE',    # RMSE will be represented instead.
            metrics = ['MAE'])

rnn_1.train()
rnn_1.calculate_error()
rnn_1.plot_training()

print('RMSE in train set:', rnn_1.loss[-1])
print('MAE in train set:', rnn_1.metric[-1])
print('RMSE in validation set:', rnn_1.val_loss[-1])
print('MAE in validation set:', rnn_1.val_metric[-1])
# Predict(with loop) and compare - preparation

# Function to reshape the keras.Model.prediction
def reshape(df):
    predicted_VKOSPI= np.zeros(df.shape[0])
    for i,n in enumerate(df):
        predicted_VKOSPI[i] = n[0]
    return predicted_VKOSPI # return dataframe

# Prediction loop
def ml_loop(model,iter): 
    ''' param: model: an object(model) which has a method 'train' and 'predict', iter: int
    Predict the value with iteration. 
    Iteration is done because the prediction may vary because of the randomness. 
    '''
    # Create mean_prediction dataframe
    model.train()
    model.predict()
    sum_prediction = reshape(model.prediction) * std_train['VKOSPI'] + mean_train['VKOSPI']  # scale back
    
    # loop
    for n in range(iter-1):
        model.train()
        model.predict()
        sum_prediction += reshape(model.prediction) * std_train['VKOSPI'] + mean_train['VKOSPI']       
    return sum_prediction/iter #mean

# Original data
real_VKOSPI = test_org['VKOSPI']
HV = test_org['Historical Volatility']

mse_hv = abs(real_VKOSPI-HV).mean()
rmse_hv = (real_VKOSPI - HV)**2
rmse_hv = rmse_hv.mean()**0.5
# Predict and compare - execution (nn)
model=nn_1
iter=50
set_seed = 0

# predict
if set_seed:
    tf.random.set_seed(18) # reproducible result. 
predicted_VKOSPI = ml_loop(model,iter)
# Show result (NN)  
# plot
fig, ax = plt.figure(figsize=(20,7)), plt.axes()
ax.plot(test_org['Date'],real_VKOSPI,label='VKOSPI')
ax.plot(test_org['Date'],predicted_VKOSPI,label='GARCH-NN')
ax.plot(test_org['Date'],HV,label='Historical Volatility (GARCH)')
plt.legend()
plt.show()

# MAE
mae_predict = abs(real_VKOSPI - predicted_VKOSPI).mean()

# RMSE
rmse_predict = (real_VKOSPI - predicted_VKOSPI)**2
rmse_predict = rmse_predict.mean()**0.5

# prediction result
print('MAE:',mse_predict,'\nError {}% reduced'.format((mse_hv-mse_predict)/mse_hv))
print('RMSE:',rmse_predict,'\nError {}% reduced'.format((rmse_hv-rmse_predict)/rmse_hv))
# Predict and compare - execution (rnn)
model = rnn_1
iter=50
set_seed_rnn = 0

# data reshape for rnn
x_test_rnn = x_test.to_numpy().reshape((x_test.shape[0],1,x_test.shape[1]))

# predict
if set_seed_rnn:
    tf.random.set_seed(18) # reproducible result. not recommended when trying to gain precise prediction
predicted_VKOSPI = ml_loop(model,iter)
##### Show result (RNN)
# plot
fig, ax = plt.figure(figsize=(20,7)), plt.axes()
ax.plot(test_org['Date'],real_VKOSPI,label='Real VKOSPI')
ax.plot(test_org['Date'],predicted_VKOSPI,label='GARCH-RNN')
ax.plot(test_org['Date'],HV,label='Historical Volatility (GARCH)')
plt.legend()
plt.show()

# MAE
mae_predict = abs(real_VKOSPI - predicted_VKOSPI).mean()

# RMSE
rmse_predict = (real_VKOSPI - predicted_VKOSPI)**2
rmse_predict = rmse_predict.mean()**0.5

# prediction result
print('')
print('MAE:',mse_predict,'\nError {}% reduced'.format((mse_hv-mse_predict)/mse_hv))
print('RMSE:',rmse_predict,'\nError {}% reduced'.format((rmse_hv-rmse_predict)/rmse_hv))