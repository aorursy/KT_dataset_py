#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# load up the libraries etc.

#===========================================================================

import pandas  as pd

import numpy   as np

import matplotlib.pyplot as plt

import matplotlib.ticker as plticker

from sklearn.metrics import mean_squared_log_error

pd.options.mode.chained_assignment = None 
#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# also, read in the 'solution' data 

#===========================================================================

solution   = pd.read_csv('../input/house-prices-advanced-regression-solution-file/submission.csv')

y_true     = solution["SalePrice"]



#===========================================================================

# select only ONE feature: OverallQual

#===========================================================================

features = ['OverallQual']



#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]

final_X_test  = test_data[features]
fig, ax = plt.subplots(figsize=(15, 5))

plt.scatter(X_train,y_train)

loc = plticker.MultipleLocator(base=1.0)

ax.xaxis.set_major_locator(loc)

ax.set_title ("House Prices data", fontsize=18)

ax.set_xlabel ("OverallQual", fontsize=18)

ax.set_ylabel ("SalePrice", fontsize=18);
# The model:

mean_price = y_train.mean()

print("The mean price is", mean_price)

final_X_test.loc[:,'y_pred'] = mean_price

# calculate the score

RMSLE = np.sqrt( mean_squared_log_error(y_true, final_X_test['y_pred']) )

print("The score is %.5f" % RMSLE )



#===========================================================================

# compare the model to the data

fig, ax = plt.subplots(figsize=(15, 5))

# plot the training data

plt.scatter(X_train,y_train)

# now plot the results of our model

plt.plot(final_X_test['OverallQual'],final_X_test['y_pred'],color='orange',linewidth=3)

ax.set_title ("House Prices data", fontsize=18)

ax.set_xlabel ("OverallQual", fontsize=18)

ax.set_ylabel ("SalePrice", fontsize=18);
# The model:

fit = (np.polyfit(X_train['OverallQual'], y_train, 1 ))

c = fit[1]

m = fit[0]

final_X_test.loc[:,'y_pred'] = (m*final_X_test['OverallQual'] + c)

# set any negative prices to be zero:

final_X_test.loc[final_X_test.y_pred < 0, 'y_pred'] = 0

# calculate the score

RMSLE = np.sqrt( mean_squared_log_error(y_true, final_X_test['y_pred']) )

print("The score is %.5f" % RMSLE )



#===========================================================================

# compare the model to the data

fig, ax = plt.subplots(figsize=(15, 5))

# plot the training data

plt.scatter(X_train,y_train)

# now plot the results of our model

x = np.linspace(0,10,100)

y = m*x + c

plt.plot(x,y,color='orange',linewidth=3)

#plt.scatter(final_X_test['OverallQual'],final_X_test['y_pred'],color='orange',s=75)

ax.set_title ("House Prices data", fontsize=18)

ax.set_xlabel ("OverallQual", fontsize=18)

ax.set_ylabel ("SalePrice", fontsize=18)

ax.set_ylim(ymin=0);
# The model:

fit = (np.polyfit(X_train['OverallQual'], y_train, 2 ))

c = fit[2]

b = fit[1]

a = fit[0]

final_X_test.loc[:,'y_pred'] = (a*final_X_test['OverallQual']**2 +b*final_X_test['OverallQual'] + c)

# calculate the score

RMSLE = np.sqrt( mean_squared_log_error(y_true, final_X_test['y_pred']) )

print("The score is %.5f" % RMSLE )



#===========================================================================

# compare the model to the data

fig, ax = plt.subplots(figsize=(15, 5))

# plot the training data

plt.scatter(X_train,y_train)

# now plot the results of our model

y = a*x**2 + b*x + c

plt.plot(x,y,color='orange',linewidth=3)

#plt.scatter(final_X_test['OverallQual'],final_X_test['y_pred'],color='orange',s=75)

ax.set_title ("House Prices data", fontsize=18)

ax.set_xlabel ("OverallQual", fontsize=18)

ax.set_ylabel ("SalePrice", fontsize=18);
# The model:

fit = (np.polyfit(X_train['OverallQual'], np.log(y_train), 1))

A = np.exp(fit[1])

B = fit[0]

final_X_test.loc[:,'y_pred'] = (A*np.exp(B*final_X_test['OverallQual']))

# calculate the score

RMSLE = np.sqrt( mean_squared_log_error(y_true, final_X_test['y_pred']) )

print("The score is %.5f" % RMSLE )



#===========================================================================

# compare the model to the data

fig, ax = plt.subplots(figsize=(15, 5))

# plot the training data

plt.scatter(X_train,y_train)

# now plot the results of our model

y = A*np.exp(B*x)

plt.plot(x,y,color='orange',linewidth=3)

#plt.scatter(final_X_test['OverallQual'],final_X_test['y_pred'],color='orange',s=75)

ax.set_title ("House Prices data", fontsize=18)

ax.set_xlabel ("OverallQual", fontsize=18)

ax.set_ylabel ("SalePrice", fontsize=18);
#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({"Id":test_data.Id, "SalePrice":final_X_test['y_pred']})

output.to_csv('submission.csv', index=False)