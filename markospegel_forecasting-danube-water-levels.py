## Let's start with loading all dependencies



import numpy as np 

from numpy import concatenate



import pandas as pd

from pandas import read_csv

from pandas import DataFrame



import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D



import sklearn 

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression



import keras 

from keras.layers import Dense

from keras.layers import Input, LSTM

from keras.models import Model

from keras.models import load_model

import h5py



import os # accessing directory structure



%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = None # specify 'None' if want to read whole file

df = pd.read_csv('/kaggle/input/danube-daily-level-in-cm-kienstock-20022019/danube-waterlevel-Kienstock_2002-2019.csv', delimiter=',', nrows = nRowsRead)

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 1000]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (3 * nGraphPerRow, 2 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    # plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
# we use a subset of the data (first year) to get a first idea

df_plot = df.iloc[:365,1:4]

df_plot.dataframeName = 'danube-waterlevel-Kienstock_2002-2019.csv'
plotPerColumnDistribution(df_plot,3,3)
plotScatterMatrix(df_plot,9,10)
plotCorrelationMatrix(df_plot,4)
df1 = df.loc[:, ["date", "max"]]
df1_plot = df1.iloc[:365,1:2].values.astype(float)

plt.figure(num=None, figsize=(10,4))

plt.plot(df1_plot, color = 'blue')

plt.title('Danube Water Level @ Kienstock 2002')

plt.xlabel('Time (Days)')

plt.ylabel('max level in cm')

plt.show()
df1_plot = df1.iloc[:,1:2].values.astype(float)

plt.figure(num=None, figsize=(10,4))

plt.plot(df1_plot, color = 'blue')

plt.title('Danube Water Level @ Kienstock 2002-2019')

plt.xlabel('Time (Days)')

plt.ylabel('max level in cm')

plt.show()
missing_data = df1.isnull()
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("") 
for i in range (0, len(df1)):

    if missing_data.iloc[i,1]:

        df1.iloc[i,1] = (df1.iloc[i-1,1] + df1.iloc[i+1,1]) / 2

        print('Index: ', i, '  New value: ', df1.iloc[i,1])
# defining our parameters as in the oil price prediction example

test_share = 0.1 # test set share of data set

timesteps = 30 # size of time window for predictions

batchsize = 64 # important for LSTM
# setting data set lengths

print(f'nRow: {nRow}')# total data set size

max_train = int(nRow * (1 - test_share))

length_train = max_train - 2 * timesteps # subtract window size (twice for LSTM training)

length_train = length_train - int(length_train % batchsize) # training set size must be multiples of batchsize

nBatch_train = length_train / batchsize

upper_train = length_train + 2 * timesteps

print(f'upper_train: {upper_train}')

print(f'length_train: {length_train}')

print(f'nBatch_train: {nBatch_train}')



max_test = nRow - upper_train

length_test = max_test - timesteps # subtract window size

length_test = length_test - int(length_test % batchsize) # training set size must be multiples of batchsize

nBatch_test = length_test / batchsize

upper_test = length_test + timesteps

print(f'upper_test: {upper_test}')

print(f'length_test: {length_test}')

print(f'nBatch_test: {nBatch_test}')
# creating data sets with feature scaling between 0 and 1.



sc = MinMaxScaler(feature_range = (0, 1))

total_set = df1.iloc[:,1:2].values

total_set_scaled = sc.fit_transform(np.float64(total_set))



training_set = total_set[0:upper_train]

print(training_set.shape)



training_set_scaled = total_set_scaled[0:upper_train]

# print(training_set_scaled.shape)



test_set = total_set[upper_train:upper_train+upper_test]

print(test_set.shape)



test_set_scaled = total_set_scaled[upper_train:upper_train+upper_test]

# print(test_set_scaled.shape)
# Real water levels for comparison to predictions

y_train = training_set[timesteps:length_train+timesteps]

print(y_train.shape)

y_test = test_set[timesteps:length_test+timesteps]

print(y_test.shape)  
# Creating training data structures with windows

lstm_X_train_scaled = []

lstm_Y_train_scaled = []



for i in range(timesteps, length_train+timesteps): 

    lstm_X_train_scaled.append(training_set_scaled[i-timesteps:i,0])

    lstm_Y_train_scaled.append(training_set_scaled[i:i+timesteps,0])



lstm_X_train_scaled = np.asarray(lstm_X_train_scaled)

lstm_Y_train_scaled = np.asarray(lstm_Y_train_scaled)

lstm_X_train_scaled = np.reshape(lstm_X_train_scaled, (lstm_X_train_scaled.shape[0], lstm_X_train_scaled.shape[1], 1))

lstm_Y_train_scaled = np.reshape(lstm_Y_train_scaled, (lstm_Y_train_scaled.shape[0], lstm_Y_train_scaled.shape[1], 1))



print(lstm_X_train_scaled.shape)

# print(lstm_Y_train_scaled.shape)
# Creating test data structures

lstm_X_test_scaled = []



for i in range(timesteps, length_test+timesteps): 

    lstm_X_test_scaled.append(test_set_scaled[i-timesteps:i,0])



lstm_X_test_scaled = np.asarray(lstm_X_test_scaled)

lstm_X_test_scaled = np.reshape(lstm_X_test_scaled, (lstm_X_test_scaled.shape[0], lstm_X_test_scaled.shape[1], 1))



print(lstm_X_test_scaled.shape)

# print(lstm_X_test_scaled[0:3])    
# Initialising the LSTM Model with MSE Loss-Function using Functional API



inputs1_1 = Input(batch_shape=(batchsize,timesteps,1))

lstm1_1 = LSTM(10, stateful=True, return_sequences=True)(inputs1_1)

lstm1_2 = LSTM(10, stateful=True, return_sequences=True)(lstm1_1)

output1_1 = Dense(units = 1)(lstm1_2)



lstm1_reg = Model(inputs=inputs1_1, outputs = output1_1)



#adam is fast starting off and then gets slower and more precise

#mse -> mean sqare error loss function

lstm1_reg.compile(optimizer='adam', loss = 'mse')

lstm1_reg.summary()
epochs = 30

print("Number of epochs: ", epochs)



for i in range(epochs):

    print("Epoch: " + str(i))

    #run through all data but the cell, hidden state are used for the next batch.

    lstm1_reg.fit(lstm_X_train_scaled, lstm_Y_train_scaled, shuffle=False, epochs = 1, batch_size = batchsize)

    #resets only the states but the weights, cell and hidden are kept.

    lstm1_reg.reset_states()
# save the model

lstm1_reg.save(filepath="lstm_with_mse_30_ts.h5")
#load model

# lstm1_reg = load_model(filepath="lstm_with_mse_30_ts.h5")
# get predicted data on the test set



lstm_Y_test_pred_scaled = lstm1_reg.predict(lstm_X_test_scaled, batch_size=batchsize)

lstm1_reg.reset_states()

# print(lstm_Y_test_pred_scaled.shape)



lstm_y_test_pred_scaled = lstm_Y_test_pred_scaled[:,-1,:] # keep only value of last timestep

# print(lstm_y_test_pred_scaled.shape)



# inverse transform (reverse feature scaling)

lstm_y_test_pred = sc.inverse_transform(lstm_y_test_pred_scaled)
# calculate mean squared error on the test set predictions

lstm_test_residuals = lstm_y_test_pred - y_test

lstm1_test_mae = np.sum(np.fabs(lstm_test_residuals)) / len(lstm_test_residuals)

lstm1_test_rmse = np.sqrt(np.sum(np.power(lstm_test_residuals,2)) / len(lstm_test_residuals))

print('LSTM test MAE: ', lstm1_test_mae)

print('LSTM test RMSE: ', lstm1_test_rmse)
# Visualising the results for the whole test set

lo = 0

hi = len(y_test)

plt.figure(num=None, figsize=(10,4))

plt.plot(y_test[lo:hi], color = 'blue', label = 'Real Water Level')

plt.plot(lstm_y_test_pred[lo:hi], color = 'red', label = 'LSTM Predicted')

plt.title('Danube Level at Kienstock (km 2015.021)')

plt.xlabel('time in days')

plt.ylabel('max water level in cm')

plt.legend()

plt.show()
# redefining our parameters

timesteps = 2

batchsize = 4
# setting data set lengths

print(f'nRow: {nRow}')# total data set size

max_train = int(nRow * (1 - test_share))

length_train = max_train - 2 * timesteps # subtract window size (twice for LSTM training)

length_train = length_train - int(length_train % batchsize) # training set size must be multiples of batchsize

nBatch_train = length_train / batchsize

upper_train = length_train + 2 * timesteps

print(f'upper_train: {upper_train}')

print(f'length_train: {length_train}')

print(f'nBatch_train: {nBatch_train}')



max_test = nRow - upper_train

length_test = max_test - timesteps # subtract window size

length_test = length_test - int(length_test % batchsize) # training set size must be multiples of batchsize

nBatch_test = length_test / batchsize

upper_test = length_test + timesteps

print(f'upper_test: {upper_test}')

print(f'length_test: {length_test}')

print(f'nBatch_test: {nBatch_test}')
# creating data sets with feature scaling between 0 and 1.



sc = MinMaxScaler(feature_range = (0, 1))

total_set = df1.iloc[:,1:2].values

total_set_scaled = sc.fit_transform(np.float64(total_set))



training_set = total_set[0:upper_train]

print(training_set.shape)

# print(training_set_scaled[0:3])



training_set_scaled = total_set_scaled[0:upper_train]

# print(training_set_scaled.shape)

# print(training_set_scaled[0:3])



test_set = total_set[upper_train:upper_train+upper_test]

print(test_set.shape)

# print(test_set[0:3])



test_set_scaled = total_set_scaled[upper_train:upper_train+upper_test]

# print(test_set_scaled.shape)

# print(test_set_scaled[0:3])
# Real water levels for comparison to predictions



y_train = training_set[timesteps:length_train+timesteps]

print(y_train.shape)

# print(y_train[:3])

y_test = test_set[timesteps:length_test+timesteps]

print(y_test.shape)  

# print(y_test[:3])
# Creating training data structures with windows

lstm_X_train_scaled = []

lstm_Y_train_scaled = []



for i in range(timesteps, length_train+timesteps): 

    lstm_X_train_scaled.append(training_set_scaled[i-timesteps:i,0])

    lstm_Y_train_scaled.append(training_set_scaled[i:i+timesteps,0])



lstm_X_train_scaled = np.asarray(lstm_X_train_scaled)

lstm_Y_train_scaled = np.asarray(lstm_Y_train_scaled)

lstm_X_train_scaled = np.reshape(lstm_X_train_scaled, (lstm_X_train_scaled.shape[0], lstm_X_train_scaled.shape[1], 1))

lstm_Y_train_scaled = np.reshape(lstm_Y_train_scaled, (lstm_Y_train_scaled.shape[0], lstm_Y_train_scaled.shape[1], 1))



print(lstm_X_train_scaled.shape)

# print(lstm_X_train_scaled[:2])

# print(lstm_Y_train_scaled.shape)

# print(lstm_Y_train_scaled[:2])
# Creating test data structures

lstm_X_test_scaled = []



for i in range(timesteps, length_test+timesteps): 

    lstm_X_test_scaled.append(test_set_scaled[i-timesteps:i,0])



lstm_X_test_scaled = np.asarray(lstm_X_test_scaled)

lstm_X_test_scaled = np.reshape(lstm_X_test_scaled, (lstm_X_test_scaled.shape[0], lstm_X_test_scaled.shape[1], 1))



print(lstm_X_test_scaled.shape)

# print(lstm_X_test_scaled[0:3])    
# Initialising a simplified LSTM Model with MSE Loss-Function using Functional API



inputs_1 = Input(batch_shape=(batchsize,timesteps,1))

lstm_1 = LSTM(3, stateful=True, return_sequences=True)(inputs_1)

output_1 = Dense(units = 1)(lstm_1)



lstm_reg = Model(inputs=inputs_1, outputs = output_1)



#adam is fast starting off and then gets slower and more precise

#mse -> mean sqare error loss function

lstm_reg.compile(optimizer='adam', loss = 'mse')

lstm_reg.summary()
epochs = 30

print("Number of epochs: ", epochs)



#Statefull

for i in range(epochs):

    print("Epoch: " + str(i))

    #run through all data but the cell, hidden state are used for the next batch.

    lstm_reg.fit(lstm_X_train_scaled, lstm_Y_train_scaled, shuffle=False, epochs = 1, batch_size = batchsize)

    #resets only the states but the weights, cell and hidden are kept.

    lstm_reg.reset_states()

    

#Stateless

#between the batches the cell and hidden states are lost.

#regressor_mae.fit(X_train, y_train, shuffle=False, epochs = epochs, batch_size = batch_size)
# save the model

lstm_reg.save(filepath="lstm_with_mse_2_ts.h5")
#load model

# lstm_reg = load_model(filepath="lstm_with_mse_2_ts.h5")
# get predicted data on the test set



lstm_Y_test_pred_scaled = lstm_reg.predict(lstm_X_test_scaled, batch_size=batchsize)

lstm_reg.reset_states()

print(lstm_Y_test_pred_scaled.shape)



lstm_y_test_pred_scaled = lstm_Y_test_pred_scaled[:,-1,:] # keep only value of last timestep

print(lstm_y_test_pred_scaled.shape)



# inverse transform (reverse feature scaling)

lstm_y_test_pred = sc.inverse_transform(lstm_y_test_pred_scaled)
# calculate mean squared error on the test set predictions

lstm_test_residuals = lstm_y_test_pred - y_test

lstm_test_mae = np.sum(np.fabs(lstm_test_residuals)) / len(lstm_test_residuals)

lstm_test_rmse = np.sqrt(np.sum(np.power(lstm_test_residuals,2)) / len(lstm_test_residuals))

print('LSTM2 test MAE: ', lstm_test_mae)

print('LSTM2 test RMSE: ', lstm_test_rmse)

print('LSTM1 test MAE: ', lstm1_test_mae)

print('LSTM1 test RMSE: ', lstm1_test_rmse)
# Visualising the results

lo = 0

hi = len(y_test)

plt.figure(num=None, figsize=(10,4))

plt.plot(y_test[lo:hi], color = 'blue', label = 'Real Water Level')

plt.plot(lstm_y_test_pred[lo:hi], color = 'red', label = 'LSTM Predicted')

plt.title('Danube at Kienstock (km 2015.021)')

plt.xlabel('time in days')

plt.ylabel('max water level in cm')

plt.legend()

plt.show()
# Visualising the results for a couple of months

lo = 270

hi = min(330, len(y_test))

plt.figure(num=None, figsize=(10,4))

plt.plot(y_test[lo:hi], color = 'blue', label = 'Real Water Level')

plt.plot(lstm_y_test_pred[lo:hi], color = 'red', label = 'LSTM Predicted')

plt.title('Danube Level at Kienstock (km 2015.021)')

plt.xlabel('time in days')

plt.ylabel('max water level in cm')

plt.legend()

plt.show()
# Creating training data structures with windows for LinReg

# It is actually the same as the LSTM data structure, just shaped differently (one array dimension less)



lr_timesteps = timesteps # window size for LinReg, must not be larger that LSTM window size

lr_X_train_scaled = []



for i in range(timesteps, length_train+timesteps): 

    lr_X_train_scaled.append(training_set_scaled[i-lr_timesteps:i,0])



lr_X_train_scaled = np.asarray(lr_X_train_scaled)

lr_y_train_scaled = training_set_scaled[timesteps:length_train+timesteps]



print(lr_X_train_scaled.shape)

# print(lr_y_train_scaled.shape)
# Creating test data structure for LinReg

lr_X_test_scaled = []



for i in range(timesteps, length_test+timesteps): 

    lr_X_test_scaled.append(test_set_scaled[i-lr_timesteps:i,0])



lr_X_test_scaled = np.asarray(lr_X_test_scaled)



print(lr_X_test_scaled.shape)
# do the LinReg fit

lr_model = LinearRegression()

lr_model.fit(lr_X_train_scaled, lr_y_train_scaled)

# the regression coefficients

print ('Coefficients: ', lr_model.coef_)
# get predicted data on the test set

lr_y_test_pred_scaled = lr_model.predict(lr_X_test_scaled)

# print(lr_y_test_pred_scaled.shape)



# inverse transform (reverse feature scaling)

lr_y_test_pred = sc.inverse_transform(lr_y_test_pred_scaled)



# print(lr_y_test_pred.shape)
# calculate mean squared error on the test set predictions

lr_test_residuals = lr_y_test_pred - y_test

lr_test_mae = np.sum(np.fabs(lr_test_residuals)) / len(lr_test_residuals)

lr_test_rmse = np.sqrt(np.sum(np.power(lr_test_residuals,2)) / len(lr_test_residuals))

print('LinReg test MAE: ', lr_test_mae)

print('LinReg test RMSE: ', lr_test_rmse)

print('LSTM test MAE: ', lstm_test_mae)

print('LSTM test RMSE: ', lstm_test_rmse)
# Visualising the results for a couple of months

lo = 270

hi = min(330, len(y_test))

plt.figure(num=None, figsize=(10,4))

plt.plot(y_test[lo:hi], color = 'blue', label = 'Real Water Level')

plt.plot(lr_y_test_pred[lo:hi], color = 'green', label = 'LinReg Predicted')

plt.plot(lstm_y_test_pred[lo:hi], color = 'red', label = 'LSTM Predicted')

plt.title('Danube Level at Kienstock (km 2015.021)')

plt.xlabel('time in days')

plt.ylabel('max water level in cm')

plt.legend()

plt.show()
triv_y_test_pred = test_set[timesteps-1:length_test+timesteps-1]

print(triv_y_test_pred.shape)
# calculate mean squared error on the trivial predictions

triv_residuals = triv_y_test_pred - y_test

triv_mae = np.sum(np.fabs(triv_residuals)) / len(triv_residuals)

triv_rmse = np.sqrt(np.sum(np.power(triv_residuals,2)) / len(triv_residuals))

print('Trivial test MAE: ', triv_mae)

print('Trivial test RMSE:', triv_rmse)

print('LinReg test MAE: ', lr_test_mae)

print('LinReg test RMSE: ', lr_test_rmse)

print('LSTM test MAE: ', lstm_test_mae)

print('LSTM test RMSE: ', lstm_test_rmse)
# Visualising the results for a couple of months

lo = 270

hi = min(330, len(y_test))

plt.figure(num=None, figsize=(10,4))

plt.plot(y_test[lo:hi], color = 'blue', label = 'Real Water Level')

plt.plot(lr_y_test_pred[lo:hi], color = 'green', label = 'LinReg Predicted')

plt.plot(lstm_y_test_pred[lo:hi], color = 'red', label = 'LSTM Predicted')

plt.plot(triv_y_test_pred[lo:hi], color = 'black', label = 'Trivially Predicted')

plt.title('Danube Level at Kienstock (km 2015.021)')

plt.xlabel('time in days')

plt.ylabel('max water level in cm')

plt.legend()

plt.show()
# Visualising the results for a couple of months

lo = 0

hi = len(y_test)

plt.figure(num=None, figsize=(10,4))

plt.plot(triv_y_test_pred[lo:hi], color = 'black', label = 'Trivially Predicted')

plt.plot(y_test[lo:hi], color = 'blue', label = 'Real Water Level')

plt.plot(lr_y_test_pred[lo:hi], color = 'green', label = 'LinReg Predicted')

plt.plot(lstm_y_test_pred[lo:hi], color = 'red', label = 'LSTM Predicted')

plt.title('Danube Level at Kienstock (km 2015.021)')

plt.xlabel('time in days')

plt.ylabel('max water level in cm')

plt.legend()

plt.show()
res = pd.DataFrame(triv_residuals[:,0])

res.insert(loc=1,column='LinReg',value=lr_test_residuals[:,0])

res.insert(loc=2,column='LSTM',value=lstm_test_residuals[:,0])

res.rename(columns={0: 'Trivial'})

plotPerColumnDistribution(res,3,3)