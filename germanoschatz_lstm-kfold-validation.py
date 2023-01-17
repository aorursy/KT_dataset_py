# Importing the libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

#os.chdir("../input/usd-jpy/")
print("Current Working Directory " , os.getcwd())
#random.seed( 30 )
path = "/kaggle/input/usd-jpy/"
# Importing the training set
dataset_train = pd.read_csv(path+'USD_JPY.csv')
dataset_train['sma2'] =dataset_train["Close"].rolling(window=2).mean()
dataset_train= dataset_train.dropna()
dataset_train= dataset_train.drop("averages",axis=1)

training_set = dataset_train.iloc[:, [1,2,3,4,5]].values #6
print(dataset_train.head())
print(training_set)
print("Dimensions:",training_set.shape)
# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
lag=60
X_train = []
y_train = []
for i in range(lag, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-lag:i, : ])
    y_train.append(training_set_scaled[i, 4])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

#X_train, X_Validation, y_train, y_Validation = train_test_split(X_train, y_train,random_state=1,test_size=0.10) #stratify=Y shuffle = True
print("Dimensions:",X_train.shape)
#print("Dimensions:",X_Validation.shape)
print("Dimensions:",y_train.shape)
#print("Dimensions:",y_Validation.shape)
#Concatenate test with train dataset
dataset_test = pd.read_csv(path+'USD_JPY_TEST.csv')
real_stock_price = dataset_test.iloc[:, [1,2,3,4,5,6]].values

# Getting the predicted stock price 
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Price']), axis = 0)
dataset_total2 = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
dataset_total3 = pd.concat((dataset_train['High'], dataset_test['High']), axis = 0)
dataset_total4 = pd.concat((dataset_train['Low'], dataset_test['Low']), axis = 0)
#dataset_total5 = pd.concat((dataset_train['averages'], dataset_test['averages']), axis = 0)
dataset_total5 = pd.concat((dataset_train['sma2'], dataset_test['sma2']), axis = 0)

dataset_total = pd.concat((dataset_total, dataset_total2,dataset_total3,dataset_total4,dataset_total5), axis=1) #,dataset_total6
print(dataset_total.head(10))
print(dataset_total.shape)
#creating the test set with shape rows x lag x features dimension
inputs = dataset_total[len(dataset_total) - len(dataset_test) - lag :].values
#inputs = inputs.reshape(-1,1)
print(inputs.shape)

inputs = sc.transform(inputs)
X_test = []
y_test = []
for i in range(lag, 313):   ##313
    X_test.append(inputs[i-lag:i, :])
    y_test.append(inputs[i, 4]) #5
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_test.shape)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5)) #6
print(X_test.shape)
print(y_test.shape)
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras import regularizers
save_path= "/kaggle/output"
loss_per_fold = []
loss_per_fold.append(0)
min_loss=1
fold_no = 1
kfold = KFold(n_splits=10, shuffle=True)
for train, test in kfold.split(X_train, y_train):

#optimizer = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999)
    regressor = Sequential()
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train[train].shape[1], 5))) #, return_sequences = True
    #regressor.add(Dropout(0.10))
    regressor.add(LSTM(units = 100))
    regressor.add(Dropout(0.20))
    regressor.add(Dense(units = 1))

# Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.summary()

# Fitting the RNN to the Training set
#x=regressor.fit(X_train, y_train, epochs = 15, batch_size = 32, validation_split=0.25,shuffle= True)
    x=regressor.fit(X_train[train], y_train[train], epochs = 10, batch_size = 64, validation_data=(X_train[test],y_train[test]), verbose=2)
    test_acc = regressor.evaluate(X_train[test], y_train[test], verbose=2) #X_test,y_tesy
    loss_per_fold.append(test_acc)
    print(test_acc)
    
    if min_loss > loss_per_fold[fold_no]:
        min_loss = loss_per_fold[fold_no]
        regressor.save(save_path+'stock_prediction.h5')
             # Visualize the training data
        plt.figure(figsize=(10,5))
        plt.plot(x.history['loss']) #x
        plt.plot(x.history['val_loss']) #x
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
  # Increase fold number
    fold_no = fold_no + 1
#make prediction according to the best model
regressor = load_model(save_path+'stock_prediction.h5')
predicted_stock_price = regressor.predict(X_test)
print(predicted_stock_price.shape)

# Get something which has as many features as dataset
trainPredict_extended = np.zeros((len(predicted_stock_price),5)) #6
# Put the predictions there
trainPredict_extended[:,0] = predicted_stock_price[:,0]
# Inverse transform it and select the 3rd column.
trainPredict = sc.inverse_transform(trainPredict_extended)[:,0]

# evaluate the model
test_acc = regressor.evaluate(X_test, y_test, verbose=1)
print(regressor.metrics_names)
# Visualising the results
plt.figure(figsize=(10,5))
plt.plot(real_stock_price[150:180,5], color = 'red', label = 'Real Price')
plt.plot(trainPredict[150:180], color = 'blue', label = 'Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.figure(figsize=(15,10))
plt.show()
#plt.savefig('image.png')
# Visualising the results
plt.figure(figsize=(10,5))
plt.plot(real_stock_price[1:50,5], color = 'red', label = 'Real Price')
plt.plot(trainPredict[1:50], color = 'blue', label = 'Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
# Visualising the results
plt.figure(figsize=(10,5))
plt.plot(real_stock_price[:,5], color = 'red', label = 'Real Price')
plt.plot(trainPredict, color = 'blue', label = 'Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


#trainPredict = pd.DataFrame(trainPredict)
#trainPredict.to_csv("usd_predicted2.csv")