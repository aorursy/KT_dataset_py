import pandas as pd

import seaborn as sb

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing 

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import metrics

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers import SimpleRNN

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.models import load_model





data=pd.read_csv("../input/energydata_complete.csv",index_col='date', parse_dates=True)# data in index



#Copy of the data

data_copy=data.copy()



#plt.plot(data_copy.index,data_copy['Appliances'])



#Supression useless data

data_copy=data_copy.drop(columns=['rv1','rv2'])#feature



#Cleaning data -> count the missing values

    #missing_data = data_copy.isnull()

#for column in missing_data.columns.values.tolist():

#    print(column)

#    print (missing_data[column].value_counts())

#    print("") 



## Correlation

print('Correlation with T9'.center(50))

corr=data_copy.corr()

print(corr.Appliances)

f, ax = plt.subplots(figsize=(7, 7))

sb.heatmap(corr, square=False)

plt.show()



#Scalling data

scaler=preprocessing.StandardScaler()

X=data_copy.drop(columns=['Appliances'])#features

Y=data_copy[['Appliances']]#target

X=X.astype(float)#transform int in float

colnames=list(X)

idxnames=X.index

X=scaler.fit_transform(X) # apply the standardization

X=pd.DataFrame(X, columns=colnames, index=idxnames)



#Creation TestSet and TrainSet

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print("Multilayer Perceptron".center(50))

#Train 

model=Sequential()

#Inputs layer

model.add(Dense(800, activation="elu", input_dim=X_train.shape[1]))



#Hidden layer

model.add(Dense(800, activation="elu"))

model.add(Dense(800, activation="elu"))





#Output layer

model.add(Dense(1))



model.compile(loss='mse',optimizer='adam',metrics=['mae','mse'])

monitor=EarlyStopping(monitor='val_loss', mode='min', patience=30)

earlyStop= ModelCheckpoint(filepath='best_model0.h5')

model.fit(X_train,Y_train.values, validation_split= 0.1,callbacks=[earlyStop,modCheck], epochs=10)# epoch-->back-propagation

model.load_weights('best_model0.h5')

model.summary()



prediction_train=model.predict(X_train)

print('Rsquared= ',metrics.r2_score(Y_train,prediction_train))

# Test Set

prediction_test=model.predict(X_test)

print('Rsquared= ',metrics.r2_score(Y_test,prediction_test))
X_trainS = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

X_testS = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

print("Simple RNN".center(50))



model =Sequential()



model.add(SimpleRNN(200,input_shape=(X_trainS.shape[1],X_trainS.shape[2]),return_sequences=True))

model.add(SimpleRNN(100,return_sequences=True))

model.add(SimpleRNN(20))

model.add(Dense(1))



earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,  patience=50)



modCheck = ModelCheckpoint('best_model1.h5')



model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse','mae'])



model.fit(X_trainS, Y_train, validation_split=0.1,callbacks=[earlyStop,modCheck], epochs = 500)



save = load_model('best_model1.h5')

Y_train_pred = model.predict(X_trainS)

r2=metrics.r2_score( Y_train,Y_train_pred)

print('Rsquared: ', r2)
Y_test_pred = model.predict(X_testS)

r2=metrics.r2_score( Y_test,Y_test_pred)

print('Rsquared: ', r2)
print("LSTM".center(50))

model = Sequential()



model.add(LSTM(units = 150, return_sequences = True, input_shape = (X_trainS.shape[1],X_trainS.shape[2])))

model.add(LSTM(units = 400, return_sequences = True))

model.add(LSTM(units = 20))



model.add(Dense(units = 1))

earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,  patience=50)



modCheck = ModelCheckpoint('best_model2.h5')



model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse','mae'])



model.fit(X_trainS, Y_train, validation_split=0.1,callbacks=[earlyStop,modCheck], epochs = 1)



save = load_model('best_model2.h5')

Y_train_pred = model.predict(X_trainS)

r2=metrics.r2_score( Y_train,Y_train_pred)

print('Rsquared: ', r2)
Y_test_pred = model.predict(X_testS)

r2=metrics.r2_score( Y_test,Y_test_pred)

print('Rsquared: ', r2)