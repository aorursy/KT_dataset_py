import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

import gc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  MinMaxScaler, StandardScaler

from xgboost import XGBRegressor

from keras import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import ModelCheckpoint

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.ensemble import RandomForestRegressor

from keras import regularizers

import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
train = pd.read_excel("../input/Data_Train.xlsx")

test = pd.read_excel("../input/Data_Test.xlsx")

y = train.Price
# Calculate car age

def car_age(df):

    df['Age'] = [2019 - year for year in df.Year]

car_age(train)

car_age(test)
print("Null values in train data\n", train.isnull().sum(), end = "\n\n")

print("Null values in test data\n", test.isnull().sum(), end = "\n\n")
# Replace Null values with 0 for further operations

train.Mileage.fillna('0', inplace=True)

train.Engine.fillna('0 cc', inplace=True)

train.Power.fillna('0 bhp', inplace=True)

train.Seats.fillna(0, inplace=True)

train.New_Price.fillna('0 Lakh', inplace = True)



test.Mileage.fillna('0', inplace=True)

test.Engine.fillna('0 cc', inplace=True)

test.Power.fillna('0 bhp', inplace=True)

test.Seats.fillna(0, inplace=True)

test.New_Price.fillna('0 Lakh', inplace = True)
train.sort_values(by = 'Kilometers_Driven', ascending=False).head(5)
train.Kilometers_Driven[4387] = train.Kilometers_Driven[train.Year == 2017].mean()
def replace_vales(df):

    df['new_mileage'] = [float(i[:5]) if len(i) == 10 else float(i[:4]) for i in df['Mileage']]

    df['new_engine'] = [int(i[:-2].strip()) for i in df['Engine']]

    df['new_power'] = [float(0) if i == 'null bhp' else float(i[:-3].strip()) for i in df['Power']]

    #df['New_Price'] = [float(i[:-4].strip()) for i in df['New_Price']]

    df['New_Price'] = [float(i.split(' ')[0]) if 'Lakh' in i else float(i.split(' ')[0]) * 100 for i in df['New_Price']]

    

replace_vales(train)

replace_vales(test)
train.drop(columns=['Mileage','Engine','Power'], inplace=True)

test.drop(columns=['Mileage','Engine','Power'], inplace=True)
train.new_mileage.replace(0, train.new_mileage.mean(), inplace=True)

train.new_engine.replace(0, train.new_engine.mean(), inplace=True)

train.new_power.replace(0, train.new_power.mean(), inplace=True)

train.Seats.replace(0, 5, inplace=True)



test.new_mileage.replace(0, test.new_mileage.mean(), inplace=True)

test.new_engine.replace(0, test.new_engine.mean(), inplace=True)

test.new_power.replace(0, test.new_power.mean(), inplace=True)

test.Seats.replace(0, 5, inplace=True)
train['Company'] = [name.split(' ', 2)[0] for name in train['Name']]

test['Company'] = [name.split(' ', 1)[0] for name in test['Name']]

train.Company.replace('ISUZU','Isuzu', inplace=True)

test.Company.replace('ISUZU','Isuzu',  inplace=True)
train['Model'] = [' '.join(name.split(" ")[1:]) for name in train['Name']]

test['Model'] = [' '.join(name.split(" ")[1:]) for name in test['Name']]
train.drop(columns=['Name','Year','Price'], inplace=True)

test.drop(columns=['Name','Year'], inplace=True)
train_objs_num = len(train)

dataset = pd.concat(objs=[train, test], axis=0)

dataset_preprocessed = pd.get_dummies(data=dataset, 

                                      columns=['Company','Model','Location', 'Fuel_Type', 'Transmission', 'Owner_Type'], drop_first = True)



train = dataset_preprocessed[:train_objs_num]

test = dataset_preprocessed[train_objs_num:]
col_names = ['Age','Kilometers_Driven', 'Seats', 'new_mileage', 'new_engine', 'new_power','New_Price']

minmax_scaler= MinMaxScaler()

std_scaler = StandardScaler()

train[col_names] = std_scaler.fit_transform(train[col_names])

test[col_names] = std_scaler.fit_transform(test[col_names])
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state = 42)
# Build the model

XGB_model = XGBRegressor(n_estimators=95, learning_rate=0.01, max_depth = 10, random_state= 42, subsample = 0.9,

colsample_bytree = 1 ,gamma = 0.3)

XGB_model.fit(X_train, y_train, early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)], verbose=False)





# make predictions

xgb_predict = XGB_model.predict(X_test)



# Mean Absolute Error

from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(xgb_predict, y_test)))





#Actual and predection plot

fig, ax = plt.subplots()

ax.scatter(y_test, xgb_predict)

ax.plot([y_test.min(), y_test.max()], [xgb_predict.min(), xgb_predict.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()





#Submission

sub = XGB_model.predict(test)

submission_file = pd.DataFrame(sub.tolist(),columns=['Price'])

submission_file.to_excel('sub_xgb.xlsx', index = False)
NN_model = Sequential()



# The Input Layer :

NN_model.add(Dense(24, kernel_initializer='normal',input_dim = X_train.shape[1], kernel_regularizer= regularizers.l2(0.001), activation='relu'))

NN_model.add(Dropout(0.05))

# The Hidden Layers :

NN_model.add(Dense(24, kernel_initializer='normal', kernel_regularizer= regularizers.l2(0.001), activation='relu'))

NN_model.add(Dropout(0.05))





# The Output Layer :

NN_model.add(Dense(1, kernel_initializer='normal'))



# Compile the network :

NN_model.compile(loss='mse', optimizer='adam', metrics=['mse'])

NN_model.summary()
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [checkpoint]





NN_model.fit(X_train,y_train, epochs=100, batch_size=4, validation_split = 0.25, callbacks=callbacks_list)
NN_predictions = NN_model.predict(X_test)

print("Mean Absolute Error : " + str(mean_absolute_error(y_test , NN_predictions)))
fig, ax = plt.subplots()

ax.scatter(y_test, NN_predictions)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
print(NN_model.history.history.keys())

#Loss

plt.plot(NN_model.history.history['loss'])

plt.plot(NN_model.history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
y_pred = pd.DataFrame(NN_predictions.tolist(),columns=['Price'])

pred_error = y_pred['Price'] - y_test

plt.hist(pred_error, bins = 25)

plt.xlabel("Prediction Error [Price]")

plt.ylabel("Count")
#Submission

sub = NN_model.predict(test)

submission_file = pd.DataFrame(sub.tolist(),columns=['Price'])

submission_file.to_excel('sub_nn.xlsx', index = False)