#

# Import packages

#

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler, MinMaxScaler



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation

from keras import metrics

from keras.optimizers import Adam, RMSprop
#

# Load data from CSV file

#

kc_data_org = pd.read_csv("../input/kc_house_data.csv")

kc_data_org.head(5)
#

# Transform dates into year, month and day and select columns.

#

kc_data_org['sale_yr'] = pd.to_numeric(kc_data_org.date.str.slice(0, 4))

kc_data_org['sale_month'] = pd.to_numeric(kc_data_org.date.str.slice(4, 6))

kc_data_org['sale_day'] = pd.to_numeric(kc_data_org.date.str.slice(6, 8))



kc_data = pd.DataFrame(kc_data_org, columns=[

        'sale_yr','sale_month','sale_day',

        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',

        'condition','grade','sqft_above','sqft_basement','yr_built',

        'zipcode','lat','long','sqft_living15','sqft_lot15','price'])

label_col = 'price'



print(kc_data.describe(include='all'))

kc_data.head()
#

# Split data to train and test and normalize data

#

def train_test_split(df, test_percent=.2, seed=None):

    train_percent = 1.0 - test_percent

    np.random.seed(seed)

    perm = np.random.permutation(df.index)

    m = len(df)

    train_end = int(train_percent * m)

    train = perm[:train_end]

    test = perm[train_end:]

    return train, test



train_data, test_data = train_test_split (kc_data)



x_train = np.array(kc_data.loc[train_data, :].drop(label_col, axis=1).iloc[:])

y_train = np.array(kc_data.loc[train_data, [label_col]].iloc[:])



x_test = np.array(kc_data.loc[test_data, :].drop(label_col, axis=1).iloc[:])

y_test = np.array(kc_data.loc[test_data, [label_col]].iloc[:])



# Normalize Input Data

x_scaler = MinMaxScaler().fit(x_train)

y_scaler = MinMaxScaler().fit(y_train)

    

X_train = x_scaler.transform(x_train)

X_test = x_scaler.transform(x_test) 

Y_train = y_scaler.fit_transform(y_train)

Y_test = y_scaler.fit_transform(y_test) 



input_size = X_train.shape[1]

output_size = Y_train.shape[1]



print('X_train.shape', X_train.shape)

print('Y_train.shape', Y_train.shape)



print('X_test.shape', X_test.shape)

print('Y_test.shape', Y_test.shape)



print('Input Size', input_size)

print('Output Size', output_size)

#

# Build mutli layer neural network model

#

network = Sequential()

network.add(Dense(18, activation="relu", input_shape=(input_size,)))

network.add(Dropout(0.3))

network.add(Dense(9, activation="relu"))

network.add(Dropout(0.1))

network.add(Dense(3, activation="relu"))

network.add(Dense(output_size))



network.compile(loss='mean_squared_error', optimizer=Adam(), metrics=[metrics.mae])

print(network.summary())
#

# Train the model

#

network.fit(X_train, Y_train, epochs=30, batch_size = 32, shuffle=True)



train_score = network.evaluate(X_train, Y_train, verbose=0)

valid_score = network.evaluate(X_test, Y_test, verbose=0)
# 

# Evaluate the result

#

print("----------------------------")

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4)) 

print('Test MAE: ', round(valid_score[1], 4), ', Test Loss: ', round(valid_score[0], 4))



print()



inv_y_train = y_scaler.inverse_transform(Y_train)

inv_y_test = y_scaler.inverse_transform(Y_test)



print("Some random price preditions from training set")

print("----------------------------")

y_pred = network.predict(X_train)

inv_y_pred =  y_scaler.inverse_transform(y_pred)

for i in range(0, 10):

    r = np.random.randint(0, len(X_train))

    pred = inv_y_pred[r][0]

    print(f'Train_Price[{r}] = {inv_y_train[r][0]}, Prediction = {pred}')

    

print()



print("Some random price preditions from test set")

print("----------------------------")

y_pred = network.predict(X_test)

inv_y_pred =  y_scaler.inverse_transform(y_pred)

for i in range(0, 10):

    r = np.random.randint(0, len(X_test))

    pred = inv_y_pred[r][0]

    print(f'Test_Price[{r}] = {inv_y_test[r][0]}, Prediction = {pred}')
plt.figure(figsize=(20, 10))

plt.title('Test Data Prediction')

plt.xlabel('Test data vectors')

plt.ylabel('House price')





#sub-sample each 50nth entry

y_plot = inv_y_test[0::50]

pred_plot = inv_y_pred[0::50]



plt.plot(y_plot, 'b', label="Actual Price")

plt.plot(pred_plot, 'g', label="Predicted Price")

plt.legend(loc='upper left')
