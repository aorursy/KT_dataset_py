import pandas as pd

import numpy as np

import keras

from matplotlib import pyplot

from keras.layers import Dense

from keras.models import Sequential



 

# Import training data to Pandas DataFrame, get dummies, and replace NaN with 0

dummies_df = pd.get_dummies(pd.read_csv('../input/housepricesac/train.csv')).fillna(0)



# Put SalePrice in the 1 position, the 0 position has ID which is good

cols = dummies_df.columns.tolist()

cols.insert(1, cols.pop(cols.index('SalePrice')))

dummies_df = dummies_df.reindex(columns= cols)



# Get feature and target variables

X = dummies_df.iloc[:,2:]

y = dummies_df['SalePrice']



# Save the number of columns in predictors: n_cols

n_cols = X.shape[1]



# Set up the model: model

model = Sequential()



# Add the layers

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(500, activation='relu', input_shape=(n_cols,))) 

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(500, activation='relu', input_shape=(n_cols,)))

model.add(Dense(1))



# Define early_stopping_monitor

early_stopping_monitor = keras.callbacks.EarlyStopping(patience=3)



# Compile the model

model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_logarithmic_error'])



# Fit the model

history = model.fit(X,y,epochs=30,validation_split=0.3,callbacks=[early_stopping_monitor])



# Import the test data to a Pandas DataFrame, get dummies, replace NaN with 0, and exclude the 0 column which is ID

test_df = pd.get_dummies(pd.read_csv('../input/housepricesac/test.csv')).fillna(0)

test_final_df = test_df.iloc[:,1:]



# Find out if training data and test data have the same number of columns, may not due to get_dummies

X_list = X.columns.values

test_final_df_list = test_final_df.columns.values 



# Give the test data the columns it's missing from the training data

new_cols = (list(set(X_list) - set(test_final_df_list)))

test_ready = pd.concat([test_final_df, pd.DataFrame(columns=new_cols)], axis=1)



# Our new columns need to have NaN replaced with 0

test_ready = test_ready.fillna(0)



# Make sure test data has columns in same order as training data

test_ready = test_ready[X.columns]



# Now we can predict using the test data

y_pred = model.predict(test_ready)



# Create our submission DataFrame with the ID column

submit_df = pd.DataFrame(test_df['Id'])



# Create the SalePrice column and populate with our predictions

submit_df['SalePrice'] = y_pred



# Output our submission dataframe to CSV

submit_df.to_csv('../working/submit_final.csv')



# plot metrics showing how the model improves over epochs

pyplot.plot(history.history['mean_squared_logarithmic_error'])

pyplot.xlabel('Epochs')

pyplot.ylabel('Error')

pyplot.show()