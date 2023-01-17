import numpy as np

import pandas as pd



import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



print("TensorFlow version:", tf.__version__)
# set pandas

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 10)

pd.set_option('display.width', 500)
# When set to true, the network will use two months of data for validation.

# Set to false to train on the full training set!

validate_while_training = False
number_of_items = 30490

test_days = 28

n_days_train = int(2*365) # Memory issues when training on more data



if validate_while_training:

    validation_days = test_days * 2  # (2 months for validation)

else:

    validation_days = 0

    

last_train_day = 1913 - validation_days # Last day used for training

last_val_day = last_train_day + validation_days # Last day used for validation

days_train_ini = last_train_day - n_days_train # First day used for training



days_back = 14 # Number of days to pass to the CNN (history)

days_predict = 1 # The network predicts one day at the time
# Read the sales data

sales_train_validation = pd.read_csv(

    '/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv'

)
# Replace the strings of the categorical variables by an integer

cat_col = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

for col in cat_col:

    sales_train_validation[col] = sales_train_validation[col].astype('category')

    sales_train_validation[col] = sales_train_validation[col].cat.codes
# Transpose sales_train_validation

sales_train_validation = sales_train_validation.T
# Extract categorical data from the sales dataset

items = sales_train_validation.iloc[1:6, :]
# Split the evaluation set on training, validation and test (test is here the kaggle validation set)

sales_val = sales_train_validation.iloc[6 + last_train_day:6 + last_val_day, :]

sales_test = sales_train_validation.iloc[6 + last_val_day:, :]

sales_train = sales_train_validation.iloc[6 + days_train_ini:6 + last_train_day, :]

del sales_train_validation # save memory

print(sales_train.shape)

print(sales_val.shape)

print(sales_test.shape)

print(items.shape)
# Normalize the sales using MinMax

scaler = MinMaxScaler(feature_range= (0,1))

sales_train = scaler.fit_transform(sales_train.values)

if validate_while_training:

    sales_val = scaler.transform(sales_val.values)
# Get the input for the first prediction of the model (i.e. day 1) 

input_sales= sales_train[-days_back:, :]
calendar = pd.read_csv(

    '/kaggle/input/m5-forecasting-accuracy/calendar.csv',

)



# drop the columns that are not used

drop_cols_cal = ['wm_yr_wk', 'date', 'weekday', 'month', 'year', 'wday']

calendar.drop(drop_cols_cal, axis='columns', inplace=True)
# Make event_name_1 a categorical variable

calendar.loc[:, 'event_name_1'] = calendar.event_name_1.astype('category')
# Add boolean for event 1

calendar['isevent1'] = 0

calendar.loc[~calendar.event_name_1.isna(), 'isevent1'] = 1

calendar.loc[calendar.event_name_1.isna(), 'isevent1'] = 0
# Separate training and test calendars

calendar_train = calendar.iloc[days_train_ini:last_train_day, :]

calendar_val = calendar.iloc[last_train_day:last_val_day, :]

calendar_test = calendar.iloc[last_val_day:, :]

del calendar

print(calendar_train.shape)

print(calendar_val.shape)

print(calendar_test.shape)
# Training

X_train = []

isevent1_train = []

items_train = []

y_train = []



for i in range(days_back, last_train_day - days_train_ini - days_back):

    X_train.append(sales_train[i-days_back:i])

    isevent1_train.append(calendar_train.iloc[i:i+days_predict, -1].values)

    items_train.append(items.values)

    y_train.append(sales_train[i:i+days_predict])
# Validation

X_val = []

isevent1_val = []

items_val = []

y_val = []



for i in range(days_back, len(sales_val)):

    X_val.append(sales_val[i-days_back:i])

    isevent1_val.append(calendar_val.iloc[i:i+days_predict, -1].values)

    items_val.append(items.values)

    y_val.append(sales_val[i:i+days_predict])
#Convert to np array

X_train = np.array(X_train)

isevent1_train = np.array(isevent1_train)

y_train = np.array(y_train)

X_val = np.array(X_val)

isevent1_val = np.array(isevent1_val)

y_val = np.array(y_val)



items_train = np.array(items_train)

items_val = np.array(items_val)



print('Shape of X_train: ', X_train.shape)

print('Shape of isevent1_train: ', isevent1_train.shape)

print('Shape of y_train: ', y_train.shape)

print('Shape of X_val: ', X_val.shape)

print('Shape of isevent1_val: ', isevent1_val.shape)

print('Shape of y_val: ', y_val.shape)



print('Shape of items_train: ', items_train.shape)

print('Shape of items_val: ', items_val.shape)
# Reshape the arrays (this could be done with more compact code, but it was quickly

# adapted from the CNN for all items)

X_train = X_train.reshape((X_train.shape[0]*number_of_items, days_back, 1))

X_val = X_val.reshape((X_val.shape[0]*number_of_items, days_back, 1))

y_train = y_train.reshape((y_train.shape[0]*number_of_items, days_predict, 1))

y_val = y_val.reshape((y_val.shape[0]*number_of_items, days_predict, 1))



isevent1_train = np.tile(isevent1_train, (number_of_items, 1))

isevent1_val = np.tile(isevent1_val, (number_of_items, 1))



items_train = items_train.reshape((items_train.shape[0]*number_of_items, items.shape[0], 1))

items_train = np.moveaxis(items_train, 1, -1)

items_val = items_val.reshape((items_val.shape[0]*number_of_items, items.shape[0], 1))

items_val = np.moveaxis(items_val, 1, -1)



print('Shape of X_train: ', X_train.shape)

print('Shape of isevent1_train: ', isevent1_train.shape)

print('Shape of y_train: ', y_train.shape)

print('Shape of X_val: ', X_val.shape)

print('Shape of isevent1_val: ', isevent1_val.shape)

print('Shape of y_val: ', y_val.shape)

print('Shape of items_train: ', items_train.shape)

print('Shape of items_val: ', items_val.shape)
del calendar_train, sales_train
def get_sequential_CNN():

    # Categorical input

    cat_input = tf.keras.Input(shape=(days_predict, items.shape[0]))

    isevent1_input = tf.keras.Input(shape=(days_predict, 1))



    # CNN

    inputs_sales = tf.keras.Input(shape=(days_back, 1))

    sales = tf.keras.layers.Conv1D(32, 7, strides=1, activation='relu')(inputs_sales)

    sales = tf.keras.layers.BatchNormalization()(sales)

    sales = tf.keras.layers.Dropout(0.2)(sales)

    sales = tf.keras.layers.Conv1D(64, 5, strides=1, activation='relu')(sales)

    sales = tf.keras.layers.BatchNormalization()(sales)

    sales = tf.keras.layers.Dropout(0.2)(sales)

    sales = tf.keras.layers.AveragePooling1D(4)(sales)



    # Concatenate CNN and categorical data

    concat = tf.keras.layers.Concatenate()([sales, cat_input, isevent1_input])

    concat = tf.keras.layers.BatchNormalization()(concat)



    # Dense

    dense = tf.keras.layers.Dense(100, activation='relu')(concat)

    dense = tf.keras.layers.BatchNormalization()(dense)

    dense = tf.keras.layers.Dense(100, activation='relu')(dense)

    dense = tf.keras.layers.BatchNormalization()(dense)



    # Output layer

    out = tf.keras.layers.Dense(1)(dense)



    model = tf.keras.Model(inputs=[inputs_sales, cat_input, isevent1_input], outputs=out)



    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="mean_squared_error", optimizer=opt)

    

    return model

model = get_sequential_CNN()

print(model.summary())

tf.keras.utils.plot_model(model, show_shapes=False)
# Fitting the network to the Training set

epoch_no=5

batch_size=30490



if validate_while_training:

    model.fit([X_train, items_train, isevent1_train], 

              y_train, 

              epochs = epoch_no, 

              batch_size = batch_size,

              validation_data=([X_val, items_val, isevent1_val], y_val))

else:

    model.fit([X_train, items_train, isevent1_train], 

              y_train, 

              epochs = epoch_no, 

              batch_size = batch_size)

    

#model.save('CNN_per_item_avg_3ep_2048batch.h5')
# Initialize X_test

X_test = []

predictions = []



X_test.append(input_sales[0:days_back, :])

X_test = np.array(X_test)

X_test = X_test.reshape((X_test.shape[0]*number_of_items, days_back, 1))
# Loop for each of the 28 test days

# Note that negative sales are set to 0 during each iteration

items_test = items.T.values[:, np.newaxis, :]

for j in range(days_back,days_back + test_days):

    isevent1_test = np.tile(calendar_test.iloc[j-days_back, -1], (number_of_items,1))

    test_input = [X_test[:,j - days_back:j, :].reshape(number_of_items, days_back, 1),

                  items_test,

                  isevent1_test]

    predicted_sales = model.predict(test_input)

    testInput = np.array(predicted_sales)

    testInput[testInput < 0] = 0

    X_test = np.append(X_test, testInput).reshape(number_of_items, j + 1, 1)

    predicted_sales = testInput

    predictions.append(predicted_sales)



predictions = scaler.inverse_transform(np.array(predictions).reshape(28,30490))

predictions = pd.DataFrame(data=predictions)

print(predictions.shape)
print(predictions.head(10))

print(predictions.shape)

print(sales_test.head(10))
# Simple mse

error = mean_squared_error(sales_test.values, predictions.values, squared=True)

print(error)
submission = pd.DataFrame(data=np.array(predictions).reshape(28,30490))



submission[submission < 0] = 0



submission = submission.T

    

submission = pd.concat((submission, submission), ignore_index=True)



sample_submission = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

    

idColumn = sample_submission[["id"]]

    

submission[["id"]] = idColumn  



cols = list(submission.columns)

cols = cols[-1:] + cols[:-1]

submission = submission[cols]



colsdeneme = ["id"] + [f"F{i}" for i in range (1,29)]



submission.columns = colsdeneme



submission.to_csv("/kaggle/working/submission_cnn_per_item.csv", index=False)