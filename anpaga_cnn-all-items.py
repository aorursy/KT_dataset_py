import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



print("TensorFlow version:", tf.__version__)



# set pandas

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 10)

pd.set_option('display.width', 1000)
# When set to true, the network will use two months of data for validation.

# Set to false to train on the full training set!

validate_while_training = False
number_of_items = 30490

test_days = 28

n_days_train = int(3.6*365) # Memory issues when training on more data



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
# Transpose sales_train_validation

sales_train_validation = sales_train_validation.T
# Split the evaluation set on training, validation and test (test is here the kaggle validation set)

sales_val = sales_train_validation.iloc[6 + last_train_day:6 + last_val_day, :]

sales_test = sales_train_validation.iloc[6 + last_val_day:, :]

sales_train = sales_train_validation.iloc[6 + days_train_ini:6 + last_train_day, :]

del sales_train_validation

print(sales_train.shape)

print(sales_val.shape)

print(sales_test.shape)
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

y_train = []

for i in range(days_back, last_train_day - days_train_ini - days_back):

    X_train.append(sales_train[i-days_back:i])

    isevent1_train.append(calendar_train.iloc[i:i+days_predict, -1].values)

    y_train.append(sales_train[i:i+days_predict][0:number_of_items])
# Validation

X_val = []

isevent1_val = []

y_val = []

for i in range(days_back, len(sales_val)):

    X_val.append(sales_val[i-days_back:i])

    isevent1_val.append(calendar_val.iloc[i:i+days_predict, -1].values)

    y_val.append(sales_val[i:i+days_predict][0:number_of_items])
# Convert to np array

X_train = np.array(X_train)

isevent1_train = np.array(isevent1_train)

y_train = np.array(y_train)

X_val = np.array(X_val)

isevent1_val = np.array(isevent1_val)

y_val = np.array(y_val)



print('Shape of X_train: ', X_train.shape)

print('Shape of isevent1_train: ', isevent1_train.shape)

print('Shape of y_train: ', y_train.shape)

print('Shape of X_val: ', X_val.shape)

print('Shape of isevent1_val: ', isevent1_val.shape)

print('Shape of y_val: ', y_val.shape)
def get_cnn_plus_dense(width=100, deep=False):

    ''' 

    The width parameter sets the number of neurons on the dense layer just before the output

    '''

    # Categorical input

    cat_input = tf.keras.Input(shape=(days_predict, 1))



    # Sales branch

    inputs_sales = tf.keras.Input(shape=(days_back, number_of_items))



    sales = tf.keras.layers.Conv1D(256, 7, strides=1, activation='relu')(inputs_sales)

    sales = tf.keras.layers.BatchNormalization()(sales)

    sales = tf.keras.layers.Dropout(0.2)(sales)

    sales = tf.keras.layers.Conv1D(512, 5, strides=1, activation='relu')(sales)

    sales = tf.keras.layers.BatchNormalization()(sales)

    sales = tf.keras.layers.AveragePooling1D(4)(sales)

    sales = tf.keras.layers.Dropout(0.2)(sales)



    # Concatenate + dense

    concat = tf.keras.layers.Concatenate()([sales, cat_input])

    dense = tf.keras.layers.Dense(width, activation='relu')(concat)

    dense = tf.keras.layers.BatchNormalization()(dense)

    if deep:

        dense = tf.keras.layers.Dropout(0.2)(dense)

        dense = tf.keras.layers.Dense(500, activation='relu')(dense)

        dense = tf.keras.layers.BatchNormalization()(dense)

        dense = tf.keras.layers.Dropout(0.2)(dense)

        dense = tf.keras.layers.Dense(250, activation='relu')(dense)

        dense = tf.keras.layers.BatchNormalization()(dense)

        dense = tf.keras.layers.Dropout(0.2)(dense)

        dense = tf.keras.layers.Dense(100, activation='relu')(dense)

        dense = tf.keras.layers.BatchNormalization()(dense)



    # Output branch

    out = tf.keras.layers.Dense(number_of_items)(dense)



    model = tf.keras.Model(inputs=[inputs_sales, cat_input], outputs=out)



    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="mean_squared_error", optimizer=opt)

    

    return model
# Get the relevant model and print summary and structure

model = get_cnn_plus_dense(width=100, deep=False)

print(model.summary())

tf.keras.utils.plot_model(model, show_shapes=True)
# Fitting the network to the Training set

epoch_no=25

batch_size=15

if validate_while_training:

    model.fit([X_train, isevent1_train], 

              y_train, 

              epochs = epoch_no, 

              batch_size = batch_size,

              validation_data=([X_val, isevent1_val], y_val))

else:

    model.fit([X_train, isevent1_train], 

              y_train, 

              epochs = epoch_no, 

              batch_size = batch_size)
X_test = []

isevent1_test = []



predictions = []



X_test.append(input_sales[0:days_back, :])

X_test = np.array(X_test)



for j in range(days_back,days_back + test_days):

    isevent1_test = np.expand_dims(np.array(calendar_test.iloc[j-days_back, -1]), axis=0)

    test_input = [X_test[0,j - days_back:j].reshape(1, days_back, number_of_items),

                  isevent1_test]

    predicted_sales = model.predict(test_input)

    testInput = np.array(predicted_sales)

    testInput[testInput < 0] = 0

    X_test = np.append(X_test, testInput).reshape(1, j + 1, number_of_items)

    predicted_sales = scaler.inverse_transform(np.squeeze(testInput, axis=0))[:, 0:number_of_items]

    predictions.append(predicted_sales)



predictions = pd.DataFrame(data=np.array(predictions).reshape(28,30490))

print(predictions.shape)
print('Predicted sales per item')

print(predictions.head(10))

print('Actual sales per item')

print(sales_test.head(10))
# Simple mse

error = mean_squared_error(sales_test.values, predictions.values, squared=True)

print(error)
submission = pd.DataFrame(data=np.array(predictions).reshape(28,30490))



submission[submission < 0] = 0 # Should not be any at this point, but just in case...



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



submission.to_csv("/kaggle/working/submission_avg_shallow_100.csv.csv", index=False)