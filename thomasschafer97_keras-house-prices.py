import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing the data in to a dataframe

train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train_data.isnull().any())
#Exploring the data for missing values



null_data = train_data.isnull().any(axis=0)

partial_null_data_list = []

full_null_data_list = []

counter_dict = {}

threshold_for_removal = 85



# The following will display the number of NaN values in each column of the training data

for i in range(0,81):

    counter_i = 0

    if null_data[i] == True:

        for j in range(len(train_data)):

            if train_data[null_data.index[i]][j] != train_data[null_data.index[i]][j]:

                counter_i += 1

        full_null_data_list.append(null_data.index[i])

        if counter_i >= threshold_for_removal:

            partial_null_data_list.append(null_data.index[i])

        print(null_data.index[i], 'column has', counter_i, 'null values')

        counter_dict[null_data.index[i]] = counter_i
# Choosing the data to keep

other_data_to_remove = ['SalePrice', 'Electrical']

features = []

additional_features = []



for column in train_data.columns:

    if not ((column in full_null_data_list) or (column in other_data_to_remove)):

        features.append(column)

    elif not column == 'SalePrice':

        #Exploring the data that we have decided to remove:

        try:

            plt.plot(train_data[column], train_data['SalePrice'], 'o')

            plt.title(str(str(column) + '   NaN: ' + str(100*counter_dict[column]/len(train_data)) + '%'))

            plt.xlabel(column)

            plt.ylabel('Sale price')

            plt.show()

            additional_features.append(column)

        except ValueError:

            pass

        except TypeError:

            pass
# Defining functions to check for and replace any NaN values in the data



def remove_nan(input_list):

    output = []

    for i in input_list:

        if i == i:

            output.append(i)

    return output



def replace_nan(input_list):

    list_avg =  np.mean(remove_nan(input_list))

    for i in range(len(input_list)):

        if input_list[i] != input_list[i]:

            input_list[i] = list_avg

    return input_list



def replace_nan_array(array):

    new_array = []

    assert len(array.shape) == 2

    for i in range(array.shape[0]):

        new_array.append(replace_nan(array[i]))

    return np.array(new_array)
#Formatting the data in to a numpy array



train_test_data = pd.merge(train_data, test_data, how='outer')



def toarray(data, features):

    data = pd.get_dummies(data[features])

    data = data.values

    data = np.array(data, dtype=np.float64)

    return data



train_test_data = toarray(train_test_data, features)

num_features = train_test_data.shape[1]



train_X = train_test_data[:1460, :num_features]

train_y = toarray(train_data, ['SalePrice']).reshape(1460)

test_X = train_test_data[1460:, :num_features]



train_X = replace_nan_array(train_X)

test_X = replace_nan_array(test_X)



#Normalising the data

mean = train_X.mean(axis=0)

std = train_X.std(axis=0)

train_X -= mean

train_X /= std

test_X -= mean

test_X /= std



print(train_X.shape)

print(test_X.shape)
#Setting up a function to build the model



from keras import models, layers, regularizers                          



def build_model():

    model = models.Sequential()

    model.add(layers.Dense(128, activation='relu', input_shape=(num_features,), kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model
# 4-fold validation of the data to help fine-tune the model, particularly deciding how many epochs to use in order

# to not overfit or underfit



k=4

num_val_samples = len(train_X) // k

num_epochs = 300

batch_size = 32

all_mae_histories = []



for i in range(k):

    print('Processing fold ', i)

    val_X = train_X[i*num_val_samples: (i+1)*num_val_samples]

    val_y = train_y[i*num_val_samples: (i+1)*num_val_samples]

    

    partial_train_X = np.concatenate([train_X[:i*num_val_samples], train_X[(i+1)*num_val_samples:]], axis=0)

    partial_train_y = np.concatenate([train_y[:i*num_val_samples], train_y[(i+1)*num_val_samples:]], axis=0)

    

    model = build_model()

    history = model.fit(partial_train_X, partial_train_y, validation_data=(val_X, val_y),

                        epochs=num_epochs, batch_size=batch_size, verbose=0)

    mae_history = history.history['val_mae']

    all_mae_histories.append(mae_history)



average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# # Plotting the graph of mae to check for overfitting



plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
# Deciding exactly how many epochs to train the model with

output = 'Epochs:  '

min_positions = []

for i in range(len(average_mae_history)):

    if average_mae_history[i] <= min(average_mae_history)*1.03:

        output += str(i) + ': ' + str(average_mae_history[i]) + '   '

        min_positions.append(i)

print(output)
# Building a fresh model, using the 4-fold validation results and the above to decide how many epochs to use



model = build_model()

model.fit(train_X, train_y, epochs=min_positions[len(min_positions)//2], batch_size=batch_size, verbose=0)
# Making predictions for the test data

predictions = model.predict(test_X)

predictions = predictions.reshape((len(test_X)))

predictions = replace_nan(predictions)



# Outputting the data

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})

output.to_csv('my_submission.csv', index=False)

print('Complete')