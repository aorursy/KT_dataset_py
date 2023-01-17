# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



# Scikit-Learn train-test split 

from sklearn.model_selection import train_test_split



# Images, plots, display, and visualization

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



print("Tensorflow Version: ", tf.VERSION)

print("Pandas Version: ", pd.__version__)

print("Seaborn Version: ", sns.__version__)
%matplotlib inline

style.use('ggplot')
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',

               'Acceleration', 'Model Year', 'Origin']

"""

1. As data file does not contains the column names then we have to specify the column names manually. 

2. The Null values in the data file is represented by '?' symbol. So we have to assign it to the 

'_na_values' attributes.

3. 'skipinitialspace' attribute skip spaces after delimiter(alternative argument name for sep).

"""

dataset_path ="../input/auto-mpg.data"

raw_dataset = pd.read_csv(filepath_or_buffer = dataset_path, names = column_names, na_values = "?",

                         comment = '\t', sep = " ", skipinitialspace = True)



"""Creates a copy of dataset so that raw_dataset remain unchanged."""

dataset = raw_dataset.copy()

"""Top five rows of the dataset"""

dataset.head()
"""Last five rows of the dataset"""

dataset.tail()
dataset.shape
"""We can see that the dataset's columns are of following data-types:"""

dataset.dtypes
"""To know more about the dataset use 'info' method"""

dataset.info()
"""Count the null values in the dataset"""

dataset.isna().sum()
"""Check out our 'Horsepower column"""

dataset['Horsepower'].describe()
"""Calculate the median and mode"""



print("Median of horsepower feature is {}".format(dataset['Horsepower'].median()))

print("Mode of horsepower feature is \n{}".format(dataset['Horsepower'].mode()))
"""Check out the percentage of mode value present in our dataset"""



mode_value = len(dataset[dataset['Horsepower'] == 150])

print("Percentage of mode value present is {}".format((mode_value/len(dataset)) * 100))
"""Let's see what are Null values"""



dataset[dataset['Horsepower'].isna() == True]
"""Percentage of missing values"""



print("Percentage of missing values are {}".format((6/len(dataset)) * 100))
"""Replace the NaN values with the mean"""

dataset['Horsepower'].fillna(

                       dataset['Horsepower'].mean(), inplace = True)
"""Check if still null values are present?"""

dataset.isna().sum()
"""Visualize our target feature 'MPG'"""

sns.distplot(dataset['MPG'])
"""Calculate the Skewness and Kurtosis"""



print("Skewness: {}".format(dataset['MPG'].skew()))

print("Kurtosis: {}".format(dataset['MPG'].kurt()))
"""Visualize the Cylinder feature"""

sns.countplot(dataset['Cylinders'])
"""In Percentage, type of cylinders avaliable"""



print("Percentage of type 3 cylinders are {}".format(

    (len(dataset[dataset['Cylinders'] == 3]) / len(dataset)) * 100)

     )

print("Percentagee of type 4 cylinders are {}".format(

    (len(dataset[dataset['Cylinders'] == 4]) / len(dataset)) * 100)

     )

print("Percentage of type 5 cylinders are {}".format(

    (len(dataset[dataset['Cylinders'] == 5]) / len(dataset)) * 100)

     )

print("Percentage of type 6 cylinders are {}".format(

    (len(dataset[dataset['Cylinders'] == 6]) / len(dataset)) * 100)

     )

print("Percentage of type 8 cylinders are {}".format(

     (len(dataset[dataset['Cylinders'] == 8]) / len(dataset)) * 100)

     )
"""We can also count the number of different type of cylinders available."""

dataset['Cylinders'].value_counts()
"""Visualize the Model Year feature"""

sns.countplot(dataset['Model Year'])
"""We can also count the total number of different type of model year categories available"""

dataset['Model Year'].value_counts().sort_index()
"""Visualize the Origin feature"""



sns.countplot(dataset['Origin'])
"""We can also count the total number of diffenent 'Origin' categories avaliable."""

dataset['Origin'].value_counts()
"""Removes the 'Origin' column from the dataset"""

origin = dataset.pop('Origin')

print(dataset.columns)
"""Do the one-hot encoding"""

dataset['USA'] = (origin == 1) * 1.0

dataset['Europe'] = (origin == 2) * 1.0

dataset['Japan'] = (origin == 3) * 1.0

dataset.head()
"""Split the dataset into features and labels"""

labels = dataset.pop('MPG')

features = dataset
"""Split the dataset"""

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = 3)
"""Have a look at the training dataset"""

features_train.head()
features_train_des = features_train.describe().transpose()

features_train_des
"""Define a function to normalize"""

def norm_scale(a):

    b = (a - a.mean())/(a.std())

    return b
"""Normalize the train and test dataset"""

"""Keep in mind that the training and testing dataset normalized by the same method."""

norm_features_train = norm_scale(features_train)

norm_features_test = norm_scale(features_test)
"""Have a look at the training dataset"""

norm_features_train.head()
"""Have a look at the testing dataset"""

"""Look only just to conform that testing dataset is also scaled"""

norm_features_test.head()
# """Create a build model function"""



def build_model():

    

    

#     """Intintiate the Sequential Model"""

    

    model =  keras.Sequential()

    

#     """Let's create two densely/fully-connected hidden layer"""

    

#     """1. First fully connected layer"""

    

#     """Add a fully-connected layer with 64 units to the model

#     Also add non-linearity by using activation function

#     Specifying the input array shape of norm_features_train"""

    

    model.add(layers.Dense(units = 64, activation = tf.nn.relu,

                           input_shape = [norm_features_train.shape[1]]))

#     """

#     Now our model will take an input array of shape (batch_size, input_shape)

#     The batch_size is the size we choose 

#     the output array is of shape (batch_size, units)

#     """

    

#     """

#     The output of dense layer is: output  = activation(dot(input, kernel) + bias)

#     Where <kernel> is the weight matrix created by the layer.

#     And bias is the bias vector also created by the layer.

#     """

    

#     """2. Second fully connected layer"""



#     """Now after the first layer we don't need to specify the size of the input anymore. 

#        Add another layer in the similar way"""



    model.add(layers.Dense(units = 64, activation = tf.nn.relu))

    

    

#     """Output layer"""

    

    model.add(layers.Dense(units = 1))

#     """Single unit represents single node that returns single continuous value"""



    

#     """After successfully adding the layers it's time to compile our model"""

    

#     """There are too many omtimizers out there like:

#        tf.train.MomentumOptimizer, tf.train.AdagradOptimizer, tf.train.AdadeltaOptimizer

#        tf.train.AdamOptimizer, or tf.train.RMSPropOptimizer"""

    

    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

#     """The role of the optimizer is to optimize the learning rate.

#        So that we reaches towards the result much faster.

#        It help in increasing rate of training.

#        Suppose we are having a learning rate which adapt or change itself according to the gradient.

#        This type of learning rate is known as 'Adaptive Learning Rate'."""

    

    loss = tf.keras.losses.mse

#     """Loss function is the one that needed to be minimize during optimization

#        It measures the total loss over an entire dataset.

#        Loss function is also known as cost function or object function or empirical risk.

#        Some of the loss functions are: 

#        'mean square error' --> used with regression problems, 

#        'categorical_crossentropy' --> used with multi-class classification problems, or

#        'binary_crossentropy' --> used with binary-class classification problems

        

#        mean square error(mse) = average(sum(predict - actual)^2)"""

    

    metrics = [tf.keras.metrics.mae, tf.keras.losses.mse]

#     """Metrics used to monitor training.

#        Some of the common metrices are:

#        mean absolute error, mean squared error, mean absolute percentage error, 

#        mean squared logarithmic error"""

              

    

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    

    return model
"""Call our function to look at the weights"""

model = build_model()
"""Look at the assigned weights by the layer"""

print("Weights are: \n{}".format(model.weights))
"""Use 'get_weights' method to see all the weights"""

weights = model.get_weights()
"""Look at the summary of the model we created"""

model.summary()
EPOCHS = 1000



# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs):

    if epoch % 100 == 0: print('')

    print('.', end='')

    

"""

'batch_size': Number of samples per gradient update. If unspecified, batch_size will default to 32.

(which is in our case)

'epochs': An epoch is an iteration over the entire x and y data provided.

'verbose': Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

'validation_split': Fraction of the training data to be used as validation data. 

The model will set apart this fraction of the training data, will not train on it,

and will evaluate the loss and any model metrics on this data at the end of each epoch.

'validation_data': We can also provide our own validation data to test on.

"""

history = model.fit(x = norm_features_train, y = labels_train, epochs=EPOCHS, 

                    validation_split = 0.2, verbose = 0, callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

"""Look values at the start of training"""

hist.head()
"""Look values at the end of training"""

hist.tail()
"""Improve the model"""

model_improved = build_model()



"""

The patience parameter is the amount of epochs to check for improvement.

If a set amount of epochs elapses without showing improvement, then automatically stop the training.

"""

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)



history_improved = model_improved.fit(x = norm_features_train, y = labels_train, epochs=EPOCHS,

                    validation_split = 0.2, batch_size = 32, verbose=0, 

                                      callbacks=[early_stop, PrintDot()])
hist_improved = pd.DataFrame(history_improved.history)

hist_improved['epoch'] = history_improved.epoch

"""Look values at the start of training"""

hist_improved.head()
hist_improved.tail()
"""Check accuracy on the testing dataset"""

loss, mae, mse = model.evaluate(norm_features_test, labels_test, verbose=0)



print("Testing set Mean Sqrd Error: {:5.2f} MPG".format(mse))

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

print("Testing Val Loss Error: {:5.2f} MPG".format(loss))
"""Check accuracy on the testing dataset"""

loss, mae, mse = model_improved.evaluate(norm_features_test, labels_test, verbose=0)



print("Testing set Mean Sqrd Error: {:5.2f} MPG".format(mse))

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

print("Testing Val Loss Error: {:5.2f} MPG".format(loss))
test_predictions = model_improved.predict(norm_features_test).flatten()



plt.scatter(labels_test, test_predictions)

plt.xlabel('True Values [MPG]')

plt.ylabel('Predictions [MPG]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0,plt.xlim()[1]])

plt.ylim([0,plt.ylim()[1]])

_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - labels_test

plt.hist(error, bins = 25)

plt.xlabel("Prediction Error [MPG]")

_ = plt.ylabel("Count")