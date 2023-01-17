import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline  

# This is important to display plots inline in notebook



import tensorflow as tf
# create x data with one feature

x_data = np.linspace(0.0, 10.0, 1000000)

x_data
# Some noise to be added in the data. Same data points as x data

noise = np.random.randn(len(x_data))

noise
y_true = (0.5 * x_data) + 5 + noise 

y_true
# Merge data to create pd dataframe

x_df = pd.DataFrame(data=x_data, columns=["X_Data"])

y_df = pd.DataFrame(data=y_true, columns=["Y"])



my_data = pd.concat([x_df, y_df], axis = 1)

my_data.head()
my_data.sample(250).plot(x = "X_Data", y = "Y", kind = "scatter")
# We are here taking a batch size of 8.

# There is no good, bad or optimal batch size defined, it depends on the size of data you're dealing with.

batch_size = 8
# Take two random numbers to initiate the slope and intercept variables.

# Numbers really doesn't matter here, as we're going to imrove these using grdient descent method.

np.random.randn(2)
m = tf.Variable(0.63)

b = tf.Variable(0.61)
# We're sure here of the size of the placeholder. 

# It would contain 8 data points at a time for this example, as our batch size is 8.

xph = tf.placeholder(tf.float32, [batch_size])

yph = tf.placeholder(tf.float32, [batch_size])
# Create model/operation graph

y_model = m * xph + b
# Loss function - sum of squared errors

error = tf.reduce_sum(tf.square(yph - y_model))
# Create optimizer

# Creating a Gradient Descent Optimizer to train and minimize the error.

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

train = optimizer.minimize(error)
# initiate the global initializer

init = tf.global_variables_initializer()
#Execute the operation graphs now

with tf.Session() as sess:

    

    sess.run(init)

    

    # No of epocs/no of batches to train upon

    batches = 1000

    

    for i in range(batches):

        

        # create random indices to sample out from training data

        # Creating integers upto number of rows in training data and then sampling out batch of 8 random indices

        rand_index = np.random.randint(len(x_data), size = batch_size)

        

        # Create feed dictionary

        feed = {xph: x_data[rand_index], yph: y_true[rand_index]}

        

        # run train optimizer operation graph

        sess.run(train, feed_dict = feed)

        

    model_m, model_b = sess.run([m,b])

        
model_m
model_b
#Execute the operation graphs now

with tf.Session() as sess:

    

    sess.run(init)

    

    # No of epocs/no of batches to train upon

    batches = 5000

    

    for i in range(batches):

        

        # create random indices to sample out from training data

        # Creating integers upto number of rows in training data and then sampling out batch of 8 random indices

        rand_index = np.random.randint(len(x_data), size = batch_size)

        

        # Create feed dictionary

        feed = {xph: x_data[rand_index], yph: y_true[rand_index]}

        

        # run train optimizer operation graph

        sess.run(train, feed_dict = feed)

        

    model_m, model_b = sess.run([m,b])

        
model_m
model_b
y_hat = x_data * model_m + model_b
my_data.sample(250).plot(x = "X_Data", y = "Y", kind = "scatter")

plt.plot(x_data, y_hat, 'r')
feature_col = [tf.feature_column.numeric_column('x', shape = [1])]
estimator = tf.estimator.LinearRegressor(feature_columns = feature_col)



# Ignore the warning below
# Train-test split the data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_true,

                                                   test_size = 0.3, random_state = 123)
# Check out the shape for train and test datasets

x_train.shape
x_test.shape
# This can take inputs from both numpy arrays and pandas dataframes

# input_fns = tf.estimator.inputs.pandas_input_fn()  # Example



# As we are using numpy arrays, so we'll be using below input fn

input_fns = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,

                                              batch_size = 8, num_epochs = None, shuffle = True)
train_input_fns= tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,

                                              batch_size = 8, num_epochs = 1000, shuffle = False)
test_input_fns = tf.estimator.inputs.numpy_input_fn({'x': x_test}, y_test,

                                              batch_size = 8, num_epochs = 1000, shuffle = False)
estimator.train(input_fn=input_fns, steps=1000)
train_eval_matrix = estimator.evaluate(input_fn=train_input_fns, steps=1000)
test_eval_matrix =  estimator.evaluate(input_fn=test_input_fns, steps=1000)
print("Training data matrix:")

print(train_eval_matrix)
print("Test data matrix:")

print(test_eval_matrix)
new_data = np.linspace(0,10, 10)

new_data
predict_input_fn = tf.estimator.inputs.numpy_input_fn({'x': new_data}, shuffle=False)
estimator.predict(input_fn=predict_input_fn)
list(estimator.predict(input_fn=predict_input_fn))
y_pred = []



for y_hat in estimator.predict(input_fn=predict_input_fn):

    y_pred.append(y_hat['predictions'])

    

y_pred
my_data.sample(n = 250).plot(kind = 'scatter',  x = 'X_Data', y = 'Y')

plt.plot(new_data, y_pred, 'r')