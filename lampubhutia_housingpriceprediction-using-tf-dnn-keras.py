import tensorflow as tf

import numpy as np

import pandas as pd

tf.reset_default_graph()
# importing dataset from tensorflow 



#from tensorflow.contrib.learn import datasets

#boston = datasets.load_dataset('boston')



# importing data from scikit-learn



from sklearn.datasets import load_boston

boston = load_boston()







# Print out the Dataset

print(boston)
# Seperate Data into Features and Labels and load them as a Pandas Dataframe

# Features

features_df = pd.DataFrame(np.array(boston.data), columns=[boston.feature_names])



features_df.head()
# Get the shape of the features

features_df.shape
# Describe the Dataset

features_df.describe()
# Target

prices_df = pd.DataFrame(np.array(boston.target), columns=['prices'])

prices_df.head()
print(prices_df.shape)
prices_df.describe()
# Train Test Split

from sklearn.model_selection import train_test_split
# Train Test Split

# Training Data = 80% of Dataset

# Test Data = 20% of Dataset

X_train, X_test, y_train, y_test = train_test_split(features_df, prices_df, test_size=0.2, random_state=102)
print(X_train.shape, y_train.shape)
print(type(X_train))
print(type(y_train))
print(X_test.shape, y_test.shape)
print(type(X_test))
print(type(y_test)) 
# Normalizing Data

from sklearn.preprocessing import StandardScaler
# Defining the Preprocessing Method and Fiting Training Data into it

scaler = StandardScaler()

scaler.fit(X_train)
# Convert X_train to the Scaled data

# This process scales all the values in all columns and replaces them with the new values

X_train = pd.DataFrame(data=scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_train
# Since we want  train features and prices to be a numpy array.

# so converting them from Pandas Dataframe to Numpy Arrays

X_train = np.array(X_train)

y_train = np.array(y_train)
# Getting the 'Type' of Training Data

type(X_train), type(y_train)
# Apply Normalization on Test Features

scal = StandardScaler()

scal.fit(X_test)
# Convert X_test to the Scaled data

# This process scales all the values in all columns and replaces them with the new values

X_test = pd.DataFrame(data=scal.transform(X_test), columns=X_test.columns, index=X_test.index)
X_test
# Convert test features and prices to Numpy Arrays

X_test = np.array(X_test)

y_test = np.array(y_test)
# Getting the 'Type' e of Test Data

type(X_test), type(y_test)
# Define Training Parameters



# Learning Rate

lr = 0.1



# Number of epochs for which the model will run

epochs = 2500
# Define Features and prices Placeholders



# Features

X = tf.placeholder(tf.float32,[None,X_train.shape[1]])



# Prices  

y = tf.placeholder(tf.float32,[None,1])
# Weights

W = tf.Variable(tf.ones([13,1]))



b = tf.Variable(tf.ones(X_train.shape[1]))
# Initiaize all Variables

init = tf.global_variables_initializer()
# Define Cost Function, Optimizer and the Output Predicitons Function



# Predictions

# y_hat = (W*X + b)

y_hat = tf.add(tf.matmul(X, W), b)



# Cost Function

# MSE

cost = tf.reduce_mean(tf.square(y - y_hat))



# Gradient Descent Optimizer to Minimize the Cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
# Tensor to store the cost after every Epoch

# It becomes handy while plotting the cost vs epochs

cost_history = np.empty(shape=[1],dtype=float)
with tf.Session() as sess:

    # Initialize all Variables

    sess.run(init)

    

    for epoch in range(0,epochs):

        # Run the optimizer and the cost functions

        result, err = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train})

        

        # Add the calculated cost to the array

        cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: X_train, y: y_train}))

        

      # Print the Loss/Error after every 100 epochs  

        if epoch%100 == 0:

            print('Epoch: {0}, Error: {1}'.format(epoch, err))

    

    print('Epoch: {0}, Error: {1}'.format(epoch+1, err))

    

    

    # Values of Weight & Bias after Training

    new_W = sess.run(W)

    new_b = sess.run(b)

    

    # Predicted Labels

    y_pred = sess.run(y_hat, feed_dict={X: X_test})

    

    # Mean Squared Error

    mse = sess.run(tf.reduce_mean(tf.square(y_pred - y_test)))
# New Value of Weights 

print('Trained Weights: \n', new_W)
# New Value of Biases

print('Trained Bias: \n', new_b)
# Predicted Values on TEST Data

print('Predicted Values: \n',y_pred)
# Mean Squared Error

print('Mean Squared Error [TF Session]: ',mse)
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(range(len(cost_history)),cost_history)

plt.axis([0,epochs,0,np.max(cost_history)])

plt.xlabel('Epochs')

plt.ylabel('Cost')

plt.title('Cost vs Epochs', fontsize=25)

plt.show()
features_df.columns
# Make Feature Columns

feat_cols = [tf.feature_column.numeric_column('x', shape=np.array(X_train).shape[1:])]
feat_cols
# Input Function

input_func = tf.estimator.inputs.numpy_input_fn({'x':X_train}, y_train, batch_size=20, num_epochs=2000, shuffle=True)


# Define Linear Regressor Model

linear_model = tf.estimator.LinearRegressor(feature_columns=feat_cols, optimizer='Adam')
# Set up Estimator Training Inputs

train_input_func = tf.estimator.inputs.numpy_input_fn(X_train, y_train, batch_size=1, num_epochs=1000, shuffle=False)
# Set up Estimator Test Inputs

eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': X_test}, y_test, batch_size=1, num_epochs=1, shuffle=False)
# Train the Linear Regressor Estimator

linear_model.train(input_fn=input_func, steps=5000)
# Test the Model

test_metrics = linear_model.evaluate(input_fn=eval_input_func, steps=1000)
test_metrics
# Predicted Values

list(linear_model.predict(input_fn=eval_input_func))
# Predictions

predictions = linear_model.predict(input_fn=eval_input_func)
pred = list(predictions)
# Get Predicted Values as an Array

predicted_vals = []



for pred in linear_model.predict(input_fn=eval_input_func):

    predicted_vals.append(pred['predictions'])
print(predicted_vals)
# Import Mean Squared Error from Scikit Learn

from sklearn.metrics import mean_squared_error
# Calculate the Mean Squared Error

mse = mean_squared_error(predicted_vals, y_test)
print('Mean Squared Error [LinearRegrssor]: ',mse)
# Define DNN Regressor Model

dnn_model = tf.estimator.DNNRegressor(hidden_units=[10,10,10],feature_columns=feat_cols, optimizer='Adam')
# Train the DNN Regressor Estimator

dnn_model.train(input_fn=input_func, steps=2000)
# Evaluate the Model

dnn_model.evaluate(input_fn=eval_input_func)
# Predictions

predictions = dnn_model.predict(input_fn=eval_input_func)
pred = list(predictions)
# Plotting Predicted Values

predicted_vals = []



for pred in dnn_model.predict(input_fn=eval_input_func):

    predicted_vals.append(pred['predictions'])
print(predicted_vals)
# Calculate the Mean Squared Error

mse = mean_squared_error(predicted_vals, y_test)

print('Mean Squared Error [DNNRegrssor]: ',mse)


import matplotlib.pyplot as plt



def plot_history(history):

  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error [1000$]')

  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),

           label='Train Loss')

  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),

           label = 'Val loss')

  plt.legend()

  plt.ylim([0,7])
# RMSPropOptimizer

Epochs=500

Optimizer=tf.train.RMSPropOptimizer(0.001)

#Optimizer='adam'

Loss="mse"

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(16,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(1))



model.compile(optimizer=Optimizer,loss=Loss,metrics=['mae'])



earlystop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30)



history=model.fit(X_train,y_train,epochs=Epochs,validation_split=0.2,verbose=2,callbacks=[earlystop])

plot_history(history)
loss,mae=model.evaluate(X_test,y_test)

print(' loss=' ,loss , '\n mae=', mae)


Epochs=500

#Optimizer=tf.train.RMSPropOptimizer(0.001)

Optimizer=tf.train.AdamOptimizer(0.001)

Loss="mse"

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(16,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(1))



model.compile(optimizer=Optimizer,loss=Loss,metrics=['mae'])



earlystop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30)



history=model.fit(X_train,y_train,epochs=Epochs,validation_split=0.2,verbose=2,callbacks=[earlystop])

plot_history(history)
loss,mae=model.evaluate(X_test,y_test)

print(' loss=' ,loss, '\n mae=', mae)
all_price_sum = y_train.sum()+y_test.sum()

Number_of_prices = y_train.size+y_test.size  #number of all labels

print("Sum of All Price =", all_price_sum)

print("Count of Prices=", Number_of_prices)

print("Average of prices of all the data = ",all_price_sum/Number_of_prices)