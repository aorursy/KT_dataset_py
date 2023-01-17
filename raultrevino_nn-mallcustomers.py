import pandas as pd

import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os 

import matplotlib.pyplot as plt
data = pd.read_csv( "../input/200x5-mall-customers-data/Mall_Customers_Data.csv")
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
data.head()
data.tail()
del data['CustomerID']

data.columns = ['Gender', 'Age', 'AnnualIncome', 'SpendingScore']

data.loc[(data.Gender == "Male"),'Gender'] = 0 

data.loc[(data.Gender == "Female"),'Gender'] = 1
data.shape
data.describe()
data.head()
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

data = pd.DataFrame(min_max_scaler.fit_transform(data))

data.columns = ['Gender', 'Age', 'AnnualIncome', 'SpendingScore']
data
from sklearn.model_selection import train_test_split
data_vars = data.columns.values.tolist()

Y = ['SpendingScore']

X = [v for v in data_vars if v not in Y]

X_train, X_test, Y_train, Y_test = train_test_split(data[X],data[Y],test_size = 0.3, random_state=0)
batch_size = 30

# Input and target variables

x_data = tf.placeholder(shape=[None,3],dtype = tf.float32)

y_target = tf.placeholder(shape=[None, 1], dtype = tf.float32)



# Define Hidden Layer

hidden_layer_nodes = 5

A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes])) #Three Inputs and five Nodes

b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes])) # Five one foreach node



A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))

b2 = tf.Variable(tf.random_normal(shape=[1]))



hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))

final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))
loss = tf.reduce_mean(tf.square(y_target-final_output))
my_optim = tf.train.GradientDescentOptimizer(0.005)

train_step = my_optim.minimize(loss)
init = tf.global_variables_initializer()

session = tf.Session()

session.run(init)
loss_vect = []

test_loss = []



for i in range(500):

    # Get the random indexes 

    rand_idx = np.random.choice(len(X_train), size=batch_size)

    X_train_array = X_train.to_numpy()

    Y_train_array = Y_train.to_numpy()

    rand_x = X_train_array[rand_idx]

    rand_y = Y_train_array[rand_idx]

    

    # Train the nerula Network

    session.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})

    

    # Get the temp_loss for training data  for  later make plot 

    temp_loss = session.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})

    loss_vect.append(np.sqrt(temp_loss))

    

    # Get the temp_loss for testing data  for  later make plot 

    temp_loss_test = session.run(loss, feed_dict={x_data: X_test.to_numpy(),y_target:Y_test.to_numpy()})

    test_loss.append(np.sqrt(temp_loss_test))

    

    # Every 50 steps we check the loss reduce or increase

    if(i+1)%50==0:

        print("Paso #"+str(i+1)+", Loss = "+str(temp_loss))

       

    
plt.plot(loss_vect, "r-", label="Training Loss")

plt.plot(test_loss, "b--", label ="Test Loss")

plt.title("Lost (RMSE) per iteration")

plt.xlabel("Iteration")

plt.ylabel("RMSE")

plt.legend(loc ="upper right")

plt.show()
data.head()
test_pred = [x[0] for x in session.run(final_output, feed_dict={x_data:X_test.to_numpy()})]
results_data_frame = pd.DataFrame()

results_data_frame['prediction']= test_pred

results_data_frame['real_value'] = Y_test['SpendingScore'].tolist()
results_data_frame.columns = ['prediction','real_value']
results_data_frame.head()
min = 1

max = 100

results_data_frame["prediction"] = [ (x*(max - min) + min) for x in results_data_frame["prediction"]]

results_data_frame["real_value"] = [ (x*(max - min) + min) for x in results_data_frame["real_value"]]
results_data_frame