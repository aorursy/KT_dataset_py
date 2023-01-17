# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline



import matplotlib.pyplot as plt
rides_df = pd.read_csv("/kaggle/input/bike-sharing-dataset/hour.csv")

rides_df.head()
rides_df.tail()
rides_df[:125].plot(x='dteday', y='cnt')

plt.xticks([]);
dummy_variables = ['season', 'weathersit', 'mnth', 'hr', 'weekday']

for variable in dummy_variables:

    dummies = pd.get_dummies(rides_df[variable], prefix=variable, drop_first=False)

    rides_df = pd.concat([rides_df, dummies], axis=1)
variables_to_drop = ['instant', 'dteday', 'season', 'weathersit', 

                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']

data = rides_df.drop(variables_to_drop, axis=1)

data.head()
#features which needs Normalization

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

# Store scalings in a dictionary so we can convert back later

scaled_features = {}

for feature in quant_features:

    mean, std = data[feature].mean(), data[feature].std()

    scaled_features[feature] = [mean, std]

    data.loc[:, feature] = (data[feature] - mean)/std
# Save data for approximately the last 25 days 

test_data = data[-25*24:]



# Now remove the test data from the data set 

data = data[:-25*24]



# Separate the data into features and targets

target_fields = ['cnt', 'casual', 'registered']



features  = data.drop(target_fields, axis=1)

targets = data[target_fields]



test_features  = test_data.drop(target_fields, axis=1) 

test_targets = test_data[target_fields]
# Hold out the last 60 days or so of the remaining data as a validation set

train_features = features[:-60*24]

train_targets =  targets[:-60*24]



val_features = features[-60*24:]

val_targets = targets[-60*24:]
class NeuralNetwork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        # Set number of nodes in input, hidden and output layers.

        self.input_nodes = input_nodes

        self.hidden_nodes = hidden_nodes

        self.output_nodes = output_nodes



        # Initialize weights

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,(self.hidden_nodes, self.output_nodes))

        self.lr = learning_rate

        # Replace 0 with your sigmoid calculation.

        self.activation_function = lambda x : 1 / (1 + np.exp(-x))                      



    def train(self, features, targets):

        

        n_records = features.shape[0]

        

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)

        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        

        for X, y in zip(features, targets):

            # Implement the forward pass function below

            final_outputs, hidden_outputs = self.forward_pass_train(X)  

            # Implement the backproagation function below

            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)





    def forward_pass_train(self, X):

        # signals into hidden layer

        hidden_inputs = np.dot(X,self.weights_input_to_hidden) 

        # signals from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)        



        # signals into final output layer

        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) 

        # signals from final output layer

        final_outputs = final_inputs 

        

        return final_outputs, hidden_outputs



    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        # Output layer error is the difference between desired target and actual output.

        error = y-final_outputs 

        hidden_error = np.dot(self.weights_hidden_to_output, error)

        output_error_term = error * 1.0

        

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        

        delta_weights_i_h += hidden_error_term * X[:, None]

        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o



    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):

        # update hidden-to-output weights with gradient descent step

        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records

        # update input-to-hidden weights with gradient descent step

        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records 

        

    def run(self, features):

        # signals into hidden layer

        hidden_inputs = np.dot(features, self.weights_input_to_hidden)

        # signals from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs) 

        

        # signals into final output layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

         # signals from final output layer

        final_outputs = final_inputs  

        

        return final_outputs

#hyperparameters

iterations = 200

learning_rate = 0.1

hidden_nodes = 5

output_nodes = 1
#Mean Square Error

def MSE(y, Y):

    return np.mean((y-Y)**2)
import sys



N_i = train_features.shape[1]

#Build a Network Object

network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

#Store Loss for training and validation.

losses = {'train':[], 'validation':[]}

for i in range(iterations):

    # Go through a random batch of 128 records from the training data set

    batch = np.random.choice(train_features.index, size=128)

    X, y = train_features.loc[batch].values, train_targets.loc[batch]['cnt']

                             

    network.train(X, y)

    

    # Printing out the training progress

    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)

    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)

    sys.stdout.write("\rProgress: {:2.1f}".format(100 * i/float(iterations)) \

                     + "% ... Training loss: " + str(train_loss)[:5] \

                     + " ... Validation loss: " + str(val_loss)[:5])

    sys.stdout.flush()

    

    losses['train'].append(train_loss)

    losses['validation'].append(val_loss)
plt.plot(losses['train'], label='Training loss')

plt.plot(losses['validation'], label='Validation loss')

plt.legend()

_ = plt.ylim()
fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']

#predict on test data 

predictions = network.run(test_features).T*std + mean

ax.plot(predictions[0], label='Prediction')

ax.plot((test_targets['cnt']*std + mean).values, label='Data')

ax.set_xlim(right=len(predictions))

ax.legend()



dates = pd.to_datetime(rides_df.loc[test_data.index]['dteday'])

dates = dates.apply(lambda d: d.strftime('%b %d'))

ax.set_xticks(np.arange(len(dates))[12::24])

_ = ax.set_xticklabels(dates[12::24], rotation=45)