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

%load_ext autoreload

%autoreload 2

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_path = '../input/hour.csv'



rides = pd.read_csv(data_path)
rides.head()
rides[:24*10].plot(x='dteday', y='cnt')
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']

for each in dummy_fields:

    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)

    rides = pd.concat([rides, dummies], axis=1)



fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 

                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']

data = rides.drop(fields_to_drop, axis=1)

data.head()
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

# Store scalings in a dictionary so we can convert back later

scaled_features = {}

for each in quant_features:

    mean, std = data[each].mean(), data[each].std()

    scaled_features[each] = [mean, std]

    data.loc[:, each] = (data[each] - mean)/std
# Save data for approximately the last 21 days 

test_data = data[-21*24:]



# Now remove the test data from the data set 

data = data[:-21*24]



# Separate the data into features and targets

target_fields = ['cnt', 'casual', 'registered']

features, targets = data.drop(target_fields, axis=1), data[target_fields]

test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
#Hold out the last 60 days or so of the remaining data as a validation set

train_features, train_targets = features[:-60*24], targets[:-60*24]

val_features, val_targets = features[-60*24:], targets[-60*24:]
class NeuralNetwork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        # Set number of nodes in input, hidden and output layers.

        self.input_nodes = input_nodes

        self.hidden_nodes = hidden_nodes

        self.output_nodes = output_nodes



        # Initialize weights

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 

                                       (self.input_nodes, self.hidden_nodes))



        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 

                                       (self.hidden_nodes, self.output_nodes))

        self.lr = learning_rate

        

       

        #

        # Note: in Python, we can define a function with a lambda expression,

        # as shown below.

        self.activation_function = lambda x : 1/(1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

        

        ### If the lambda code above is not something you're familiar with,

        # You can uncomment out the following three lines and put your 

        # implementation there instead.

        #

        #def sigmoid(x):

        #    return 0  # Replace 0 with your sigmoid calculation here

        #self.activation_function = sigmoid

                    



    def train(self, features, targets):

        ''' Train the network on batch of features and targets. 

        

            Arguments

            ---------

            

            features: 2D array, each row is one data record, each column is a feature

            targets: 1D array of target values

        

        '''

        n_records = features.shape[0]

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)

        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):

            

            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below

            # Implement the backproagation function below

            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 

                                                                        delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)





    def forward_pass_train(self, X):

        ''' Implement forward pass here 

         

            Arguments

            ---------

            X: features batch

        '''

        #### Implement the forward pass here ####

        ### Forward pass ###

        

        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer

        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer



       

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer

        final_outputs = final_inputs # signals from final output layer

        

        return final_outputs, hidden_outputs



    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        ''' Implement backpropagation

         

            Arguments

            ---------

            final_outputs: output from forward pass

            y: target (i.e. label) batch

            delta_weights_i_h: change in weights from input to hidden layers

            delta_weights_h_o: change in weights from hidden to output layers

        '''

        #### Implement the backward pass here ####

        ### Backward pass ###



       

        error = y - final_outputs # Output layer error is the difference between desired target and actual output.

        

        

        hidden_error = np.dot(self.weights_hidden_to_output, error) 

        

       

        output_error_term = error

        

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        

        # Weight step (input to hidden)

        delta_weights_i_h += hidden_error_term * X[:,None]

        # Weight step (hidden to output)

        delta_weights_h_o += output_error_term * hidden_outputs[:,None]

        return delta_weights_i_h, delta_weights_h_o



    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):

        ''' Update weights on gradient descent step

         

            Arguments

            ---------

            delta_weights_i_h: change in weights from input to hidden layers

            delta_weights_h_o: change in weights from hidden to output layers

            n_records: number of records

        '''

        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step

        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step



    def run(self, features):

        ''' Run a forward pass through the network with input features 

        

            Arguments

            ---------

            features: 1D array of feature values

        '''

        

        #### Implement the forward pass here ####

       

        hidden_inputs =  np.dot(features, self.weights_input_to_hidden)

        hidden_outputs = self.activation_function(hidden_inputs)

        

       

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        final_outputs = final_inputs 

        

        return final_outputs

#########################################################

# Set your hyperparameters here

##########################################################

iterations = 5000

learning_rate = 0.5

hidden_nodes = 20

output_nodes = 1
def MSE(y, Y):

    return np.mean((y-Y)**2)
import sys

N_i = train_features.shape[1]

network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)



losses = {'train':[], 'validation':[]}

for ii in range(iterations):

    # Go through a random batch of 128 records from the training data set

    batch = np.random.choice(train_features.index, size=128)

    X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['cnt']

                             

    network.train(X, y)

    

    # Printing out the training progress

    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)

    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)

    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \

                     + "% ... Training loss: " + str(train_loss)[:5] \

                     + " ... Validation loss: " + str(val_loss)[:5])

    sys.stdout.flush()

    

    losses['train'].append(train_loss)

    losses['validation'].append(val_loss)
fig, ax = plt.subplots(figsize=(8,4))



mean, std = scaled_features['cnt']

predictions = network.run(test_features).T*std + mean

ax.plot(predictions[0], label='Prediction')

ax.plot((test_targets['cnt']*std + mean).values, label='Data')

ax.set_xlim(right=len(predictions))

ax.legend()



dates = pd.to_datetime(rides.iloc[test_data.index]['dteday'])

dates = dates.apply(lambda d: d.strftime('%b %d'))

ax.set_xticks(np.arange(len(dates))[12::24])

_ = ax.set_xticklabels(dates[12::24], rotation=45)