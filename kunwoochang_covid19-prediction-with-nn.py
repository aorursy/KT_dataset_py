# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
%matplotlib inline
%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
data_path = '/kaggle/input/covid19-global-forecasting-week-4/train.csv'

data_train = pd.read_csv(data_path)
display(data_train.head())
display(data_train.describe())
display(data_train.info())
confirmed_total_date_Italy = data_train[data_train['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Italy = data_train[data_train['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)

plt.figure(figsize=(17,10))
plt.subplot(2, 2, 1)
total_date_Italy.plot(ax=plt.gca(), title='Italy')
plt.ylabel("Confirmed infection cases", size=13)

confirmed_total_date_US = data_train[data_train['Country_Region']=='US'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_US = data_train[data_train['Country_Region']=='US'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_US = confirmed_total_date_US.join(fatalities_total_date_US)

#plt.figure(figsize=(17,10))
plt.subplot(2, 2, 2)
total_date_US.plot(ax=plt.gca(), title='US')
plt.ylabel("Confirmed infection cases", size=13)

confirmed_total_date_Spain = data_train[data_train['Country_Region']=='Spain'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Spain = data_train[data_train['Country_Region']=='Spain'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Spain = confirmed_total_date_Spain.join(fatalities_total_date_Spain)

#plt.figure(figsize=(17,10))
plt.subplot(2, 2, 3)
total_date_Spain.plot(ax=plt.gca(), title='Spain')
plt.ylabel("Confirmed infection cases", size=13)

confirmed_total_date_Korea = data_train[data_train['Country_Region']=='Korea, South'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Korea = data_train[data_train['Country_Region']=='Korea, South'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Korea = confirmed_total_date_Korea.join(fatalities_total_date_Korea)

#plt.figure(figsize=(17,10))
plt.subplot(2, 2, 4)
total_date_Korea.plot(ax=plt.gca(), title='Korea, South')
plt.ylabel("Confirmed infection cases", size=13)

# Plots
plt.figure(figsize=(17,10))
plt.subplot(2, 2, 1)
plt.plot(confirmed_total_date_Italy)
plt.plot(confirmed_total_date_US)
plt.plot(confirmed_total_date_Spain)
plt.plot(confirmed_total_date_Korea)
plt.legend(["Italy", "US", "Spain", "Korea, South"], loc='upper left')
plt.title("COVID-19 infections from the first confirmed case", size=15)
plt.xlabel("Days", size=13)
plt.ylabel("Infected cases", size=13)
plt.ylim(0, 180000)
#plt.show()

# Plots
#plt.figure(figsize=(12,6))
plt.subplot(2, 2, 2)
plt.plot(fatalities_total_date_Italy)
plt.plot(fatalities_total_date_US)
plt.plot(fatalities_total_date_Spain)
plt.plot(fatalities_total_date_Korea)
plt.legend(["Italy", "US", "Spain", "Korea, South"], loc='upper left')
plt.title("COVID-19 Fatalities", size=15)
plt.xlabel("Days", size=13)
plt.ylabel("Infected cases", size=13)
plt.ylim(0, 23000)
plt.show()
def add_daily_measures(df):
    df.loc[0,'Daily Cases'] = df.loc[0,'ConfirmedCases']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Fatalities']
    for i in range(1,len(df)):
        df.loc[i,'Daily Cases'] = df.loc[i,'ConfirmedCases'] - df.loc[i-1,'ConfirmedCases']
        df.loc[i,'Daily Deaths'] = df.loc[i,'Fatalities'] - df.loc[i-1,'Fatalities']
    #Make the first row as 0 because we don't know the previous value
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    return df
df_world = data_train.copy()
df_world = df_world.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_world = add_daily_measures(df_world)
df_world.plot(title ='Covid19 World daily status', y=['Daily Cases','Daily Deaths'], x='Date', figsize=(12,6))
# USA
df_usa = data_train.query("Country_Region=='US'")
df_usa = df_usa.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_usa = add_daily_measures(df_usa)

#Italy
df_italy = data_train.query("Country_Region=='Italy'")
df_italy = df_italy.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_italy = add_daily_measures(df_italy)

#Spain
df_spain = data_train.query("Country_Region=='Spain'")
df_spain = df_spain.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_spain = add_daily_measures(df_spain)

#Korea
df_korea = data_train.query("Country_Region=='Korea, South'")
df_korea = df_korea.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_korea = add_daily_measures(df_korea)


df_usa.plot(title = "USA", y=['Daily Cases','Daily Deaths'], x='Date', figsize=(12,6))

df_italy.plot(title = "Italy", y=['Daily Cases','Daily Deaths'], x='Date', figsize=(12,6))

df_spain.plot(title = "Spain", y=['Daily Cases','Daily Deaths'], x='Date', figsize=(12,6))

df_korea.plot(title = "South Korea", y=['Daily Cases','Daily Deaths'], x='Date', figsize=(12,6))
data_flight = pd.read_csv('/kaggle/input/covid19/covid19_flight_countries_mod.csv')

data_flight.head()
data_flight.plot(x='Date', figsize=(12,6))
#full-list-covid-19-tests-per-day
data_daily_tested = pd.read_csv('/kaggle/input/covid19/full-list-covid-19-tests-per-day.csv')

data_daily_tested.head()
data_daily_tested.plot(x='Date', figsize=(12,6))
df_usa_tested = data_daily_tested.query("Code=='USA'")
df_italy_tested = data_daily_tested.query("Entity=='Italy'")
df_spain_tested = data_daily_tested.query("Entity=='Spain'")
df_korea_tested = data_daily_tested.query("Entity=='South Korea'")
#df_usa_tested.head()
#df_italy_tested.head()
#df_spain_tested.head()
#df_korea_tested.head()
df_usa_tested.plot(title='USA', x='Date', figsize=(12,6))
df_italy_tested.plot(title='Italy', x='Date', figsize=(12,6))
df_korea_tested.plot(title='Korea', x='Date', figsize=(12,6))
df_usa_merge = pd.merge(df_usa, df_usa_tested)
df_usa_merge = pd.merge(df_usa_merge, data_flight)
df_usa_merge.head()
# Currently, only consider USA data
df_usa_data = df_usa_merge.drop(['Date', 'Entity', 'Code', 'US <-> Latin America', 'US <-> China', 'Canada <-> Canada', 'Canada <-> NON Canada', 'Europe <-> Europe', 'Europe <-> UK', 'Europe <-> Latin America', 'UK <-> UK', 'UK <-> NON UK', 'Italy <-> Italy', 'China <-> China', 'Brazil <-> Brazil', 'Brazil <-> NON Brazil', 'India <-> India', 'India <-> NON India', 'Iran <-> Iran'], axis=1)
df_usa_data.head()
df_usa_data.plot(figsize=(12,6))
quant_features = ['Daily Cases', 'Daily change in cumulative total tests', 'US <-> US', 'US <-> NON US','US <-> Europe', 'ConfirmedCases']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = df_usa_data[each].mean(), df_usa_data[each].std()
    scaled_features[each] = [mean, std]
    df_usa_data.loc[:, each] = (df_usa_data[each] - mean)/std

df_usa_data.head()  
# Save data for approximately the last 15 days 
test_data = df_usa_data[-15:]

# Now remove the test data from the data set 
data = df_usa_data[:-15]

# Separate the data into features and targets
target_fields = ['Daily Cases', 'Daily change in cumulative total tests', 'ConfirmedCases']
features, targets = df_usa_data.drop(target_fields, axis=1), df_usa_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-43], targets[:-43]
val_features, val_targets = features[-43:], targets[-43:]
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
        
        # Sigmoid activation function
        self.activation_function = lambda x : (1/(1+np.exp(-x)))  
                    
    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X) 
            
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):

        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        
        error = y-final_outputs # Output layer error is the difference between desired target and actual output.
        
        # The hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, error)
        
        #Backpropagated error terms
        output_error_term = error * 1
        
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:,None]
        # Weight step (hidden to output)
        delta_weights_h_o += (output_error_term * hidden_outputs[:,None])
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_records 

        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_records 

    def run(self, features):
        
        #Hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        #Output layer 
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs
def MSE(y, Y):
    return np.mean((y-Y)**2)
#Hyperparameter

iterations = 1000
learning_rate = 0.3
hidden_nodes = 7
output_nodes = 1
import sys

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['Daily Cases']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['Daily Cases'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['Daily Cases'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
fig, ax = plt.subplots(figsize=(16,6))

mean, std = scaled_features['Daily Cases']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['Daily Cases']*std + mean).values, label='Daily Cases')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(df_usa_merge.iloc[test_data.index]['Date'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates)))
_ = ax.set_xticklabels(dates, rotation=45)
