import pandas as pd                              

import matplotlib.pyplot as plt                  

import numpy as np                               

import datetime as dt

import matplotlib

%matplotlib inline

 

import os

#Reading Mercedes-Benz given Data Set.

import os

for dirname, _, filenames in os.walk('kaggle/input/mystock.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading Tesla Stock Price Data Set

df_tesla=pd.read_csv('/kaggle/input/mystock.csv') 
# Describe method is used to view some basic statistical details like percentile, mean, std etc. of a data frame of numeric values.

df_tesla.describe()
# Given data set is not having any null values.

df_tesla.info()
df_tesla.head(2)
df_tesla.tail(2)
#Renamed column from Adj Close to Adj_Close to read data properly from Dataframe

df_tesla.rename(columns = {'Adj Close':'Adj_Close'}, inplace = True) 
#Verified column name after renaming it.

df_tesla.head(1)
#Verify whether their is any difference between Close and 'Adj_Close' Stock price before proceeding with Prediction

print(np.var(df_tesla.Close-df_tesla.Adj_Close))
#Drop Adj_Close column which is not required here.

df_tesla.drop('Adj_Close',axis=1,inplace=True)
# Verify columns

df_tesla.head(1)
# Visualizations Heading: Verifying the Close Price for the given Data Set

# Plot Used   : Plot

# Description : Visualize Close Price how stock price increases since 2011.

# Outcome     : a) Around Q2-2013 you can observe increase in Stock price 

#               b) Around Q1-2016 Stock price decrease due to some reason and again started picking up after some days immediately.

plt.figure(figsize=[10,4])                                   # Setting the figure size

plt.plot(df_tesla['Close'])                            # Plotting the close price

plt.xlabel("Date")                                           # Setting the label in x-axis

plt.ylabel("Close Price")                                    # Setting the label in y-axis

plt.title("Close Price Vs Date")                             # Title

plt.show()
# Visualizations Heading: Verifying Open and Close Price for the given Data Set

# Plot Used   : Regression Plot

# Description : Visualize how often Open and Close Price variables are closely related to each other via regression plot.

# Outcome     : a) Both Open and Close Stock price joint distribution of each other. 

import seaborn as sns

sns.set_style('whitegrid') 

sns.regplot(x ='Open', y ='Close', data = df_tesla,marker='+',color="g") 
# Visualizations Heading: Verifying how the given Data Set form regression and distributions of Data points.

# Plot Used   : Pair Plot

# Description : Visualize how the data points has been distributed between variables.

# Outcome     : a) Both Open and Close Stock price form clear regression here. 

#                b) Used Kernel density estimates to see the distribution between vairables. Mostly you can see bi-normal and right skewed distribution.

sns.pairplot(df_tesla,corner=True,diag_kind="kde")
df_tesla.head(1)
# Take a copy of df_tesla data set and create new dataframe for more visualizations

df_visual = df_tesla.copy()
# Excluded the volume column from the given columns and only considered below columns for visualizations

column_list=['Date','Open','High','Low','Close']

df_visual=df_visual[column_list]    
# Converted Date datatype from objec to datetime.

df_visual['Date'] = pd.to_datetime(df_visual['Date'])
# Created index on Date Column

df_visual = df_visual.set_index("Date")
# Verify sample data.

df_visual.head(1)
# Visualizations Heading: Verifying how the given Data Set form regression and distributions of Data points.

# Plot Used   : Rolling Plot

# Description : Rolling statistics are a third type of time series-specific operation implemented by Pandas. 

#               These can be accomplished via the rolling() attribute of Series and DataFrame objects, 

#               which returns a view similar to what we saw with the groupby operation (see Aggregation and Grouping). 

#               This rolling view makes available a number of aggregation operations by default.

# Outcome     : a) Both Open and Close Stock price how closely related to each other during periods. 





plt.figure(figsize=(30,8))

df_visual.rolling(50, center=True,win_type='gaussian').sum(std=10).plot(style=[':', '-', '+','.']);
# Visualizations Heading: Verifying stock price(mean) fluctuvate during week days.

# Plot Used   : Plot

# Description : Visualize how stock price(mean) performs during week days. 

# Outcome     : a) Observe mostly on  monday and tuesday the Close price is high when compare to Open price.

#               b) On Wed, Thurs and Fri Close price is low whenn compare to Open price.

#               c) Mostly, On Mon & Tuesday the stock price reached very high when compare to other days.

#               d) Mostly, on Friday it is good for new investors to purchase stocks.

by_weekday = df_visual.groupby(df_visual.index.dayofweek).mean()

by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri']

by_weekday.plot(style=[':', '-', '+','*']);
# Taken below required columns to proceed with ClosePprice predictions analysis

column_list=['Date','Close']

df_tesla_final=df_tesla[column_list]                                                                         

df_tesla_final.head(2)
df_tesla_final.info()
#Convert Date column data type from object to Date.

df_tesla_final['Date'] = pd.to_datetime(df_tesla_final['Date'])
df_tesla_final.info()
#Set index on Date Column

df_tesla_final = df_tesla_final.set_index("Date")

df_tesla_final.head(2)
# Given DataSet contains data from 29-Jun-2010 to 17-Mar-2017

df_tesla_final.tail(2)
# Import more packages for price predictions

import keras

from keras.models import Sequential                   # Sequential model

from keras.layers import Dense                        # For fully connected layers

from keras.layers import LSTM                         # For LSTM layers

from sklearn.preprocessing import MinMaxScaler        # Scaling the data

min_max_scaler = MinMaxScaler()

#Verify the length of final dataset

len(df_tesla_final)
# Verify whether the final set of dataframe contains any null values or not

df_tesla_final.isnull().sum()                  
# Lets make predictions for last 30 days

# Remove last 1662 days from final dataset and load it in train and 30 days data for test.

prediction_days = 1662

ts_test= df_tesla_final[prediction_days:]       # load 30 days values to test by removing 1662 records.

ts_train= df_tesla_final[:prediction_days]      # load 1662 days values for train
#Verify complete data for the given data set.

df_tesla_final.shape
#Verify data whether train data which contains by excluding last 30 days (1692-30 = 1662)

ts_train.shape
#Verify data whether test data contains only last 30 days

ts_test.shape
#Verify sample data whether test data contains latest dates close price or not.

ts_test.head(2)
ts_test.tail(2)
# Visualizations Heading: Verifying distribution of data between Train and Test Splits

# Plot Used   : Plot

# Description : Visualize how stock close price splitted between Train and Test Data set.

# Outcome     : a) First 1662 days data has splitted as a Train Data Set shows in blue color.

#               b) Last 30 days data has splitted as a Train Data Set shows in cyan color.

plt.figure(figsize = (18,9))

plt.plot(range(df_tesla_final.shape[0]),df_tesla_final,color='cyan', label='Test Data')

plt.plot(range(ts_train.shape[0]),ts_train,color='blue',label='Train Data')

plt.xlabel('No of Days Range')

plt.ylabel('Close Price')

plt.legend(fontsize=18)

plt.show()
# Scaling the train data

training_set= ts_train.values

training_set = min_max_scaler.fit_transform(training_set)
# Defining our X and Y. X is our inputs which is the training data. y is the output, which is training data shifted by 1. 

# For a given day, we want to predict the stock value for the next day.

x_train=training_set[0:len(training_set)-1]

y_train=training_set[1:len(training_set)]
x_train[:5]                                 # Checking the first 5 rows of the scaled training data
y_train[:5]                                                  # Checking the first 5 rows of the scaled test data
len(x_train)                                                 # We should have same number of records in x_train and y_train
len(y_train)                                                 # We should have same number of records in x_train and y_train
print(y_train)
x_train = np.reshape(x_train, (len(x_train), 1, 1))          # We need to reshape the data before it is passed to the model
# Hyper Parameter initializations

num_units = 32

activation_function = 'sigmoid'       #  Used through an activation layer

optimizer = 'adamax'                  # An optimizer is one of the two arguments required for compiling a Keras model

loss_function = 'mean_absolute_error' # A loss function (or objective function, or optimization score function) is one of the two parameters required to compile a model.

batch_size = 5

num_epochs = 50
# Training the data from 29-06-2010 to 13-03-2017

# Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.



regressor = Sequential()



# Adding the input layer and the LSTM layer

regressor.add(LSTM(units = num_units, activation = activation_function, input_shape=(None, 1)))



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = optimizer, loss = loss_function)



# Using the training set to train the model

regressor.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs)
# Predictions on ts_test

# Preprocess the test data

test_set = ts_test.values



inputs = np.reshape(test_set, (len(test_set), 1))                        # Reshape before passing in the input

inputs = min_max_scaler.transform(inputs)                                # Scaling the data

inputs = np.reshape(inputs, (len(inputs), 1, 1))



predicted_price = regressor.predict(inputs)                              # Make predictions on the test data

predicted_price = min_max_scaler.inverse_transform(predicted_price)      # Inverse transform the predicted price
test_set.shape
# This is the close price of stocks for 30 days

predicted_price  
# Calculate the error

error=predicted_price-test_set                                          
# Lets have a look at the error values

error
# Plot the Actual price and the predicted price

# Visualizations Heading: Plot Actual Close Price and Predicted Close Price

# Plot Used   : Plot

# Description : Visualize how stock close price splitted between Train and Test Data set.

# Outcome     : a) First 1662 days data has splitted as a Train Data Set shows in blue color.

#               b) Last 30 days data has splitted as a Train Data Set shows in cyan color.



plt.figure(figsize=(8, 8), dpi=80, facecolor = 'w', edgecolor = 'k')

plt.plot(test_set[:, 0], color='red', label='Real Tesla Price')                      # Actual Price

plt.plot(predicted_price[:, 0], color = 'blue', label = 'Predicted Close Price')      # Predicted Price

plt.title('Tesla Price Prediction from 3-02-2017 to 13-03-2017', fontsize = 20)

plt.xlabel('Time', fontsize=40)

plt.ylabel('Tesla Price', fontsize = 40)

plt.legend(loc = 'best')

plt.show()