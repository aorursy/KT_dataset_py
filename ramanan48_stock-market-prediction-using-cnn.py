import  numpy as np#Used for fast mathematical function processing
import pandas as pd#Used for Datascience processes to make indexing easy
import matplotlib.pyplot as plt#Machine Learning Library to create plots and data visualizations
import tensorflow as tf#It is an end-to end Machine learning library used to develop,train and deploy models. 
from tensorflow import keras#It is a neural network library which is more user friendly
from sklearn.preprocessing import RobustScaler#used to scale the values based on Inter-Quartile range(25%to75%)
#The dataset is given in csv format,This line is used to import the csv file containing Microsoft stock values
#Here 'data' is the Dataframe
data=pd.read_csv('../input/stock-price-prediction/MSFT.csv',index_col='Date',parse_dates=['Date'])
#print first  5 rows
data.head()
#print last 5 rows
data.tail()
#creating training and testing time
train_time=int(len(data)*0.7)#70% of the length of original dataset is taken as training time
test_time=len(data)-train_time#Remaining 30% of the length of original dataset is taken as testing time
#spliting of original dataset into training dataset and testing dataset
x_train=data[:train_time]#The original dataset starting from 0 to training time is taken as training dataset
x_test=data[train_time:]# The original dataset starting after the training time is taken as testing dataset
#Get the shape and type of train_data
print("Shape of train data - {}".format(x_train.shape))#print the shape of training data
print("Type of train data - {}".format(type(x_train)))#print the shape of testing data.

#convert the scale on inter-quartile range
scaler=RobustScaler()#create object for class Robust Scaler.
#It is used to scale the values and is robust to outliers as it eliminates outliers by taking 25% to 75% of data
x_scaled=scaler.fit_transform(x_train)#It is used to transform the training dataset to scaled values.
x=[]#empty list
y=[]#empty list
#here the features are first 7 days and output to be predicted is 8th day
for i in range(7,x_scaled.shape[0]):#looping on the dataset for 7 days
  x.append(x_scaled[i-7:i])#appending first 7 days windowed list to x as features. 
  y.append(x_scaled[i,0])#appending only the opening price of first 7 days windowed list to y as target labels.
#convert into numpy array
x_window_set,y_window_set=np.array(x),np.array(y)#convert the list x and y to numpy arrays
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=6, kernel_size=30,
                      strides=1, padding="same",
                      activation="relu",
                      input_shape=[x_window_set.shape[1], 6]),#First convolutional layer
  tf.keras.layers.Activation('relu'),#First activation layer
  tf.keras.layers.AveragePooling1D(pool_size=2, strides=None, padding="valid"),#First pooling layer

  tf.keras.layers.Conv1D(filters=4, kernel_size=15,
                      strides=1, padding="same",
                      activation="relu"),#second convolutional layer
  tf.keras.layers.Activation('relu'),#second activation layer
  tf.keras.layers.AveragePooling1D(pool_size=2, strides=None, padding="valid"),#second pooling layer

  tf.keras.layers.Conv1D(filters=2, kernel_size=7,
                      strides=1, padding="same",
                      activation="relu"),#third convolutional layer
  tf.keras.layers.Activation('relu'),#third activation layer

  tf.keras.layers.Flatten(),#First flatten layer.It is used to flatten the matrix of data.
  tf.keras.layers.Dense(32),#First dense layer.
  tf.keras.layers.Dense(1)#Output layer

  
])

model.summary()#To show the layers, output shape and parameters used.
model.compile(loss='mse',#loss used is mean squared error.
              optimizer='adam',#Optimizer used is Adam,since it smooths the oscillation of gradient and can self-tune the learning rate
              metrics=["mae"])#metric is mean absolute error.
model.fit(x_window_set,y_window_set, epochs=100,batch_size=32 )#the model is run on scaled training data
#It is run for 100 epochs with batch size of 32 .
x_real_test=data[train_time-7:]#create dataset
x_a_real_test=np.array(x_real_test)#convert into numpy array
x_t_scaled=scaler.transform(x_a_real_test)#scale the values using the Robust scaler parameters .
x_t=[]#empty list
y_t=[]#empty list
#here the features are first 7 days and output to be predicted is 8th day
for i in range(7,x_t_scaled.shape[0]):#looping on the dataset for 7 days
  x_t.append(x_t_scaled[i-7:i])#appending first 7 days windowed list to x_t as features. 
  y_t.append(x_t_scaled[i,0])#appending only the opening price of first 7 days windowed list to y_t as target labels.
x_test_window_set,y_test_window_set=np.array(x_t),np.array(y_t)#convert the list to numpy array
predicted_value=model.predict(x_test_window_set)#Predict the value of test dataset features
#calculating Mean Absolute Percentage error (MAPE)
m = tf.keras.metrics.MeanAbsolutePercentageError()#initialize the MAPE class.
m.update_state(y_test_window_set,predicted_value)#calculate MAPE on Actual testing values and forecasted values
m.result().numpy()#print the result
time=np.arange(1,2518)#create a range of numbers from 1 to 2518 
#visualizing the results
plt.figure(figsize=(10,6))#Set the figure size
plt.plot(time[1761:],y_test_window_set,color='blue',label="true values")#Plot the true testing value in blue
plt.plot(time[1761:],predicted_value,color='red',label='predicted value')#Plot the predicted value in red
plt.xlabel('Time in Days')#set label of x as Time
plt.ylabel('Stock Price')#set label of y as Value
plt.legend()#show the labels
plt.show()#show the plot