# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import tensorflow as tf # Tensor Flow Library.
import matplotlib.pyplot as plt # Plotting matplot library graphs
import seaborn as sns # Plotting graphs in seaborn
%matplotlib inline # Plots the graphs directly in Jupyter

from sklearn.preprocessing import MinMaxScaler # For normalization of Data
# Loading the data from csv files to Pandas Dataframe.
train_data_df=pd.read_csv("../input/train.csv")
stores_data_df=pd.read_csv("../input/store.csv")
test_data_df=pd.read_csv("../input/test.csv")
# Looking at training Data
train_data_df.head(5)
# Looking at sample data for Stores
stores_data_df.head(5)
test_data_df.head()
# Insert Null values with zeros or median values.
# I will see to fine tune it later, but lgoing with simplest approach for now
stores_data_df['CompetitionDistance'].fillna(stores_data_df['CompetitionDistance'].median(), inplace = True)

# Filling other null variables with zeros
stores_data_df.fillna(0, inplace = True)
# Merge both of the datasets.
train_store_df = pd.merge(train_data_df, stores_data_df, how = 'inner', on = 'Store')
print("Shape of Merged Dataset", train_store_df.shape)
train_store_df.head(5)
print("Start Date", min(train_store_df.Date))
print("End Date", max(train_store_df.Date))
train_store_df.Store.unique()
# Re-Run point.
m=2

train_store_Begin=train_store_df[(train_store_df['Store']==1)|(train_store_df['Store']==2)]
# train_store_Begin=train_store_Begin.set_index('Date')
# train_store_Begin.index = pd.to_datetime(train_store_Begin.index)

train_store_Begin.Date = pd.to_datetime(train_store_Begin.Date) # Converting Date column to Time Stamp
train_store_Begin=train_store_Begin[['Store', 'Date', 'Open', 'Promo','Sales']]
train_store_Begin=train_store_Begin[(train_store_Begin.Date>pd.to_datetime('2013-01-01')) &
                                    (train_store_Begin.Date<pd.to_datetime('2013-07-31')) 
                                    ]
train_store_Begin.shape
train_store_Begin=train_store_Begin.sort_values(by=['Store','Date'])
train_store_Begin=pd.concat([pd.get_dummies(train_store_Begin.Store, prefix='Store'),train_store_Begin],1)
train_store_Begin.head()
from sklearn.preprocessing import MinMaxScaler
sc_X=MinMaxScaler()
train_store_Begin=sc_X.fit_transform(train_store_Begin[['Store_1','Store_2','Store', 'Open','Promo','Sales']])
train_store_Begin=np.array(train_store_Begin)
train_store_Begin
train_store_Begin=train_store_Begin.reshape(m,int(train_store_Begin.shape[0]/2),6)
# Reversing the Array because the original values sorted by Date Function are in reverse Order.
# train_store_Begin=train_store_Begin.reindex(index=train_store_Begin.index[::-1])
# train_store_Begin=np.array(train_store_Begin[['Sales','Open', 'Promo', 'SchoolHoliday']])
# sc_X=MinMaxScaler()
# train_store_Begin=sc_X.fit_transform(train_store_Begin)
train_store_Begin=train_store_Begin[:,:,[0,1,3,4,5]]
train_store_Begin
## Reshaping the array to 3D Vector: The shape for RNN is supposed to be [n_instances, n_steps, n_features]
## Since the n_instances=1 (As we are taking 400 records of Store 1) and n_features=1 (as we are considering 
## only 'Sales'), the shape would be (1,400,1).
#X_train_np_Begin=train_store_Begin[0:400].reshape(1,-1,4).copy()
#
## Output would be same as input. But shifted forward by one day.
#y_train_np_Begin=train_store_Begin[1:401,0].reshape(1,-1,1).copy()
#
#X_test_np_Begin=train_store_Begin[0:500].reshape(1,-1,4).copy()
#X_test_np_Begin[0, 401:, 0]=0

train_store_Begin.shape
X_train_np_Begin=train_store_Begin[:,0:190,:].copy()
y_train_np_Begin=train_store_Begin[:,1:191,4:5].copy()
print('Shape of Training Vector', X_train_np_Begin.shape)
print('Shape of Output Vector', y_train_np_Begin.shape)
X_train_np_Begin
y_train_np_Begin
# Data that would be used for testing purposes
X_test_np_Begin=train_store_Begin[:,:,:].copy()
X_test_np_Begin[:,190:210,4:5]=0
X_test_np_Begin
##############################################################
# Implementation of Tensor Flow Model
# RNN Model
#############################################################
n_steps_Begin=190   # Records for last 400 days
n_features_Begin=5 # Number of features to be used. To begin with, using only 'Sales' as Input Feature
n_neurons_Begin=100 # Number of neurons on each Cell
n_outputs_Begin=1 # 1 outout, since only Sales has to be predicted.
learning_rate_Begin=0.001

n_iterations_Begin=10000

tf.reset_default_graph()
X_Begin=tf.placeholder(tf.float32, [None, n_steps_Begin, n_features_Begin], name="InputData")
y_Begin=tf.placeholder(tf.float32, [None, n_steps_Begin, n_outputs_Begin], name="OutputData")

cell_Begin=tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons_Begin, activation=tf.nn.elu),
    output_size=n_outputs_Begin)
outputs_Begin,states_Begin=tf.nn.dynamic_rnn(cell_Begin, X_Begin, dtype=tf.float32)

loss_Begin=tf.reduce_mean(tf.square(outputs_Begin-y_Begin))
optimizer_Begin=tf.train.AdamOptimizer(learning_rate=learning_rate_Begin)

training_op_Begin=optimizer_Begin.minimize(loss_Begin)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(10000):
        sess.run(training_op_Begin, feed_dict={X_Begin:X_train_np_Begin, y_Begin:y_train_np_Begin})
        if epoch%1000==0:
            mse_Begin=loss_Begin.eval(feed_dict={X_Begin:X_train_np_Begin, y_Begin:y_train_np_Begin})
            print(epoch, "\tMSE:", mse_Begin)
            
    sequence_Begin = X_test_np_Begin.copy()
    i=0
    for iteration in range(20):
        X_test_Begin = sequence_Begin[:,0+i:n_steps_Begin+i,:].copy()
        y_pred_Begin = sess.run(outputs_Begin, feed_dict={X_Begin: X_test_Begin})
        sequence_Begin[:,n_steps_Begin+i,4:5]=y_pred_Begin[:, -1, :]
        i+=1
    print("Execution Done")
sequence_Begin[0,190:200, 4:5]
train_store_Begin[0,190:200,4:5]
Pred_Begin=plt.plot(sequence_Begin[0,180:210, 4:5], label="Prediction")
Actual_Begin=plt.plot(train_store_Begin[0,180:210,4:5], label="Actual")
plt.legend()
plt.show()
