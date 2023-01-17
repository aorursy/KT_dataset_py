#%matplotlib inline

import numpy as np

import pandas as pd

import itertools

from sklearn import preprocessing

import matplotlib.pyplot as plt

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import metric_spec

from tensorflow.contrib.learn.python.learn.estimators import _sklearn

from tensorflow.contrib.learn.python.learn.estimators import estimator

from tensorflow.contrib.learn.python.learn.estimators import model_fn

from tensorflow.python.framework import ops

from tensorflow.python.saved_model import loader

from tensorflow.python.saved_model import tag_constants

from tensorflow.python.util import compat

tf.logging.set_verbosity(tf.logging.FATAL) 

df = pd.read_csv("../input/loan.csv", low_memory=False)
df['Default_Binary'] = int(0)

for index, value in df.loan_status.iteritems():

    if value == 'Default':

        df.set_value(index,'Default_Binary',int(1))

    if value == 'Charged Off':

        df.set_value(index, 'Default_Binary',int(1))

    if value == 'Late (31-120 days)':

        df.set_value(index, 'Default_Binary',int(1))    

    if value == 'Late (16-30 days)':

        df.set_value(index, 'Default_Binary',int(1))

    if value == 'Does not meet the credit policy. Status:Charged Off':

        df.set_value(index, 'Default_Binary',int(1))    
df['Purpose_Cat'] = int(0) 

for index, value in df.purpose.iteritems():

    if value == 'debt_consolidation':

        df.set_value(index,'Purpose_Cat',int(1))

    if value == 'credit_card':

        df.set_value(index, 'Purpose_Cat',int(2))

    if value == 'home_improvement':

        df.set_value(index, 'Purpose_Cat',int(3))    

    if value == 'other':

        df.set_value(index, 'Purpose_Cat',int(4))    

    if value == 'major_purchase':

        df.set_value(index,'Purpose_Cat',int(5))

    if value == 'small_business':

        df.set_value(index, 'Purpose_Cat',int(6))

    if value == 'car':

        df.set_value(index, 'Purpose_Cat',int(7))    

    if value == 'medical':

        df.set_value(index, 'Purpose_Cat',int(8))   

    if value == 'moving':

        df.set_value(index, 'Purpose_Cat',int(9))    

    if value == 'vacation':

        df.set_value(index,'Purpose_Cat',int(10))

    if value == 'house':

        df.set_value(index, 'Purpose_Cat',int(11))

    if value == 'wedding':

        df.set_value(index, 'Purpose_Cat',int(12))    

    if value == 'renewable_energy':

        df.set_value(index, 'Purpose_Cat',int(13))     

    if value == 'educational':

        df.set_value(index, 'Purpose_Cat',int(14))  
x = np.array(df.int_rate.values).reshape(-1,1) 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df['int_rate_scaled'] = pd.DataFrame(x_scaled)

print (df.int_rate_scaled[0:5])
x = np.array(df.funded_amnt.values).reshape(-1,1) 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df['funded_amnt_scaled'] = pd.DataFrame(x_scaled)

print (df.funded_amnt_scaled[0:5])
training_set = df[0:500000] # Train on first 500k rows

testing_set = df[500001:849999] # Test on next 350k rows

prediction_set = df[850000:] # Predict on final 37k rows



COLUMNS = ['Purpose_Cat','funded_amnt_scaled','int_rate_scaled','Default_Binary']          

FEATURES = ['Purpose_Cat','funded_amnt_scaled','int_rate_scaled']

LABEL = 'Default_Binary'



def input_fn(data_set):

    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES} 

    labels = tf.constant(data_set[LABEL].values)

    return feature_cols, labels
feature_cols = [tf.contrib.layers.real_valued_column(k)

              for k in FEATURES]

#config = tf.contrib.learn.RunConfig(keep_checkpoint_max=1) ######## DO NOT DELETE

regressor = tf.contrib.learn.DNNRegressor(

  feature_columns=feature_cols, hidden_units=[10, 20, 10], ) 

regressor.fit(input_fn=lambda: input_fn(training_set), steps=251)
# Score accuracy

ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=10)

loss_score = ev["loss"]

print("Loss: {0:f}".format(loss_score))
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))

predictions = list(itertools.islice(y, 37379))
plt.plot(prediction_set.int_rate_scaled, predictions, 'ro')

plt.ylabel("Model Prediction Value")

plt.xlabel("Interest Rate of Loan (Scaled between 0-1)")

plt.show()
plt.plot(prediction_set.funded_amnt_scaled, predictions, 'ro')

plt.ylabel("Model Prediction Value")

plt.xlabel("Funded Amount of Loan (Scaled between 0-1)")

plt.show()
plt.plot(prediction_set.Purpose_Cat, predictions, 'ro')

plt.ylabel("Default Prediction Value")

plt.xlabel("Loan Purpose")

plt.title("DNN Regressor Predicting Default By Loan Purpose")

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 8

fig_size[1] = 8

plt.rcParams["figure.figsize"] = fig_size

labels = ['Debt Consolidation', 'Credit Card', 'Home Improvement', 'Other',

         'Major Purchase', 'Small Business', 'Car', 'Medical',

         'Moving', 'Vacation', 'House', 'Wedding',

         'Renewable Energy']



plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14], labels, rotation='vertical')



plt.show()