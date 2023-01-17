#!pip install urllib2

url='https://drive.google.com/uc?export=download&id=10I5aboiafcqf0iVWnd1SiAEVbgWzgkW4'

import requests

r = requests.get(url, allow_redirects=True)

open('data.zip', 'wb').write(r.content)



#extracting data.zip

from zipfile import ZipFile 

  

# specifying the zip file name 

file_name = "data.zip"

  

# opening the zip file in READ mode 

with ZipFile(file_name, 'r') as zip: 

    # printing all the contents of the zip file 

    zip.printdir() 

  

    # extracting all the files 

    print('Extracting all the files now...') 

    zip.extractall() 

    print('Done!')
import pandas as pd
#You need to import the necessary libraries to train the model.

import numpy as np

import pandas as pd

from sklearn import datasets

import itertools

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 
#Step 1) Import the data with panda.



#You define the column names and store it in COLUMNS. 

#You can use pd.read_csv() to import the data.



COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",

           "dis", "tax", "ptratio", "medv"]

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)



test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)



prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)



#You can print the shape of the data.
print(training_set.shape, test_set.shape, prediction_set.shape)			
FEATURES = ["crim", "zn", "indus", "nox", "rm",

                 "age", "dis", "tax", "ptratio"]

LABEL = "medv"
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

for i in feature_cols:

    print(i)
estimator = tf.estimator.LinearRegressor(    

        feature_columns=feature_cols,   

        model_dir="train")
def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):    

        

        return tf.estimator.inputs.pandas_input_fn(       

         x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),       

         y = pd.Series(data_set[LABEL].values),       

         batch_size=n_batch,          

         num_epochs=num_epochs, 

         shuffle=shuffle)
estimator.train(input_fn=get_input_fn(training_set,                                       

                                           num_epochs=None,                                      

                                           n_batch = 128,                                      

                                           shuffle=False),                                      

                                           steps=1000)
%load_ext tensorboard

%tensorboard --logdir=/train

# if tensorboard is not showing then download and do it manually
ev = estimator.evaluate(    

          input_fn=get_input_fn(test_set,                          

          num_epochs=1,                          

          n_batch = 128,                          

          shuffle=False))
#You can print the loss with the code below:



loss_score = ev["loss"]

print("Loss: {0:f}".format(loss_score))	

#see some other results

print(ev)
#You can check the summary statistic to get an idea of how big the error is.

training_set['medv'].describe()
#see some other results

print(ev)
y = estimator.predict(    

         input_fn=get_input_fn(prediction_set,                          

         num_epochs=1,                          

         n_batch = 128,                          

         shuffle=False))	
#To print the estimated values of , you can use this code:



predictions = list(p["predictions"][0] for p in itertools.islice(y, 6))



print('House    Prediction')

for i in range(len(predictions)):

    print(i,"       ",predictions[i])
def prepare_data(df):     

        X_train = df[:, :-3]    

        y_train = df[:,-3]    

        return X_train, y_train

training_set_n = pd.read_csv("boston_train.csv").values

test_set_n = pd.read_csv("boston_test.csv").values

prediction_set_n = pd.read_csv("boston_predict.csv").values
def prepare_data(df):     

        X_train = df[:, :-3]    

        y_train = df[:,-3]    

        return X_train, y_train
#You can use the function to split the label from the features of the train/evaluate dataset

X_train, y_train = prepare_data(training_set_n)

X_test, y_test = prepare_data(test_set_n)

X_train
prediction_set_n
#You need to exclude the last column of the prediction dataset because it contains only NaN



x_predict = prediction_set_n[:, :-2]

x_predict
#Confirm the shape of the array. Note that, the label should not have a dimension, it means (400,).



print(X_train.shape, y_train.shape, x_predict.shape)
#You can construct the feature columns as follow:



feature_columns = [ tf.feature_column.numeric_column('x', shape=X_train.shape[1:])]

feature_columns
#The estimator is defined as before, you instruct the feature columns and 

#where to save the graph.



estimator = tf.estimator.LinearRegressor(    

         feature_columns=feature_columns,    

         model_dir="train1")	
X_train
#You can use the numpy estimapor to feed the data to the model and 

#then train the model. Note that, we define the input_fn function before to ease the readability.



# Train the estimator

train_input = tf.estimator.inputs.numpy_input_fn(   

           x={"x": X_train},    

           y=y_train,    

           batch_size=128,    

           shuffle=False,    

           num_epochs=None)

estimator.train(input_fn = train_input,steps=5000)
#now create input and evaluate

eval_input = tf.estimator.inputs.numpy_input_fn(    

       x={"x": X_test},    

       y=y_test, 

       shuffle=False,    

       batch_size=128,    

       num_epochs=1)

estimator.evaluate(eval_input,steps=None)
#Finaly, you can compute the prediction. It should be the similar as pandas.



test_input = tf.estimator.inputs.numpy_input_fn(    

        x={"x": x_predict},    

        batch_size=128,    

        num_epochs=1,   

        shuffle=False)

y = estimator.predict(test_input)

predictions = list(p["predictions"][0] for p in itertools.islice(y, 6))

print(" House      ","Prediction" )

for i in range(len(predictions)):

    print("  ",i,'      ',predictions[i])