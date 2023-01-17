1+2
import numpy

a = numpy.array([[1,2,3],[4,5,6],[7,8,9]])

print (a)
print (a.T) 
# Numpy is a library for the Python, that adds support for large, 

# multi-dimensional arrays and matrices, along with a large collection of 

# high-level mathematical functions to operate on these arrays.

# It's extremely handy for "vectorizing" operations.

import numpy as np
# Pandas is a data manipulation and analysis library. 

# In particular, it offers data structures and operations for manipulating numerical 

# tables and time series.

import pandas as pd



# scikit-learn is a Machine Learning library. It features various classification, 

# regression and clustering algorithms and is designed to interoperate with Python libraries like NumPy.

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder



# XGBoost is one of many machine learning algorithms

from xgboost import XGBClassifier
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Let's load the CSV into a variable called 'dataset'

dataset = pd.read_csv('../input/voice.csv', header=0).values



# dataset is a matrix

print (dataset.shape)
print (dataset[0])
my_matrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16], [17,18,19,20]])

print (my_matrix)
# Get the element from the first row, second column (zero-indexed)

print (my_matrix[0, 1])
# You can specify ranges using colon (:)



# Print rows from 1 up to 3, columns 0 up to 3

print (my_matrix[1:3, 0:3])
# You can omit either "start" or "end"



# print all rows up to 3rd row, every column starting from the fourth one

print (my_matrix[:3, 2:])
# print all rows, second column

print (my_matrix[:, 2])
# QUIZ!



# print all rows, all columns with the exception of last one

# print (my_matrix[start:end, start:end])



# print all rows, only last column

# print (my_matrix[start:end, start:end])
# Load all features into 'x' and labels into 'y'

x = dataset[:, :-1]

y = dataset[:, -1]



print (x.shape)

print (y.shape)
# Let's print the first 3 rows of x

print (x[:3])
# Let's print the first 3 rows of y

print (y[:3])
label_encoder = LabelEncoder()

label_encoded_y = label_encoder.fit_transform(y)



# print first 3 and last 3 instances of y

print (y[:3])

print (y[-3:])



# print first 3 and last 3 instances of label_encoded_y

print (label_encoded_y[:3])

print (label_encoded_y[-3:])
# Now we are ready to pass it to a Machine Learning Algorithm

model = XGBClassifier()

model.fit(x, label_encoded_y)



# Now 'model' is ready to do predictions!

print ('Mary')

x_test0 = np.array([

    0.217427436852732, 0.0452543270933, 0.234337899543379, 0.206392694063927, 0.243470319634703, 0.0370776255707762,

    4.46856483331244, 28.9546502761532, 0.868510554863495, 0.272468042638548, 0.235981735159817, 0.217427436852732,

    0.177765492697815, 0.0452772073921971, 0.279113924050633, 1.4383721534242, 0, 12.080126953125, 12.080126953125,

    0.0789463260051495

])[:, np.newaxis].T

y_pred0 = model.predict(x_test0)

print ('Male' if y_pred0[0] == 1 else 'Female')
# Recap



# XGBoost is pretty popular because it typically doesn't require tuning of hyper-parameters in

# order to make robust predictions.



import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder



dataset = pd.read_csv('../input/voice.csv', header=0).values

x = dataset[:, :-1]

y = dataset[:, -1]



label_encoder = LabelEncoder()

label_encoded_y = label_encoder.fit_transform(y)



model = XGBClassifier()

model.fit(x, label_encoded_y)



print ('Mary')

x_test0 = np.array([

    0.217427436852732, 0.0452543270933, 0.234337899543379, 0.206392694063927, 0.243470319634703, 0.0370776255707762,

    4.46856483331244, 28.9546502761532, 0.868510554863495, 0.272468042638548, 0.235981735159817, 0.217427436852732,

    0.177765492697815, 0.0452772073921971, 0.279113924050633, 1.4383721534242, 0, 12.080126953125, 12.080126953125,

    0.0789463260051495

])[:, np.newaxis].T

y_pred0 = model.predict(x_test0)

print ('Male' if y_pred0[0] == 1 else 'Female')

y_predicted = model.predict(x)

accuracy = accuracy_score(label_encoded_y, y_predicted)

print("Accuracy XGBoost: %.2f%%" % (accuracy * 100.0))

test_size = 0.33

x_training, x_test, y_training, y_test = train_test_split(x,

                                                          label_encoded_y,

                                                          test_size=test_size,

                                                          random_state=7)



print ("x.shape: {}".format(x.shape))

print ("x_training.shape: {}".format(x_training.shape))

print ("x_test.shape: {}".format(x_test.shape))
# train_test_split takes care of shuffling data



print (label_encoded_y.shape)

print (y_training.shape)

print (y_test.shape)



print (y_training.sum())

print (y_test.sum())
model = XGBClassifier()

model.fit(x_training, y_training)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy XGBoost: %.2f%%" % (accuracy * 100.0))
