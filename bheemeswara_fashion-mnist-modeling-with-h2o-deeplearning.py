### 

### Objective: Is to learn modeling with H2ODeeplearning  

###



# Call basic libraries

import pandas as pd        # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np         # linear algebra

import os



print(os.listdir("../input"))



# For plotting

import matplotlib.pyplot as plt



# For measuring time elapsed

from time import time



# Model building

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator



# Change ipython options to display all data columns

pd.options.display.max_columns = 300



# Any results you write to the current directory are saved as output.
os.chdir("../input")
#Read file

train = pd.read_csv("fashion-mnist_train.csv")
train.shape
train.head(3)
train['label'].value_counts() # balanced classes 
# Get the first row excluding first column

#    First column contains class labels and other columns contain pixel-intensity values

xyz = train.values[1, 1:]

xyz.shape    # (784,)
xyz = xyz.reshape(28,28)   # Reshape to 28 X 28
# Plot the reshapped image

plt.imshow(xyz)

plt.show()
#Get list of predictor column names and target column names

#     Column names are given by H2O when we converted array to H2o dataframe

X_columns = train.columns[1:786]        # Only column names and no data.
y_columns = train.columns[0]
y_columns
# Get the unique  in an array

train["label"].unique()
# Initialize H2o

h2o.init()
# Loading train & Test data

train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")
test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")
#Get list of predictor column names and target column names

#     Column names are given by H2O when we converted array to H2o dataframe

X_columns = train.columns[1:785]        # Only column names and no data C1 to C786
y_columns = train.columns[0]
y_columns
train["label"]=train["label"].asfactor()
# Get the levels in train

train['label'].levels()
# Model

model = H2ODeepLearningEstimator(

                             distribution="multinomial",

                             activation = "RectifierWithDropout",

                             hidden = [32,32,32],

                             input_dropout_ratio=0.2,  

                             standardize=True,

                             epochs = 500

                             )
# Build the Deep Learning Model

start = time()

model.train(X_columns,

               y_columns,

               training_frame = train)



end = time()

(end - start)/60
# Now predict the result

result = model.predict(test[: , 1:785])
result.shape
# Class-wise predictions

result.as_data_frame().head(10)   
#  Ground truth: Convert H2O frame back to pandas dataframe

xe = test['label'].as_data_frame()

xe['result'] = result[0].as_data_frame()

xe.head()
xe.columns
#Accuracy of the Prediction

out = (xe['result'] == xe['label'])

np.sum(out)/out.size