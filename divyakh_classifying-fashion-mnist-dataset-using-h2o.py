# Call libraries

# basic libraries

import pandas as pd

import numpy as np

import os

# For plotting

import matplotlib.pyplot as plt



# For measuring time elapsed

from time import time



# Model building

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator





# Change ipython options to display all data columns

pd.options.display.max_columns = 300
train = pd.read_csv("../input/fashion-mnist_train.csv")
train.shape

train.head(1)
train['label'].value_counts() 
abc = train.values[1, 1:]

abc.shape    # (784,)

abc = abc.reshape(28,28)
plt.imshow(abc)

plt.show()
X_columns = train.columns[1:786] 

y_columns = train.columns[0]

y_columns
h2o.init()
train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")

test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")
X_columns = train.columns[1:785] 
X_columns
y_columns = train.columns[0]
y_columns
train["label"]=train["label"].asfactor()
train["label"].unique()
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

result.as_data_frame().head(2)
xe = test['label'].as_data_frame()
xe.columns
xe['result'] = result[0].as_data_frame()
xe.columns
#Accuracy :

out = (xe['result'] == xe['label'])

np.sum(out)/out.size