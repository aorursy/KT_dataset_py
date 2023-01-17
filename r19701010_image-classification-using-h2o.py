#Objective:  Read and classify the image using H20 Deep learning



# 1. Classify fashion_mnist dataset using H2O deeplearning after reducing dimensionality of its data and after 

#    standardizing the integer values. And check how good your classification is.

# 2. Get feature importance of PCA components
# 1.0 Call libraries

%reset -f

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



# 1.0.1 For measuring time elapsed

from time import time
# 1.2 Processing data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct
# 1.3 

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator
# 1.4 Set working directly



print(os.listdir("../input"))

# read the file



train_data = pd.read_csv("../input/fashion-mnist_train.csv")

train_data.shape
X_columns = train_data.columns.values

X_columns
train_data.dtypes.value_counts()
train_data['label'].unique()
# To see the image of one particular observation after removing the label which is in the first column

img1 = train_data.values[1, 1:]
type(img1)
img1 = img1.reshape(28,28)
img1.shape
plt.imshow(img1)

plt.show()
# Separate target (y) and predictors (X)

y = train_data.iloc[:,0]

X = train_data.iloc[:,1:]
type(y)
X.shape

y.shape

# Scale X using StandardScaler() class of sklearn.



X= X.values

type(X)
# Create a deeplearning model using (X_train,y_train)



h2o.init()

y_columns = X_columns[0]

y_columns
X_columns = np.delete(X_columns, 0)
df_train = h2o.import_file("../input/fashion-mnist_train.csv")

df_test  =  h2o.import_file("../input/fashion-mnist_test.csv")
dl_model = H2ODeepLearningEstimator(epochs=500,

                                    distribution = 'multinomial',                 # Response has two levels

                                    nfolds = 2,                           # CV folds

                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis                                    

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [32,32],                  # ## more hidden layers 

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')
X_columns = df_train.columns[1:785]
y_columns = df_train.columns[0]
# For H20 classification, target column must be factor



df_train[0] = df_train[0].asfactor()

# Train the model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df_train)





end = time()

(end - start)/60
result = dl_model.predict(df_test[: , 1:785])
result.shape
# Convert H2O frame back to pandas dataframe

Actual = df_test[0].as_data_frame()
Actual.shape
Actual['result'] = result[0].as_data_frame()
Actual.shape
Actual.head()
# compare the actual vs predict

out = (Actual['result'] == Actual['label'])
np.sum(out)
np.sum(out)/out.size