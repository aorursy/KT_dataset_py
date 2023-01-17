import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import h2o

import os

# 1.1

from h2o.estimators.deeplearning import H2ODeepLearningEstimator

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct

from sklearn.decomposition import PCA
# Change working folder and read fashionmnist data

os.chdir("../input")

train = pd.read_csv("fashion-mnist_train.csv")

test = pd.read_csv("fashion-mnist_test.csv")
# Plotting the image - not necessary for the modelling exercise

# Get any row excluding first column; First column contains class labels and other columns contain pixel-intensity values

zzz = train.values[4, 1:]

zzz.shape    # (784,)

zzz = zzz.reshape(28,28)   # Reshape to 28 X 28



# plot 

plt.imshow(zzz)

plt.show()
# Plotting the image - not necessary for the modelling exercise

for i in range(5000,5005): 

    sample = np.reshape(train[train.columns[1:]].iloc[i].values, (28,28))

    plt.figure()

    #plt.title("labeled class {}".format(get_label_cls(fash_train["label"]/255.iloc[i])))

    plt.imshow(sample)
train.head(2)
train.shape
test.shape
train.dtypes.value_counts() 
test.dtypes.value_counts()
train.info
train.describe()
# Separate the target (y) and the predictors (X)

y = train.iloc[:,0]

X = train.iloc[:,1:]

X.shape
# Transform X to numpy array

# Henceforth we will work with array only

X = X.values

X.shape
# Before PCA, data must be standardized

scale = ss()

X = scale.fit_transform(X)
# Perform pca

# Create PCA object

pca = PCA(0.95)

out = pca.fit_transform(X)

out.shape
# How much variance has been explained by each column

pca.explained_variance_ratio_ 
# Get cumulative sum

# 95% of variance explained by first 256 columns

pca.explained_variance_ratio_.cumsum()  
# This was the original dataset

x_result = np.corrcoef(X, rowvar = False)

np.around(x_result, 256) 
# Find correlation between columns

# round the result to two-decimal places

result = np.corrcoef(out, rowvar = False)

np.around(result, 256)
final_data = out[:, :256]
################## Model building #####################

# Split train into training and validation dataset

X_train, X_test, y_train, y_test = train_test_split(

                                                    final_data,

                                                    y,

                                                    test_size = 0.3)



# 

X_train.shape    # 42000 X 784 

X_test.shape     # 18000 X 784          
X_train.shape
# Composite data with both predictors and target

y_train = y_train.values.reshape(len(y_train),1)

y_train

X = np.column_stack((X_train,y_train))

X.shape            # 42000 X 257
# Start h2o

h2o.init()
# Transform data to h2o dataframe

df = h2o.H2OFrame(X)

len(df.columns)    

df.shape           

df.columns
X_columns = df.columns[0:256]        # Only column names. No data

X_columns       # C1 to C257

y_columns = df.columns[256]

y_columns



df['C257'].head()      

# 14.1 As required by H2O , For classification, target column must be factor

df['C257'] = df['C257'].asfactor()
# Build a deeplearning model on balanced data

#     http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html

dl_model = H2ODeepLearningEstimator(epochs=100,

                                    distribution = 'multinomial',                 # Response has two levels

                                    missing_values_handling = "skip", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 3,                           # CV folds

                                    fold_assignment = "auto",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [50,50],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')
from time import time
# Train model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df)





end = time()

(end - start)/60
print (dl_model)
# Predictions on actual  'test' data

# Create a composite X_test data before transformation to

# H2o dataframe.

y_test = (y_test.values).reshape(len(y_test), 1)     # Needed to hstack

y_test.shape     # 18000,1

X_test.shape     # 42000,256
# Column-wise stack now

X_test = np.hstack((X_test,y_test))         # cbind data

X_test.shape     # 42000,257
# Transform X_test to h2o dataframe

X_test = h2o.H2OFrame(X_test)

X_test['C257'] = X_test['C257'].asfactor()
# Make prediction on X_test

result = dl_model.predict(X_test[: , 0:256])

result.shape       # 18000 X 3

result.as_data_frame().head() 
# Ground truth

# Convert H2O frame back to pandas dataframe

xe = X_test['C257'].as_data_frame()

xe['result'] = result[0].as_data_frame()

xe.head()

xe.columns
# So compare ground truth with predicted

out = (xe['result'] == xe['C257'])

np.sum(out)/out.size
import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics
# Also create confusion matrix using pandas dataframe

f  = confusion_matrix( xe['C257'], xe['result'] )

f