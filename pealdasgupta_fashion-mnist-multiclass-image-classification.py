# Objectives

# Image data multiclass classification and object identification - apparel and clothing

# Using H2O Deep Learning 

# 2 Parts - Part I without reducing dimensionality; Part II reducing dimensionality using random projection

# Dataset : Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. 

# Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
# 1.0 Call libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

# 1.1 For measuring time elapsed

from time import time

from imblearn.over_sampling import SMOTE, ADASYN



# 1.2 Processing data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler 

from sklearn.compose import ColumnTransformer as ct

from sklearn.random_projection import SparseRandomProjection as sr  # Projection features



# 1.3 Data imputation

from sklearn.impute import SimpleImputer



# 1.4 Model building

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator



# 1.5 for ROC graphs & metrics

# conda install -c conda-forge scikit-plot 

import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics





# 1.6 Change ipython options to display all data columns

pd.options.display.max_columns = 300

# 2.0 Read data from Local directory

#os.chdir("C:\\Backup\\My Documents\\My Documents\\Training\\Data ScienceAnalytics\\Python")

fash_train = pd.read_csv("../input/fashion-mnist_train.csv")

fash_test  = pd.read_csv("../input/fashion-mnist_test.csv")
# 3.0 Plotting the image - not necessary for the modelling exercise

#  Get any row excluding first column; First column contains class labels and other columns contain pixel-intensity values

abc = fash_train.values[4, 1:]

abc.shape    # (784,)

abc = abc.reshape(28,28)   # Reshape to 28 X 28



# plot 

plt.imshow(abc)

plt.show()

# 4.0 Plotting the image - not necessary for the modelling exercise

for i in range(5000,5005): 

    sample = np.reshape(fash_train[fash_train.columns[1:]].iloc[i].values, (28,28))

    plt.figure()

    #plt.title("labeled class {}".format(get_label_cls(fash_train["label"]/255.iloc[i])))

    plt.imshow(sample)
# 5.0 Data exploration

fash_train.shape     # 60000 X 785

fash_test.shape      # 10000 X 785

#fash_train.dtypes

fash_train.label.value_counts()

fash_test.label.value_counts()
# 6.0 Combine test & train  

tmp = pd.concat([fash_train,fash_test],

               axis = 0,            # Stack one upon another (rbind)

               ignore_index = True

              )

tmp.shape



# 7.0 Separation into target/predictors

y = tmp.iloc[:,0]

X = tmp.iloc[:,1:]

X.shape              # 70000 X 784

y.shape              # 70000 X 1

y.head()

X.head()
# 8.0 Transform to numpy array



fash_tmp = X.values

fash_tmp.shape       # (70000 X 784)

target_tmp = y.values
# 9.1 Create a StandardScaler instance

ss = StandardScaler()

# 9.2 fit() and transform() in one step

fash_tmp = ss.fit_transform(fash_tmp)

# 9.3

fash_tmp.shape               # 70000 X 784 (an ndarray)
# 10.1 Separate train and test

X = fash_tmp[: fash_train.shape[0], : ]

X.shape                             # 60000 X 784



# 10.2

test = fash_tmp[fash_train.shape[0] :, : ]

test.shape                         # 10000X 784



target_train = target_tmp[: fash_train.shape[0]]

target_test  = target_tmp[fash_train.shape[0]: ]

target_train.shape

target_test.shape
################## Model building #####################

# 11.0 Split train into training and validation dataset

X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    target_train,

                                                    test_size = 0.3)



# 11.1

X_train.shape    # 42000 X 784 

X_test.shape     # 18000 X 784

X_test.shape
X_train.shape
# 11.0 composite data with both predictors and target

y_train = y_train.reshape(len(y_train),1)

y_train

X = np.hstack((X_train,y_train))

X.shape            # 42000 X 785

# 12.0 Modelling using H2O Deep Learning

# 12.1 Start h2o

h2o.init()



# 12.2 Transform data to h2o dataframe

df = h2o.H2OFrame(X)

len(df.columns)    # 20

df.shape           # 42000 X 785

df.columns



X_columns = df.columns[0:784]        # Only column names. No data

X_columns       # C1 to C784

y_columns = df.columns[784]

y_columns



df['C785'].head()      

# 14.1 As required by H2O , For classification, target column must be factor

df['C785'] = df['C785'].asfactor()
# 13. Build a deeplearning model on balanced data

#     http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html

dl_model = H2ODeepLearningEstimator(epochs=500,

                                    distribution = 'multinomial',                 # Response has two levels

                                    missing_values_handling = "skip", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 3,                           # CV folds

                                    fold_assignment = "auto",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [100,100],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')
# 14.1 Train model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df)





end = time()

(end - start)/60

y_test.shape



#X_test.shape
print (dl_model)
# 15. Predictions on actual  'test' data

#     Create a composite X_test data before transformation to

#     H2o dataframe.

y_test = y_test.reshape(len(y_test), 1)     # Needed to hstack

y_test.shape     # 18000 X 1

X_test.shape     # 18000 X 784
# 16.1 Column-wise stack 

X_test = np.hstack((X_test,y_test))         # cbind data

X_test.shape     # 18000 X 785
X_test = h2o.H2OFrame(X_test)

X_test['C785'] = X_test['C785'].asfactor()
# 17. Make prediction on X_test

result = dl_model.predict(X_test[: , 0:784])

result.shape       # 18000 X 11

result.as_data_frame().head()   # Class-wise predictions
result.shape

fash_pred = X_test['C785'].as_data_frame()

fash_pred['result'] = result[0].as_data_frame()

fash_pred.head()

fash_pred.columns
fash_pred.columns   # 2 columns 'C785', 'result'

fash_pred.head(5)
# 18. So compare ground truth with predicted --accuracy of the model

out = (fash_pred['result'] == fash_pred['C785'])

np.sum(out)/out.size
# 19.1  create confusion matrix using pandas dataframe

g  = confusion_matrix(fash_pred['C785'], fash_pred['result'] )

g

# the diagonal elements represent the correct elements.
fash_tmp.shape  # Numpy ndarray  70000 X 784

type(fash_tmp)
############################################################################

######  Part 2   Reducing dimensionality Using Random Projection

################ Feature creation Using Random Projections ##################

# 20. Using Random projection to reduce dimensionality of data

#     create 100 random projections/columns



NUM_OF_COM = 100



# 20.1 Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)



# 20.2 fit and transform the (original) dataset

#      Random Projections with desired number

#      of components are returned

rp = rp_instance.fit_transform(fash_tmp)



# 20.3 Look at some features

rp[: 5, :  3]





# 20.4 Create some column names for these columns

#      We will use them at the end of this code

rp_col_names = ["r" + str(i) for i in range(100)]





rp.shape   ### (70000 X 100)
# 21.1 Separate train and test

X = rp[: fash_train.shape[0], : ]

X.shape                             # 60000 X 100



# 21.2

test = rp[fash_train.shape[0] :, : ]

test.shape                         # 10000X 100



target_train = target_tmp[: fash_train.shape[0]]

target_test  = target_tmp[fash_train.shape[0]: ]

target_train.shape

target_test.shape
X.shape

test.shape
################## Model building #####################

# 22.0 Split train into training and validation dataset

X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    target_train,

                                                    test_size = 0.3)



# 11.1

X_train.shape    # 42000 X 100

X_test.shape     # 18000 X 100
X_train.shape
# 23.0 composite data with both predictors and target

y_train = y_train.reshape(len(y_train),1)

y_train

X = np.hstack((X_train,y_train))

X.shape            # 42000 X 101
# 24.0 Modelling using H2O Deep Learning

# 24.1 Start h2o

h2o.init()



# 24.2 Transform data to h2o dataframe

df = h2o.H2OFrame(X)

len(df.columns)    # 20

df.shape           # 42000 X 101

#df.columns

df.columns

#len(df.columns) 
X_columns = df.columns[0:100]        # Only column names. No data

X_columns       # C1 to C100

y_columns = df.columns[100]

y_columns



df['C101'].head()      

# 14.1 As required by H2O , For classification, target column must be factor

df['C101'] = df['C101'].asfactor()
# 25. Build a deeplearning model on data

#     http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html

dl_model = H2ODeepLearningEstimator(epochs=500,

                                    distribution = 'multinomial',                 # Response has two levels

                                    missing_values_handling = "skip", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 2,                           # CV folds

                                    fold_assignment = "auto",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [100,100],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')
# 26 Train model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df)





end = time()

(end - start)/60

print (dl_model)
type(X_test)

# 27. Predictions on actual unbalanced 'test' data

#     Create a composite X_test data before transformation to

#     H2o dataframe.

y_test = y_test.reshape(len(y_test), 1)     # Needed to hstack

y_test.shape     # 18000 X 1

X_test.shape     # 18000 X 100
# 28.1 Column-wise stack now

X_test = np.hstack((X_test,y_test))         # cbind data

X_test.shape     # 18000 X 101
X_test = h2o.H2OFrame(X_test)

X_test['C101'] = X_test['C101'].asfactor()
# 29. Make prediction on X_test

result = dl_model.predict(X_test[: , 0:100])

result.shape       # 18000 X 11

result.as_data_frame().head()   # Class-wise predictions
result.shape

fash_pred = X_test['C101'].as_data_frame()

fash_pred['result'] = result[0].as_data_frame()

fash_pred.head()

fash_pred.columns
fash_pred.columns   # 2 columns 'C101', 'result'

fash_pred.head(5)
# 30. Accuracy of the model reducing dimenionsality

out = (fash_pred['result'] == fash_pred['C101'])

np.sum(out)/out.size
## Accuracy without reducing dimensionality is 87%

## Accuracy after reducing dimensionality features random projection and 100 features is 86%