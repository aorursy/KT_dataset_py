# Import usual libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew



# Import Neural Network libraries

import tensorflow as tf



from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping



# Import data processing libraries

import sklearn

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline



# Not the best practice, but ok for now. (surpresses all warnings)

import warnings

warnings.filterwarnings("ignore")



# Read in the data with pandas .read_csv

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_train.head() # get a general view of the data. The test data has the same columns, only SalePrice is missing.



# Concatenate the test and training data (apart from the SalePrice column) to treat the input columns equally during the preprocessing step.



complete_input = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],

                      df_test.loc[:,'MSSubClass':'SaleCondition']))
# Output data skewness



df_train.SalePrice.hist() # raw output (target) data is skewed

plt.figure()

df_train.SalePrice.apply(np.log1p).hist() # log_transformed data is more normal.



df_train.SalePrice = df_train.SalePrice.apply(np.log1p) # transform the output column with a log+1 transformation: x -> log(1+x). NOTE: in the end, to get back to real dollar results, we need to invers transofrm.
# Input data skewness. 

# Naturally, skewness is only defined for numerical variables, so we first need to determine which columns are represented by numberical values.



numeric_feats = complete_input.dtypes[complete_input.dtypes != 'object'].index # .dtypes gives a df with cols as index and dtypes as varaible.

                                                                               # .index then gives the col names

    

# calculate skewness of each num feat, discarting NaN's since they don't contribute to skewness and will distort the result.

skewed_feats = complete_input[numeric_feats].apply(lambda x: skew(x.dropna())) # select only the numerical columns

skewed_feats = skewed_feats[skewed_feats > 0.1] # Which level of skewness do you want to correct for? The lower, the more data will be transformed

skewed_feats = skewed_feats.index



print(skewed_feats) # names of each column where the skewness is larger than 0.1, with NaN's discarted.



# Now transform all the columns that have skewed, numerical feats.

complete_input[skewed_feats] = np.log1p(complete_input[skewed_feats])
# First get an overview of all the missing data

# NOTE that here we assume that the test dataset is a good representation of the training dataset concerning any parameter.



total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'], sort=False)

print(missing_data.head(20))



# how to deal with the missing values can be to drop, set to zero, or to mean, depending on what the variables denotes.



complete_input = complete_input.fillna(0) # for now we set all the missing data to zero.



# pandas has a single function to do this. Sklearn also has a built in function in the preprocessing package.



complete_input = pd.get_dummies(complete_input)



print(complete_input.info())
# preparing data for sklearn models

print(df_train.shape)

print(df_train.shape[0]) # split the complete_input dataframe at this point since this is the original number of entries of the traning data set



X_train = complete_input[:df_train.shape[0]]

X_test = complete_input[df_train.shape[0]:]

Y_train = df_train.SalePrice



#X_train = preprocessing.StandardScaler(X_train)

#X_test = preprocessing.StandardScaler(X_test)
num_cols = len(X_train.columns) # get the number of columns as the number of input nodes in our network



model = Sequential() # initiating the model

model.add(Dense(15, input_shape=(num_cols,), activation = 'relu')) # input layer

model.add(Dense(15, activation='relu'))  # hidden layer 1

model.add(Dense(15, activation='relu'))  # hidden layer 2

model.add(Dense(15, activation='relu'))  # hidden layer 3

model.add(Dense(1,))                    # output layer



#Compiles model

model.compile(Adam(lr=0.003), 'mean_squared_error') # optimizing method and error function, LR should be large for large outputs



#Fits model

history = model.fit(X_train, Y_train, epochs = 1000, validation_split = 0.2,verbose = 0)

history_dict=history.history



#Plots model's training cost/loss and model's validation split cost/loss

loss_values = history_dict['loss']

val_loss_values=history_dict['val_loss']

plt.figure()

plt.plot(loss_values,'bo',label='training loss')

plt.plot(val_loss_values,'r',label='val training loss')

plt.legend()



# Test how model holds up for our training data (not that good of an indicator but it gives us an approximate sense)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



print("The MSE score on the Train set is:\t{:0.3f}".format(np.sqrt(mean_squared_log_error(Y_train,y_train_pred))))



# Make dataframe of performance model on the training data

y_train_pred_df = pd.DataFrame(y_train_pred, columns=['SalePrice']) # MAKE DIMENSION OUTPUT OK

Compare_df = pd.DataFrame({'TrueValue': np.expm1(Y_train), 'PredValue': np.expm1(y_train_pred_df.SalePrice)})

print(Compare_df)
y_test_pred_df = pd.DataFrame(y_test_pred, columns=['SalePrice']) # MAKE DIMENSION OUTPUT OK



# making a submission file

my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': np.expm1(y_test_pred_df.SalePrice)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_NeuralNets.csv', index=False)





print(my_submission)