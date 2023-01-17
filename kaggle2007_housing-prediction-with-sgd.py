# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../"))



# Any results you write to the current directory are saved as output.
filename = "../input/" + 'sample_submission.csv'

pan_submission = pd.read_csv(filename)
pan_submission.head() #Final submission should be in this form of a Sales Price
################

# Housing Price Competition

# Using SGD

# Hemal Vakharia

# 2-Feb-2020

#################

import pandas as pan # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import sklearn.preprocessing as pre

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

import xgboost as xgb

import keras as ks

import matplotlib.pyplot as plt

from keras.layers import Dense

from keras.callbacks import EarlyStopping

import h5py

from keras.callbacks import ModelCheckpoint

from keras.regularizers import l2

from keras.layers import Dropout

from keras import optimizers

from sklearn.preprocessing import LabelEncoder

from keras.optimizers import SGD
#Given: Train and Test Data set - separately



#Reading Train Data set



def readfile(filename):

    filename = "../input/" + filename

    return(pan.read_csv(filename))
def fill_and_encode_pandas_array(pandas_array, fill_value, col_type, encoding_type):



    ####################################################

    # Fill null or NaN in the array with specified value

    ####################################################



    if(fill_value == 0.0):

        pandas_array.fillna(fill_value, inplace=True)



    ####################################################

    # Imputation fills in the missing value

    # with some number. The imputed value won't be exactly

    # right in most cases, but it usually gives more

    # accurate models than dropping the column entirely.

    #####################################################

    elif(fill_value == 'impute'): # imputation

        from sklearn.impute import SimpleImputer

        my_imputer = SimpleImputer(strategy="most_frequent")

        pandas_array = my_imputer.fit_transform(pandas_array)

        pandas_array=pan.DataFrame(pandas_array) #impute returns ndarray and not pandas array



    ####################################################

    # If column type is object then we have to encode

    # It has to be first converted into category type and

    # then encode

    ####################################################

    if(encoding_type == 'label_encoding'):

        if(col_type == 'object'):

            for col in pandas_array.columns:

                if(pandas_array[col].dtypes == col_type):

                    pandas_array[col] = pandas_array[col].astype('category')

                    pandas_array[col] = pandas_array[col].cat.codes

    else:

        print("\n\n==== other encoding technique is not supported====\n\n")



    return(pandas_array)
def plots(plot_train, x_axis, y_axis, flag = True):

    if(flag):

        fig, ax = plt.subplots()

        ax.scatter(x = plot_train[x_axis], y = plot_train[y_axis])

        plt.ylabel(y_axis, fontsize=13)

        plt.xlabel(x_axis, fontsize=13)

        plt.show()
###########################################################

# Read the Training data set. Will read test data set later

###########################################################

data_set_housing_train = readfile("train.csv")
print(pan.DataFrame(data_set_housing_train.describe()))

    #in pynotebook, DataFrame prints values better (in the frame)



#Saving 'Id' column for future

train_ID = data_set_housing_train['Id']
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

data_set_housing_train.drop("Id", axis = 1, inplace = True)



train_X = data_set_housing_train
##########################################################

# In housing the overall area and the price are related

# Lets plot them and see how close they are related

##########################################################



plots(train_X, 'GrLivArea', 'SalePrice')
##########################################################

# Anything above 3,000 sq ft is of no use as these are

# outliers. These houses have very high prices, as well.

# High value priced houses are also less in numbers in the

# data set and hence, it is better to remove them else

# they can skew the learning

##########################################################

#Deleting outliers

train_X = train_X.drop(train_X[(train_X['GrLivArea']>3000) & (train_X['SalePrice']<300000)].index)
##########################################################

# Anything above $500,000 of no use as these are

# outliers. Very less number of houses will skew the model

# as it will try to include that value in the regression

# based on the standard deviation and SD will vary a lot.

##########################################################

#Deleting outliers



train_X = train_X.drop(train_X[(train_X['SalePrice']>=500000)].index)

train_X = train_X.drop(train_X[(train_X['GrLivArea']>2900)].index) # further removing outliers
##########################################################

# Checking how's it looking now

##########################################################



plots(train_X,'GrLivArea', 'SalePrice', True)
# Looks reasonably good now in the above figure but there could be

# a very high valued but low in numbers (inventory) houses which 

# we should drop
#########################################################

# Shape of the Train and Test data

#########################################################

print(train_X.shape) #(1454, 80) # after getting rid of outliers

print(train_X.head())
# Printing column headers

train_X.select_dtypes(include=['object', 'int64']).columns
#########################################################

# Feature Engineering

#########################################################



n_train = train_X.shape[0] #will be useful later



# SalePrice is the Target. This is the only column not in the TEST data

# Scaling it using the np log1p function



for_compare_Y = train_X["SalePrice"] #to be used in future

train_X["SalePrice"] = np.log1p(train_X["SalePrice"])

train_Y = train_X["SalePrice"]

train_X.drop(['SalePrice'], axis=1, inplace=True) # SalePrice is the target



print("Printing Train Y values")

print(for_compare_Y[0])

print(train_Y[0])

print(train_Y.shape)

print(np.exp(train_Y[0])) #reverse log. Just to make sure, we get the same values back

                          #as you can see there is a slight differene when we reverse it back

                          #precision is not maintained.
# Even though there are NON NULL columns in the data set, it is important to 

# understand for a given feature of the house, such as Swimming Pool,

# how many houses in the inventory actually has a Pool. If for most of the given

# inventory has no Pool, we might want to drop those columns as well.
missing_na = (train_X.isnull().sum() / len(train_X)) * 100



#print(missing_na) #this will be the list of cols with null values

missing_na = missing_na.drop(missing_na[missing_na == 0].index).sort_values(ascending=False)

print("Sorted missing_na...")

print(missing_na)
#PoolQC          99.725369 #less than 1% of all houses has a pool in this inventory

#MiscFeature     96.395469

#Alley           93.202884

#Fence           80.432544



##################################################

# We can drop these features since majority of

# houses do not have one

##################################################

train_X.drop(['PoolQC'], axis=1, inplace=True)

train_X.drop(['MiscFeature'], axis=1, inplace=True)

train_X.drop(['Alley'], axis=1, inplace=True)

train_X.drop(['Fence'], axis=1, inplace=True)



#FireplaceQu     49.032481 # It will be good to remove.. let's see how model shapes up
#          [0]          [1]          [2]           [3]           [4]

#Index(['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BldgType',

#       'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',

#       'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF',

#           [15]        [16]

#       'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'EnclosedPorch',

#                                                                   [24]

#       'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'FireplaceQu',

#                                      [27]                        [29]

#       'Fireplaces', 'Foundation', 'FullBath', 'Functional', 'GarageArea',

#           [30]                                                    [34]

#       'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType',

#           [35]           [36]        [37]        [38]

#       'GarageYrBlt', 'GrLivArea', 'HalfBath', 'Heating', 'HeatingQC',

#           [40]

#       'HouseStyle', 'KitchenAbvGr', 'KitchenQual', 'LandContour', 'LandSlope',

#           [45]

#       'LotArea', 'LotConfig', 'LotFrontage', 'LotShape', 'LowQualFinSF',

#           [50]

#       'MSSubClass', 'MSZoning', 'MasVnrArea', 'MasVnrType', 'MiscVal',

#           [55]

#       'MoSold', 'Neighborhood', 'OpenPorchSF', 'OverallCond', 'OverallQual',

#           [60]

#       'PavedDrive', 'PoolArea', 'RoofMatl', 'RoofStyle', 'SaleCondition',

#           [65]

#       'SaleType', 'ScreenPorch', 'Street', 'TotRmsAbvGrd', 'TotalBsmtSF',

#           [70]                                                  [74]

#       'Utilities', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold'],

#      dtype='object')
#########################################################

# Shape of the Train after dropping columns that we

# identified with least impact on the outcome

#########################################################

print(train_X.shape) #(1459, 75)
#########################################################

# For the remaining one lets impute (fill-in) most common

# values for the type Object

#########################################################



train_X = fill_and_encode_pandas_array(train_X, 'impute', 'object', 'no encoding')

print(train_X.head())
missing_na = (train_X.isnull().sum() / len(train_X)) * 100

missing_na = missing_na.drop(missing_na[missing_na == 0].index).sort_values(ascending=False)

print("missing_na shape and size..")



print(missing_na.shape)

print(missing_na.size)

if(missing_na.size > 0):

    print("Need to remove columns with all NULL, NaN or zeros")

    exit(-1)
#########################################################

# Label encoding categorical columns

#########################################################

train_X = fill_and_encode_pandas_array(train_X, 'none', 'object', 'label_encoding')

print("The shape of train_X data is...")

print(train_X.shape)
#########################################################

# Train Test Split. Dropping Sales Price as it is the target

# From the train data set itself, setting 30% (0.3) data

# apart for training the model

# In below, y_train and y_test are actually the Sales Price

# while on "X" side entire housing data remains except for

# the Sales price as housing data is an input to the model

# with the target is a Sales Price

#########################################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, # all but "SalePrice"

                                                    train_Y, #Only SalePrice

                                                    test_size=0.3, random_state=101)



print("Train Split info")

print(X_train.head())

print(X_test.head())

print(y_train.head())

print(y_test.head())
#################################

# we are going to scale to data

#################################



y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



print("y train and y test after reshape")

print(y_train.shape)

print(y_test.shape) # This will be the 30% (.3) of the combined data

print(y_test.size)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_y.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)



print("X Train Shape & size")

print(X_train.shape)

print(X_train.size)
#####################################

# Modeling

#Defining the Neural Network Model

#####################################



model = ks.Sequential()



from keras import optimizers

from keras.layers import Activation, Dense



model = ks.Sequential()

model.add(Dense(128, kernel_initializer='uniform', input_shape=(75,)))

model.add(Activation('sigmoid'))

model.add(Dense(1, activation='sigmoid'))
print("---- compiling the model ----\n")



sgd = optimizers.SGD(lr=.0001, decay=1e-5, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=sgd, metrics = ['accuracy'])



from keras.layers.normalization import BatchNormalization

################################################

# BatchNormalization HELPED significantly to minimize the loss

################################################

model.add(BatchNormalization())



# simple early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
##########################################################################

# You can save your model, if you wish and use it in future

##########################################################################



#mc = ModelCheckpoint("C:\\Users\\Hemal\\PycharmProjects\\Kaggle.Santander\\best_model.1.h5",

#                            monitor='val_loss', mode='min', save_best_only=True, verbose=1)
print("---- running the model.fit ----\n")

####################################################################################

# Had to spend quite a bit of time - as well do - to tune this and get the loss down

####################################################################################

history=model.fit(X_train, y_train, validation_split=0.40,

                  callbacks=[es], epochs=1000, batch_size=32, shuffle=False, verbose=0)
# plot the accuracy and loss

# in the "mean_squared_error" loss, accuracy does not make any sense



plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='test loss')

plt.title('Plot History: Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
#################################################

# Now we will use the test data and see how model

# is behaving

#################################################



data_set_housing_testing = readfile("test.csv")

test_id = data_set_housing_testing['Id']

id_column = pan.DataFrame(test_id, columns=['Id'])
#################################################

# Performing the same clean-up out of test data

# Here axis = 1 will persist the data in the dataframe

#################################################

data_set_housing_testing.drop(['PoolQC'], axis=1, inplace=True)

data_set_housing_testing.drop(['MiscFeature'], axis=1, inplace=True)

data_set_housing_testing.drop(['Alley'], axis=1, inplace=True)

data_set_housing_testing.drop(['Fence'], axis=1, inplace=True)



#we have to drop one more column, as training data does not have the SalePrice

data_set_housing_testing.drop(['FireplaceQu'], axis=1, inplace=True) 
#########################################################

# Label encoding categorical columns - Test Data

#########################################################

data_set_housing_testing = fill_and_encode_pandas_array(data_set_housing_testing, 'impute', 'object', 'no encoding')

data_set_housing_testing = fill_and_encode_pandas_array(data_set_housing_testing, 'none', 'object', 'label_encoding')



data_set_housing_testing = sc_X.fit_transform(data_set_housing_testing)
#########################################

#Predictions

#########################################

predict = model.predict(data_set_housing_testing)



print("Looking at predictions")

print(predict.shape)

test_prediction =sc_y.inverse_transform(predict)

test_prediction = np.exp(test_prediction) #This will reverse it back from "log of Sales Price"

test_prediction = pan.DataFrame(test_prediction, columns=['SalePrice'])

print("Predictions of Sales Price with the test data")

print(test_prediction.head())

result = pan.concat([test_id,test_prediction], axis=1)

print(result.head())
result.to_csv("../working/submission.Hemal.Feb142020.csv", index=False)