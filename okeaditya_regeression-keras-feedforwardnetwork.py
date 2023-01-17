# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import preprocessing

import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')
from keras.models import Sequential

from keras import layers

from keras import optimizers

from keras.callbacks import ModelCheckpoint

from keras.callbacks import ReduceLROnPlateau

from keras.layers.advanced_activations import LeakyReLU
sample_sub_csv = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
flag = 0 

for column in test_df.columns:

    if(column == "SalePrice"):

        flag = 1

if(flag == 1):

    print("There is SalePrice beware ")

else:

    print("You need to predict SalePrice")
# We have not removed the SalePrice to analyse the Missing Values and do class-conditions Regression

print("The number of columns before making the augmentation = ",train_df.shape[1])

train_df.drop(['Alley', 'Exterior1st', 'Exterior2nd', 'FireplaceQu', 'Fence', 'MiscFeature'],axis = 1, inplace = True)

train_df.drop(['PoolQC', 'Utilities', '3SsnPorch', 'BsmtFinSF2', 'PoolArea', 'Street','MiscVal'],axis = 1, inplace = True)

train_df.drop(['EnclosedPorch','ScreenPorch','LotShape','KitchenAbvGr','GarageQual','LowQualFinSF','RoofMatl','Heating','OpenPorchSF','MasVnrArea','LandSlope'], axis = 1, inplace = True)

train_df.drop(['GarageCond','BsmtHalfBath','BsmtFinType2','GarageCars','Condition2','WoodDeckSF'],axis = 1, inplace = True)

print("The number of columns after dropping the missing valued columns and the predicting variable = ",train_df.shape[1])
print("The number of columns before making augmenation in Test set",test_df.shape[1])

test_df.drop(['Alley', 'Exterior1st', 'Exterior2nd', 'FireplaceQu', 'Fence', 'MiscFeature'],axis = 1, inplace = True)

test_df.drop(['PoolQC', 'Utilities', '3SsnPorch', 'BsmtFinSF2', 'PoolArea', 'Street','MiscVal'],axis = 1, inplace = True)

test_df.drop(['EnclosedPorch','ScreenPorch','LotShape','KitchenAbvGr','GarageQual','LowQualFinSF','RoofMatl','Heating','OpenPorchSF','MasVnrArea','LandSlope'], axis = 1, inplace = True)

test_df.drop(['GarageCond','BsmtHalfBath','BsmtFinType2','GarageCars','Condition2','WoodDeckSF'],axis = 1, inplace = True)

print("The number of columns after dropping the missing valued columns = ",test_df.shape[1])
def check_mismatch(column, train_df, test_df):

    if(train_df[column].isnull().any()):

        print("Number of rows haivng the NUll value train_df = ",train_df[column].isnull().sum())

    else:

        print("There is no Null value in train_df column ",column)

    

    if(test_df[column].isnull().any()):

        print("Number of rows haivng the NUll value in test_df = ",test_df[column].isnull().sum())

    else:

        print("There is no Null value in train_df column ",column)    
def check_missing(column, check_df):

    if(check_df[column].isnull().any()):

        print("Number of rows having the NULL value in ",check_df," are ",check_df[column].isnull().sum())

    else:

        print("There is no NULL vale in the column", column)
def encode_categorical(df, column_list, index_list):

    for column in column_list:

        df[column] = df[column].astype('str')

        encoder = preprocessing.LabelEncoder()

        encoded_list = encoder.fit_transform(df[column])

#         print(encoded_list)

#         print(len(encoded_list))

        encoded_series = pd.Series(encoded_list, index = index_list)

        df[column] = encoded_series

        print("The ", column, "is encoded ")

    return(df)
# Numeric types need to do MinxMaxScaler

def scale_data(df, column_list, index_list):

    for column in column_list:

        df[column] = df[column].astype('float')

        encoder = preprocessing.StandardScaler()

        df[column] = encoder.fit_transform(df[column].values.reshape(-1,1))

        print("The ",column, "is encoded")

    return(df)
Y_actual = train_df['SalePrice'].values

print(Y_actual)
reg_enc = preprocessing.MinMaxScaler()

Y_enc_train = train_df['SalePrice'].values.reshape(-1,1)

Y_enc_train = reg_enc.fit_transform(Y_enc_train)



Y_train = []

for i in range(len(Y_enc_train)):

    Y_train.append(Y_enc_train[i][0])



Y_train = np.array(Y_train)

    

print(len(Y_train))

print(Y_train.shape)
# print(Y_train.shape)

print(Y_train)
train_df.drop('SalePrice',axis = 1, inplace = True)
index_list = []

for i in range(1, 2920):

    index_list.append(i)
train_ending = 1460       # This is the Id numbeer

train_starting = 1

test_starting = 1461

test_ending = 2919
# Merge the two datafames to enocde them properly

train_df.set_index('Id', inplace = True)

test_df.set_index('Id', inplace = True)
# Merging the two dataframes using Id as the index

frames = [train_df, test_df]

combined_df = pd.concat(frames)
#pp.ProfileReport(combined_df)
columns_encoded = 0

categorical_list = ['MSSubClass','MSZoning','BldgType','YearBuilt','YrSold','BsmtFullBath','CentralAir','Condition1','Electrical','ExterCond','YearRemodAdd']

columns_encoded += len(categorical_list)

combined_df = encode_categorical(combined_df, categorical_list, index_list)



print("Number of columns encoded ",columns_encoded)
categorical_list = ['ExterQual','Fireplaces','Foundation','FullBath','Functional','GarageYrBlt','HalfBath','HeatingQC','HouseStyle','KitchenQual','LandContour']

columns_encoded += len(categorical_list)

combined_df = encode_categorical(combined_df, categorical_list, index_list)



print("Number of columns encoded ",columns_encoded)
categorical_list = ['LotConfig','MasVnrType','MoSold','Neighborhood','OverallCond','OverallQual','PavedDrive','RoofStyle','SaleCondition','SaleType','TotRmsAbvGrd']

columns_encoded += len(categorical_list)

combined_df = encode_categorical(combined_df, categorical_list, index_list)



print("Number of columns encoded ",columns_encoded)
#pp.ProfileReport(combined_df)
scale_column_list = ['BsmtUnfSF', '2ndFlrSF', 'BsmtFinSF1', 'GarageArea', 'GrLivArea', 'LotArea', 'TotalBsmtSF','1stFlrSF']

columns_encoded += len(scale_column_list)

combined_df = scale_data(combined_df, scale_column_list, index_list)



print("Number of columns encoded ",columns_encoded)
combined_df['GarageType'] = combined_df['GarageType'].fillna(value = 'None')
combined_df['GarageFinish'] = combined_df['GarageFinish'].fillna(value = 'None')

combined_df['BsmtQual'] = combined_df['BsmtQual'].fillna(value = 'None')

combined_df['BsmtFinType1'] = combined_df['BsmtFinType1'].fillna(value = 'None')

combined_df['BsmtExposure'] = combined_df['BsmtExposure'].fillna(value = 'None')

combined_df['BsmtCond'] = combined_df['BsmtCond'].fillna(value = 'None')
categorical_list = ['GarageType','GarageFinish', 'BsmtQual', 'BsmtFinType1', 'BsmtExposure','BsmtCond']

columns_encoded += len(categorical_list)

combined_df = encode_categorical(combined_df, categorical_list, index_list)

print(columns_encoded)
combined_df['LotFrontage'] = combined_df['LotFrontage'].fillna(value = 0)

scale_list = ['LotFrontage']

columns_encoded += len(scale_list)

combined_df = scale_data(combined_df, scale_list, index_list)

print(columns_encoded)
combined_df.shape
# The only non-enocded value is the ID column which I will use to spilt to data into train and test as it was before

# The data may sound crazy as LotArea is being treated as negative and Garage Area has negative values

# But they have no meaning to us as our network needs to learn the values properly



# We again need to scale the categorical values so that our network can learn properly.
combined_df.head()
l1 = ['MSSubClass','MSZoning','BldgType','YearBuilt','YrSold','BsmtFullBath','CentralAir','Condition1','Electrical','ExterCond','YearRemodAdd']

l2 = ['ExterQual','Fireplaces','Foundation','FullBath','Functional','GarageYrBlt','HalfBath','HeatingQC','HouseStyle','KitchenQual','LandContour']

l3 = ['LotConfig','MasVnrType','MoSold','Neighborhood','OverallCond','OverallQual','PavedDrive','RoofStyle','SaleCondition','SaleType','TotRmsAbvGrd']

l4 = ['GarageType','GarageFinish', 'BsmtQual', 'BsmtFinType1', 'BsmtExposure','BsmtCond']
column_list = l1 + l2 + l3 + l4

combined_df = scale_data(combined_df, column_list, index_list)
# Split it as it was before

train_df = combined_df[:1460]

test_df = combined_df[1460:]
train_df.head()
train_df.tail()
test_df.head()
test_df.tail()
# Drop the indices for the test dataset

train_df = train_df.rename_axis(None)
train_df.head()
train_df.shape
# Create the train_set from the dataframe

X_train = train_df.values

print(X_train.shape)

#Note that we already created the Y_train above
# Now we can easily regress with our Keras Feed-Forward Nueral Network

model = Sequential()

model.add(layers.Dense(49, input_shape = (49,)))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.2))



model.add(layers.Dense(40, activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.1))



model.add(layers.Dense(32, activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.1))



model.add(layers.Dense(24, activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.1))





model.add(layers.Dense(24, activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.2))



# model.add(layers.Dense(8, activation = 'relu'))

# model.add(layers.BatchNormalization(axis=1))

# model.add(layers.Dropout(0.1))



model.add(layers.Dense(1, activation = 'relu'))
# All the hyperparameters are here

learning_rate = 0.025

batch_size = 32

epochs = 100
optim = optimizers.Adam(lr = learning_rate)
model.compile(optimizer = optim, loss = 'mean_squared_error', metrics = ['mse'])
checkpoint = ModelCheckpoint("/kaggle/input/best_model.h5", monitor = 'val_loss', save_best_only=True, verbose = 1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=0.001)
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1, callbacks = [reduce_lr, checkpoint])
print(X_train.shape)
model.load_weights('/kaggle/input/best_model.h5')
# Checking how our model performs on Train Dataset



#our predictions

pred_price = model.predict(X_train)

pred_price = reg_enc.inverse_transform(pred_price)
Y_pred = []

for i in range(len(pred_price)):

    Y_pred.append(pred_price[i][0])



Y_pred = np.array(Y_pred)



print(Y_pred)
print(Y_actual)
X = []

for i in range(len(Y_pred)):

    X.append(i)
plt.plot(X,Y_pred, c = 'b')

plt.plot(X,Y_actual, c= 'g')

plt.show()
plt.scatter(X,Y_pred, c = 'b')

plt.scatter(X,Y_actual, c= 'g')

plt.show()
test_final = test_df.rename_axis(None, inplace = False)
test_final_arr = test_final.values
predictions = model.predict(test_final_arr)

predictions =  reg_enc.inverse_transform(predictions)
Y_test_predictions = []

for i in range(len(pred_price)):

    Y_test_predictions.append(pred_price[i][0])



Y_test_predictions = np.array(Y_pred)
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.DataFrame()

submission['Id'] = test_df['Id']

submission['SalePrice'] = pd.Series(Y_test_predictions)
submission.head()
submission.tail()
submission.to_csv('submission.csv', index=False)