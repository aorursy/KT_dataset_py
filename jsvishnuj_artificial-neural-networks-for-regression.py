import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

# importing the libraries for the conversion of categorical data

from sklearn.preprocessing import LabelEncoder

# importing the metrics 

from sklearn.metrics import mean_squared_error
# importing the dataframe into the code

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.shape

# 81 features :) or :( 
df.columns
pd.options.display.max_columns = 81
df = df.drop(['Id'], axis = 1)
# checking the missing values

for i in df.columns:

    print("Name - ", i,", Missing Values - ", df[i].isnull().sum())
df = df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
# dropping the missing samples from the missing values

df = df.dropna()
# checking the missing values

for i in df.columns:

    print("Name - ", i,", Missing Values - ", df[i].isnull().sum())
df.info()
for i in df.columns:

    if(df[i].dtypes == 'O'):

        print('feature name - ', i)

        print(df[i].value_counts())

        sns.countplot(df[i])

        plt.show()
df = df.drop(['Utilities'], axis = 1)
# converting all the categorical data into numerical data

encoder = LabelEncoder()

def encoding(dataframe_feature):

    if (dataframe_feature.dtype == 'O'):

        return encoder.fit_transform(dataframe_feature)

    return dataframe_feature

df_encoded = df.apply(encoding)
df_encoded.head()
# seperating the training and testing data

X = df_encoded.drop(['SalePrice'], axis = 1)

y = df_encoded[['SalePrice']]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 101)
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
# let us create a neural network now

# importing the neural network libraries

from keras.models import Sequential

from keras.layers import Dense
# adding the layers in the neural network

neural_network = Sequential()

neural_network.add(Dense(input_dim = train_x.shape[1], output_dim = 37, activation = 'relu', init = 'uniform'))

neural_network.add(Dense(output_dim = 37, activation = 'relu', init = 'uniform'))

neural_network.add(Dense(output_dim = train_y.shape[1], activation = 'sigmoid', init = 'uniform'))

neural_network.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])
# training the model

neural_network.fit(train_x, train_y, batch_size = 3, epochs = 30)
# predicting using the model

prediction = neural_network.predict(test_x)
# checking the accuracy of the model 

print('RMSE - ', np.sqrt(mean_squared_error(test_y, prediction)))
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df.shape
test_df = test_df.set_index('Id')

test_df = test_df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)

test_df = test_df.drop(['Utilities'], axis = 1)
# we will impute the missing values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'most_frequent')

imputed_data = imputer.fit_transform(test_df)
test_df = pd.DataFrame(data = imputed_data, columns = test_df.columns)
test_df_encoded = test_df.apply(encoding)
# apply the test data to the model

SalePrice = neural_network.predict(test_df_encoded)
submission_dataframe = pd.DataFrame()

submission_dataframe['Id'] = test_df_encoded.index

submission_dataframe['SalePrice'] = SalePrice
submission_dataframe.head()
submission_dataframe.to_csv("ANN_Regression", index = False)