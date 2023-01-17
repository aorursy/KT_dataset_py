from IPython import display

display.Image('https://raw.githubusercontent.com/Dutta-SD/Images_Unsplash/master/Kaggle/dorian-mongel-5Rgr_zI7pBw-unsplash.jpg', width = 3000, height = 500)
# Import Necessary libraries

import pandas as pd

import numpy as np
# import dataset

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
# Head of training data

train_data.head()
# Head of submission file

test_data.head()
PassengerID = test_data.PassengerId

## code for dropping data

train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True, axis=1)

test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True, axis=1)



test_data.head()
# Check for NaN values

print(train_data.isnull().any())
test_data.isnull().any()
# Split into dependent and independent dataframes

y = train_data.Survived



# drop the Survived columns from the independent features

## Retaining only dependendt features

X = train_data.drop(['Survived'], axis = 1)



print(y.head())

print(X.head())
## Useful Plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns
# survived

sns.barplot(x = y.unique(), y = y.value_counts());
# Pairplot

sns.pairplot(data = train_data, corner = True, palette = 'summer');
X_train, X_test, y_train, y_test = X, test_data, y, None
# The indexes are random order, we need to reset them

X_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)



y_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)



X_train.info()
# Lets separate the object data from numerical data

s = (X_train.dtypes=='object')

categorical_cols = list(s[s].index)



# Get numerical data column names

numerical_cols = [ i for i in X_train.columns if not i in categorical_cols ]

numerical_cols
from sklearn.impute import KNNImputer

##from sklearn.preprocessing import StandardScaler   ## We turned off scaling here, you can try if you want



# Imputer Object

nm_imputer = KNNImputer()

## ss is the scaler, you can try it if you want

### We will not scale here

# ss = StandardScaler()



# Transform the necessary columns

X_train_numerical = pd.DataFrame(nm_imputer.fit_transform(X_train[numerical_cols]),

                                 columns = numerical_cols)

###X_train_numerical = pd.DataFrame(ss.fit_transform(X_train_numerical[numerical_cols]), columns = numerical_cols)



X_test_numerical = pd.DataFrame(nm_imputer.transform(X_test[numerical_cols]),

                                 columns = numerical_cols)

#X_test_numerical = pd.DataFrame(ss.transform(X_test_numerical[numerical_cols]), columns = numerical_cols)
# Drop the non required columns(with missing values)

X_train = X_train.drop(numerical_cols, axis = 1)

X_test = X_test.drop(numerical_cols, axis = 1)



# put new colums in dataframe by joining

X_train = X_train.join(X_train_numerical)

X_test = X_test.join(X_test_numerical)



X_train.isnull().any()
# Impute categorical columns

from sklearn.impute import SimpleImputer



# Imputer Object

nm_imputer = SimpleImputer(strategy='most_frequent')



# Transform the necessary columns

X_train_numerical = pd.DataFrame(nm_imputer.fit_transform(X_train[categorical_cols]),

                                 columns = categorical_cols)



X_test_numerical = pd.DataFrame(nm_imputer.transform(X_test[categorical_cols]),

                                 columns = categorical_cols)
# Drop the non required columns(with missing values)

X_train = X_train.drop(categorical_cols, axis = 1)

X_test = X_test.drop(categorical_cols, axis = 1)



# put new colums in dataframe

X_train = X_train.join(X_train_numerical)

X_test = X_test.join(X_test_numerical)



X_train.isnull().any()
from sklearn.preprocessing import OneHotEncoder



OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse=False)



OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[categorical_cols]) )



#Reset the index

OH_cols_train.index = X_train.index

OH_cols_test.index = X_test.index



# Remove Categorical Columns

num_X_train = X_train.drop(categorical_cols, axis = 1)

num_X_test = X_test.drop(categorical_cols, axis = 1)



# Join

X_train = num_X_train.join(OH_cols_train, how='left')

X_test = num_X_test.join(OH_cols_test, how='left')



X_train.head()



                             
X_test.info()
X_train.info()
# Create a validation set 

from sklearn.model_selection import train_test_split



X_train_2, X_val, y_train_2,  y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 10)
from tensorflow import keras
from keras import Sequential

from keras.layers import BatchNormalization, Dense

## Dropout is a form of regularisation for neural networks
model = Sequential()



model.add(Dense(128, activation = 'relu', input_shape = (10,) ))

model.add(BatchNormalization())

model.add(Dense(64, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dense(8, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dense(1, activation = 'sigmoid'))



model.summary()
model.compile(optimizer='adam',

              loss=keras.losses.BinaryCrossentropy(),

              metrics = ['accuracy'])
history = model.fit(

    X_train_2,

    y_train_2,

    batch_size=32,

    epochs=20,

    validation_data=(X_val, y_val)

)
model.compile(optimizer='adam',

              loss=keras.losses.BinaryCrossentropy(),

              metrics = ['accuracy'])





history = model.fit(

    X_train,

    y_train,

    batch_size=32,

    epochs=20

)
y_preds = model.predict_classes(X_test)
y_preds[:2]
file_name = "MyTitanicSubmission.csv"



y_pred_series = pd.Series(y_preds.flatten(), name = 'Survived')



file = pd.concat([PassengerID, y_pred_series], axis = 1)



file.to_csv(file_name, index = False)