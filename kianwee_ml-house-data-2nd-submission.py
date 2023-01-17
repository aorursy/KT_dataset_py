import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as sns 

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

%matplotlib inline
def write_submissions(file_name, test_df, predictions):

    test_df.Id = test_df.Id.astype('int32')



    output = pd.DataFrame({

        'Id': test_df.Id, 'SalePrice': predictions

    })

    output.to_csv(file_name, index=False)
def get_categorical_columns(data_df):

    return list(data_df.select_dtypes(include=['category', 'object']))



def get_numerical_columns(data_df):

    return list(data_df.select_dtypes(exclude=['category', 'object'])) 
def read_train_test_data():

    train_df = pd.read_csv('../input/home-data-for-ml-course/train.csv')

    test_df = pd.read_csv('../input/home-data-for-ml-course/test.csv')

    

    print("Shape of Train Data: " + str(train_df.shape))

    print("Shape of Test Data: " + str(test_df.shape))

    

    categorical_columns = get_categorical_columns(train_df)

    print("No of Categorical Columns: " + str(len(categorical_columns)))

    numeric_columns = get_numerical_columns(train_df)

    print("No of Numeric Columns: " + str(len(numeric_columns)))



    return train_df, test_df



train_df, test_df = read_train_test_data()
categorical_columns = get_categorical_columns(train_df)

train_df[categorical_columns].describe()
numerical_columns = get_numerical_columns(train_df)

train_df[numerical_columns].describe()
num_train_df = train_df[numerical_columns]

num_train_df.isnull().sum()
# Preparing model

X = train_df[numerical_columns].copy()

X.drop(columns=["SalePrice"], axis=1, inplace=True)

y = train_df.SalePrice



# Setting up imputer

imputer = SimpleImputer(strategy='median')

imputed_numeric_X = imputer.fit_transform(X)



imputed_numeric_test = imputer.transform(test_df[X.columns])
def simple_model(X, y, X_test):

    model = DecisionTreeRegressor(random_state=1)

    predictions = None

    try:

        model.fit(X, y)

        predictions = model.predict(X_test)

    except Exception as exception:

        print(exception)

        pass



    return predictions



predictions = simple_model(imputed_numeric_X, y, imputed_numeric_test)