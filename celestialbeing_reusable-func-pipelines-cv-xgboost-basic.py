# Role programming concepts and Python best practices

# are not being used only on software development.

# You could utilize the same on data science as well.

# Our dataset here will also be for house price prediction



# Define needed libaries

import statistics

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error



# Define needed models

from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# It would be better if you could make your code reusable

# you can do this by defining reusable functions

# this could be improved further but the goal here is to just give you an idea



def split_mae_scorer(df, model):

    

    # Create the features and target

    df.dropna(axis=0, subset=['SalePrice'], inplace=True)

    y = df.SalePrice

    df.drop(['SalePrice'], axis=1, inplace=True)

    X = df.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    

    # Instead of dropping null values, we will do imputation instead

    numerical_transformer = SimpleImputer(strategy='constant')

    object_transformer    = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

    ])

    

    # Define numerical features and categorical features from the dataframes

    object_cols    = X_train.select_dtypes(include=['object']).columns

    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    

    # Complete all missing values from the data set by means of imputation

    # bundle all transformers using ColumnTransformer

    data_cleanser = ColumnTransformer(

        transformers=[

            ('num', numerical_transformer, numerical_cols),

            ('obj', object_transformer, object_cols)

        ]

    )

    

    # Create a pipeline

    pipe = Pipeline(steps=[

        ('Cleanser', data_cleanser),

        ('Model', model)

    ])

    

    # Start model fitting and scoring

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    return mae
# Since we have defined everything prior,

# and made everything parametric (if not all, most of them)

# we could actually reuse it over and over again if for example

# we could loop a certain number of different machine learning models

# then compare which one has a better score



# Read the csv file

data = pd.read_csv('../input/train.csv', index_col='Id')

data.head()
# Check the shape of the dataset

data.shape
# List all the models that you would like to use

models = [

    RandomForestRegressor(n_estimators=100, random_state=0),

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

]
# Execute

for model in models:

    data1 = data.copy()

    print(model)

    print(split_mae_scorer(data1, model))
# Take note that the purpose of these kernel is to show 

# how we could reuse our code. It is implicitly obvious 

# that classifier models will be of no help in this situation

# since we are doing linear prediction of house prices, thus

# regressors would be a better choice!
# I'll create a function for cross validation for comparison. Based on the result above,

# for RandomForestRegressor, we have a score of 17472.173424657536 

# Let's see if the score will somehow improve when we use cross validation

from sklearn.model_selection import cross_val_score



def cv_mae_scorer(df, model):

    

    # Create the features and target

    df.dropna(axis=0, subset=['SalePrice'], inplace=True)

    y = df.SalePrice

    df.drop(['SalePrice'], axis=1, inplace=True)

    X = df.copy()

    

    # Impute missing values

    numerical_transformer = SimpleImputer(strategy='constant')

    object_transformer    = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

    ])

    

    # Define numerical features and categorical features from the dataframes

    object_cols    = X.select_dtypes(include=['object']).columns

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    

    # Complete all missing values from the data set by means of imputation

    # bundle all transformers using ColumnTransformer

    data_cleanser = ColumnTransformer(

        transformers=[

            ('num', numerical_transformer, numerical_cols),

            ('obj', object_transformer, object_cols)

        ]

    )

    

    # Create a pipeline

    pipe = Pipeline(steps=[

        ('Cleanser', data_cleanser),

        ('Model', model)

    ])

    

    # Do cross validation on the data set

    scores = -1 * cross_val_score(pipe, X, y, cv=10, scoring='neg_mean_absolute_error')

    return scores
# Create a second copy of original data

data2 = data.copy()

data2.head()
# Using the function above for cross validation

model = RandomForestRegressor(n_estimators=100, random_state=0)

list_of_scores = cv_mae_scorer(data2, model)

print(list_of_scores)
# As you could see, per iteration, it has its own mae. So how to get the score

# of our cross validation? just get the mean

print(list_of_scores.mean())
# So far, based on the examples above, we have done:

# - defining reusable python functions

# - pipelines and transformers

# - mean absolute error scoring

# - cross validation + mean absolute error scoring
# This time, we will do gradient boosting

# This method is very common and when you read ML books,

# you will probably encounter this one. The concept is simple

# you can search it in google :) but for the function that I 

# will create below, I will use extreme gradient boosting



def xgboost_mae_scorer(df):

    

    # model definition will be done inside this function

    # together with data cleansing steps

    df.dropna(axis=0, subset=['SalePrice'], inplace=True)

    y = df.SalePrice

    df.drop(['SalePrice'], axis=1, inplace=True)

    X = df.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    

    # Impute missing values

    numerical_transformer = SimpleImputer(strategy='constant')

    object_transformer    = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

    ])

    

    # Define numerical features and categorical features from the dataframes

    object_cols    = X_train.select_dtypes(include=['object']).columns

    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    data_cleanser  = ColumnTransformer(

        transformers=[

            ('num', numerical_transformer, numerical_cols),

            ('obj', object_transformer, object_cols)

        ]

    )

    

    # We'll define three different models namely model1, model2 and model3

    # The reason why this is like is because I would like to show the effect

    # of gradient boost. We will retrain the model three times!

    

    model1 = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

    pipe   = Pipeline(steps=[('Cleanser', data_cleanser),('Model', model1)])

    pipe.fit(X_train, y_train)

    preds  = pipe.predict(X_test)

    mae1   = mean_absolute_error(y_test, preds)

    

    model2 = XGBRegressor(n_estimators=750, learning_rate=0.05, n_jobs=4)

    pipe   = Pipeline(steps=[('Cleanser', data_cleanser),('Model', model2)])

    pipe.fit(X_train, y_train)

    preds  = pipe.predict(X_test)

    mae2   = mean_absolute_error(y_test, preds)

    

    model3 = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

    pipe   = Pipeline(steps=[('Cleanser', data_cleanser),('Model', model3)])

    pipe.fit(X_train, y_train)

    preds  = pipe.predict(X_test)

    mae3   = mean_absolute_error(y_test, preds)

    

    # Create a list of results from different models, then get the mean

    result_list = []

    result_list.insert(0,mae1)

    result_list.insert(1,mae2)

    result_list.insert(2,mae3)

    return statistics.mean(result_list)
# Create a 3rd copy of our data then utilize the newly created function

data3 = data.copy()

print(xgboost_mae_scorer(data3))