# math libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix



# Training model

# Check out 'https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html' for which model to use

from sklearn.linear_model import LinearRegression, LogisticRegression # Linear -> regression, Logistic -> classification

from xgboost import XGBRegressor, XGBClassifier # Regression and Classification

from sklearn.impute import SimpleImputer # for imputing Nan values

from sklearn.utils import shuffle



# validating model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, accuracy_score



# misc.

from IPython import display

import matplotlib.pyplot as plt

%matplotlib inline



# display data sources

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data_dir = '../input/'

# df = pd.read_csv(data_dir + '')
def discover_data(df):

    print("Head\n")

    display.display(df.head())

    print("\nDescription\n")

    display.display(df.describe())

    print("\nInfo\n")

    display.display(df.info())



# df.some_feature.value_counts() shows you all types of values for that feature
# _ = df.hist(figsize=(20, 15), bins=50)



# _ = df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 

#                 s=housing["population"]/100, label="population", # s defines radius of each circle, based on population in this case

#                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True) # c defines color of circle, based on median house value in this case
# corr_matrix = df.corr()

# corr_matrix.median_house_value.sort_values(ascending=False)

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

# _ = scatter_matrix(housing[attributes], figsize=(12, 8))

# _ = housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.2)
def drop_missing_values(df):

    new_df = df.copy()

    cols_with_missing = [col for col in new_df.columns if new_df[col].isnull().any()]

    return new_df.drop(cols_with_missing, axis=1)



def impute_missing_values(df):

    my_imputer = SimpleImputer()

    new_df = pd.DataFrame(my_imputer.fit_transform(df))

    new_df.columns = df.columns

    return new_df



def impute_with_categorical_indications(df):

    # make copy to avoid changing original data (when Imputing)

    new_data = df.copy()

    # make new columns indicating what will be imputed

    cols_with_missing = (col for col in new_data.columns if new_data[col].isnull().any())

    for col in cols_with_missing:

        new_data[col + '_was_missing'] = new_data[col].isnull()

    # Imputation

    new_data = impute_missing_values(new_data)

    new_data.columns = df.columns

    return new_data
def drop_categoricals(df):

    return df.select_dtypes(exclude=['object'])



def one_hot_encode(df):

    # drop missing values

    clean_df = drop_missing_values(df)

    # get numeric cols and unique object cols and merge

    numeric_cols = [col for col in clean_df.columns if clean_df[col].dtype in ['int64', 'float64']]

    low_cardinality_cols = [col for col in clean_df.columns if

                                   clean_df[col].nunique() < 10 and

                                   clean_df[col].dtype == "object"]

    # merge cols and return one hot encoded version of df

    my_cols = numeric_cols + low_cardinality_cols

    df_with_my_cols = clean_df[my_cols]

    return pd.get_dummies(df_with_my_cols)
def preprocess_data(df):

    # Impute for missing values, create synthetic features, etc

    return df
### Important Globals



# features = []

# target = ""

# data_dir = '../input/'

# df = pd.read_csv(data_dir + 'train.csv')

# x = preprocess_data(df[features])

# y = df[target]

# discover_data(df)



### Choose a model



# model = XGBClassifier(n_estimators=1000, learning_rate=0.05)

# model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# model = LogisticRegression()

# model = LinearRegression()
def validate_model(x, y):

    # split test and train data

    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=42, test_size=0.2)



    ### Fit linear model

#     model.fit(train_x, train_y)

    ### Fit XGB model

#     model.fit(train_x, train_y, early_stopping_rounds=5, 

#              eval_set=[(val_x, val_y)], verbose=False)



    # Calculate the mean absolute error of your Random Forest model on the validation data

    val_predictions = model.predict(val_x)

    print("First 5 predictions:", val_predictions[:5])

    print("First 5 actual values:", val_y[:5])

#     print("Accuracy: ", accuracy_score(val_predictions, val_y))  # Classification

#     print("Validation MAE for Model: ", mean_absolute_error(val_predictions, val_y))  # Regression

    

# validate_model(x, y)
def submit_predictions(model, x, y, features):

    model.fit(x, y)



    # read test data file using pandas

    test_data_path = '../input/test.csv'

    test_data = pd.read_csv(test_data_path)



    # create test features and make predictions

    test_x = test_data[features]

    processed_test_x = preprocess_data(test_x)

    test_preds = model.predict(processed_test_x)



    # Save data in format necessary to score in competition

    ids = test_data[ID]

    output_dummy = list(zip(ids, test_preds))

    output = pd.DataFrame()

    

    for my_id, pred in output_dummy:

        dummy_df = pd.Series()

        dummy_df['PassengerId'] = str(my_id)

        dummy_df['Survived'] = str(pred)

        output = output.append(dummy_df, ignore_index=True)



    output.to_csv('submission.csv', index=False)
# submit_predictions(model, x, y, features)