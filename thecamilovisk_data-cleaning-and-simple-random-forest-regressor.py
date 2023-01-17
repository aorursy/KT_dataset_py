# Data manipulations libraries

import pandas as pd

import numpy as np

 

# Data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

 

# Machine learning imports

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

 

# Misc

import joblib
INPUT_PATH = "../input/bluebook-for-bulldozers/"

OUTPUT_PATH = "/kaggle/working"
# This is a utility function to display all columns and rows of a dataframe.

def display_all(df):

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        display(df)
# Load TrainAndValid.csv, parsing saledate as datetime

train_df = pd.read_csv(INPUT_PATH + "TrainAndValid.csv",

                       low_memory=False,

                       parse_dates=["saledate"])

train_df.head().T
# Print information about the dataframe

train_df.info()
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

sns.distplot(train_df.SalePrice, bins=50, ax=ax0, label="SalePrice")

ax0.legend()

sns.boxplot(train_df.SalePrice, ax=ax1)

plt.show();
# Describing the numerical data.

desc_df = train_df.describe().T



# Add more useful information

desc_df["% non-null"] = desc_df["count"] / len(train_df)



desc_df
# Ploting feature correlation matrix (only numerical features).

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(train_df.corr(), annot=True, cmap="YlGnBu", cbar=False, ax=ax)

plt.show()
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 10))



# Plotting the sell count for each year

sns.countplot(x="YearMade",

              data=train_df,

              palette=sns.color_palette("Blues_d"),

              ax=ax0)

ax0.set_ylabel("Number of Sales")



# Plotting the mean price for each year

sns.barplot(x="YearMade",

            y="SalePrice",

            data=train_df,

            palette=sns.color_palette("Blues_d"),

            ax=ax1)

ax1.set_ylabel("Sale Prices Mean")



plt.xticks(rotation=90)

plt.show()
# Print the rate of unique values for MachineID

n_uniques = len(train_df.MachineID.unique())

uniques_rate = n_uniques / len(train_df)

print(f"Number of unique MachineIDs: {n_uniques} -- Rate of uniques: {uniques_rate}")
train_df.plot.scatter(x="MachineID", y="SalePrice", c="datasource", figsize=(15, 10));
# Describe non-numerical features

desc_df = train_df.describe(include="O").T



# Add more useful information.

desc_df["% non-null"] = desc_df["count"] / len(train_df)



desc_df
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 10))



# We added the missing values as as separate class for ilustration purposes.

plt_data = train_df.fillna({"Blade_Extension": "missing"})



# Plot the SalePrice distribution per class. 

sns.boxplot(x="Blade_Extension", y="SalePrice", data=plt_data, ax=ax0)

ax0.set(xlabel="")



# Plot the class counts.

sns.countplot(x="Blade_Extension", data=plt_data, ax=ax1)



plt.show()
# Implementation of RMSLE.

def root_mean_squared_log_error(y_true, y_pred):

    return np.sqrt(mean_squared_log_error(y_true, y_pred))
def add_date_parts(df):

    saledate = df.saledate



    df["sale_day"] = saledate.dt.day

    df["sale_week"] = saledate.dt.week

    df["sale_month"] = saledate.dt.month

    df["sale_quarter"] = saledate.dt.quarter

    df["sale_year"] = saledate.dt.year

    df["sale_dayofweek"] = saledate.dt.dayofweek

    df["sale_dayofyear"] = saledate.dt.dayofyear

    df["sale_weekofyear"] = saledate.dt.weekofyear

    df["sale_is_month_start"] = saledate.dt.is_month_start

    df["sale_is_month_end"] = saledate.dt.is_month_end

    df["sale_is_quarter_start"] = saledate.dt.is_quarter_start

    df["sale_is_quarter_end"] = saledate.dt.is_quarter_end

    df["sale_is_year_start"] = saledate.dt.is_year_start

    df["sale_is_year_end"] = saledate.dt.is_year_end



    # Get rid of "saledate" column

    df.drop("saledate", axis=1, inplace=True)

    

    return df
train_df = add_date_parts(train_df)

display_all(train_df.head().T)
def downcast(df):

    """

    Downcasts the columns of a Dataframe in order to save memory

    """

    df_copy = df.copy()

    

    for nm, col in df_copy.items():

        if pd.api.types.is_integer_dtype(col):

            col_min, col_max = col.min(), col.max()

            if (col_min > np.iinfo(np.int8).min

                    and col_max < np.iinfo(np.int8).max):

                df_copy[nm] = col.astype(np.int8)

            elif (col_min > np.iinfo(np.int16).min

                  and col_max < np.iinfo(np.int16).max):

                df_copy[nm] = col.astype(np.int16)

            elif (col_min > np.iinfo(np.int32).min

                  and col_max < np.iinfo(np.int32).max):

                df_copy[nm] = col.astype(np.int32)

            else:

                df_copy[nm] = cols.astype(np.int64)

        elif pd.api.types.is_float_dtype(col):

            col_min, col_max = col.min(), col.max()

            #-----------------------------------------------------------

            # In pandas stable, half floats (float16) is not implemented

            #-----------------------------------------------------------

            # if (col_min > np.finfo(np.float16).min

            #         and col_max < np.finfo(np.float16).max):

            #     df_copy[nm] = col.astype(np.float16)

            # elif (col_min > np.finfo(np.float32).min

            #-----------------------------------------------------------

            if (col_min > np.finfo(np.float32).min

                  and col_max < np.finfo(np.float32).max):

                df_copy[nm] = col.astype(np.float32)

            else:

                df_copy[nm] = cols.astype(np.float64)

        elif pd.api.types.is_object_dtype(col):

            df_copy[nm] = col.astype("category")

            

    return df_copy
old_memory_usage = train_df.memory_usage(index=True, deep=True).sum()
train_df = downcast(train_df)

train_df.info()
new_memory_usage = train_df.memory_usage(index=True, deep=True).sum()



memory_gain_ration = new_memory_usage / old_memory_usage

print(f"Memory usage before/after downcasting: {old_memory_usage} / {new_memory_usage} -- Memory gain: {((old_memory_usage - new_memory_usage) / old_memory_usage * 100):.2f}%")
train_df.state.cat.categories
train_df.state.cat.codes
train_df.to_feather(OUTPUT_PATH + "TrainAndValid_raw.feather")
train_df = pd.read_feather(OUTPUT_PATH + "TrainAndValid_raw.feather")
for nm, col in train_df.items():

    if pd.api.types.is_categorical_dtype(col):

        # Replace the categorical values with their codes.

        # As the missing values are represented with category code "-1",

        # we add 1 to the codes. So all code values are positive

        train_df[nm] = col.cat.codes + 1
for nm, col in train_df.items():

    # Search column for missing values

    is_missing = pd.isnull(col)

    # Check if column type is numerical

    if pd.api.types.is_numeric_dtype(col) and is_missing.sum():

        # Create a missing values indicator column

        train_df[nm + "_is_missing"] = is_missing

        # Fill missing values

        train_df[nm] = col.fillna(col.median())
train_df.info()
display_all(train_df.head().T)
display_all(train_df.isna().sum() / len(train_df))
train_df.to_feather(OUTPUT_PATH + "TrainAndValid_preprocessed.feather")
train_df = pd.read_feather(OUTPUT_PATH + "TrainAndValid_preprocessed.feather")
valid_df = train_df[train_df.sale_year == 2012]

train_df = train_df[train_df.sale_year != 2012]



# Split the sets in independent variables and dependent variables

X_train, y_train = train_df.drop("SalePrice", axis=1), train_df.SalePrice

X_valid, y_valid = valid_df.drop("SalePrice", axis=1), valid_df.SalePrice



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
def score_model(model):

    """

    Computes the MAE, R2 and RMSLE scores.

    """

    train_pred = model.predict(X_train)

    valid_pred = model.predict(X_valid)

    return {

        "Train MAE": mean_absolute_error(y_train, train_pred),

        "Valid MAE": mean_absolute_error(y_valid, valid_pred),

        "Train R2": r2_score(y_train, train_pred),

        "Valid R2": r2_score(y_valid, valid_pred),

        "Train RMSLE": root_mean_squared_log_error(y_train, train_pred),

        "Valid RMSLE": root_mean_squared_log_error(y_valid, valid_pred),

    }
model = RandomForestRegressor(n_jobs=-1, random_state=42)

 

model.fit(X_train, y_train)

 

model.score(X_valid, y_valid)
# Using our evaluation function

score = score_model(model)

score
# Grid parameters.

rs_params = {

    "n_estimators": 2 ** np.arange(1, 7, 2) * 10,

    "max_features": [0.3, 0.5, "auto", "sqrt", "log2"],

    "max_depth": np.arange(5, 36, 10),

    "min_samples_leaf": np.arange(1, 7, 2),

    "min_samples_split": np.arange(10, 17, 2),

    "max_samples": [1000]

}

 

# Instantiate  the grid search class

rs_model = RandomizedSearchCV(RandomForestRegressor(),

                              param_distributions=rs_params,

                              n_jobs=-1,

                              n_iter=200,

                              verbose=True,

                              random_state=42)

 

rs_model.fit(X_train, y_train)
rs_model.best_params_
joblib.dump(rs_model, OUTPUT_PATH + "rs_model.bz2", compress=True)
rs_model = joblib.load(OUTPUT_PATH + "rs_model.bz2")
%%time

score_model(rs_model)
%%time

model = RandomForestRegressor(

    n_estimators=rs_model.best_params_["n_estimators"],

    max_depth=rs_model.best_params_["max_depth"],

    max_features=rs_model.best_params_["max_features"],

    min_samples_leaf=rs_model.best_params_["min_samples_leaf"],

    min_samples_split=rs_model.best_params_["min_samples_split"],

    n_jobs=-1,

    random_state=42

    )



model.fit(X_train, y_train)
%%time

score_model(model)
joblib.dump(model, OUTPUT_PATH + "model.bz2", compress=True)
model = joblib.load(OUTPUT_PATH + "model.bz2")
# We'll train a classifier that uses only half o features for the splits.

model = RandomForestRegressor(n_estimators=320,

                              max_features=0.5,

                              max_depth=25,

                              min_samples_leaf=3,

                              min_samples_split=10,

                              n_jobs=-1,

                              random_state=42)
%%time

model.fit(X_train, y_train)
%%time

score_model(model)
joblib.dump(model, OUTPUT_PATH + "model_max_features.bz2", compress=True)
model = joblib.load(OUTPUT_PATH + "model_max_features.bz2")
df_raw = pd.read_feather(OUTPUT_PATH + "TrainAndValid_raw.feather")
def adjust_types(df, ref_df):

    df_copy = df.copy()

    

    for nm, col in df_copy.items():

        if pd.api.types.is_categorical_dtype(col):

            categories = ref_df[nm].cat.categories

            df_copy[nm] = pd.Categorical(col, categories=categories, ordered=True)

        else:

            df_copy[nm] = col.astype(ref_df[nm].dtype)

    

    return df_copy





def preprocess(df, ref_df=None):

    

    df_copy = df.copy()

    

    df_copy = add_date_parts(df_copy)

    

    if ref_df is None:

        df_copy = downcast(df_copy)

    else:

        df_copy = adjust_types(df_copy, ref_df)

    

    for nm, col in df_copy.items():

        is_missing = pd.isnull(col)

        if pd.api.types.is_numeric_dtype(col):

            if ref_df is None:

                if is_missing.sum():

                    df_copy[nm + "_is_missing"] = is_missing

                    # Fill missing values with col median of df_copy

                    df_copy[nm] = col.fillna(col.median())

            else:

                ref_col = ref_df[nm]

                ref_have_missing = pd.isnull(ref_col).sum()

                if ref_have_missing:

                    df_copy[nm + "_is_missing"] = is_missing

                    # Fill missing values with col median of ref_df

                    df_copy[nm] = col.fillna(ref_col.median())

                

        elif pd.api.types.is_categorical_dtype(col):

            df_copy[nm] = col.cat.codes + 1

    

    return df_copy
test_df = pd.read_csv(INPUT_PATH + "Test.csv", low_memory=False, parse_dates=["saledate"])

test_df.head().T
test_df.info()
test_df = preprocess(test_df, df_raw)

display_all(test_df.head().T)
test_df.info()
# Make the predictions

test_preds = model.predict(test_df)



# Prepare the submission Dataframe

submission_df = pd.DataFrame()

submission_df["SalesID"] = test_df["SalesID"]

submission_df["SalesPrice"] = test_preds



display_all(submission_df)
# Save the predictions file.

submission_df.to_csv("test_predictions.csv")