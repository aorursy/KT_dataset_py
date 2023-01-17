import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn
# Import training and validation sets

df = pd.read_csv("../input/bluebook-for-bulldozers/TrainAndValid.csv",low_memory=False)
df.info()
fig, ax = plt.subplots()

ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000]);
df.SalePrice.plot.hist()
df.saledate[:1000]
df.saledate.dtype
# Import data again but this time parse dates

df = pd.read_csv("../input/bluebook-for-bulldozers/TrainAndValid.csv",

                 low_memory=False,

                 parse_dates=["saledate"])
df.saledate.dtype
df.saledate[:1000]


fig, ax = plt.subplots()

ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000]);
df.head()
df.head().T
# Sort DataFrame in date order

df.sort_values(by=["saledate"], inplace=True, ascending=True)

df.saledate.head(20)
# Make a copy of the original DataFrame to perform edits on

df_tmp = df.copy()
df_tmp
df_tmp["saleYear"] = df_tmp.saledate.dt.year

df_tmp["saleMonth"] = df_tmp.saledate.dt.month

df_tmp["saleDay"] = df_tmp.saledate.dt.day

df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek

df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear
df_tmp.head().T
# Now we've enriched our DataFrame with date time features, we can remove 'saledate'

df_tmp.drop("saledate", axis=1, inplace=True)
# Check the values of different columns

df_tmp.state.value_counts()
df_tmp.info()
df_tmp["UsageBand"].dtype
df_tmp.isna().sum()
pd.api.types.is_string_dtype(df_tmp["UsageBand"])
# Find the columns which contain strings

for label, content in df_tmp.items():

    if pd.api.types.is_string_dtype(content):

        print(label)
# This will turn all of the string value into category values

for label, content in df_tmp.items():

    if pd.api.types.is_string_dtype(content):

        df_tmp[label] = content.astype("category").cat.as_ordered()
df_tmp.info()
df_tmp.state.cat.categories
df_tmp.state.value_counts()
df_tmp.state.cat.codes
# Check missing data

df_tmp.isnull().sum()/len(df_tmp)
# Export current tmp dataframe

df_tmp.to_csv("../train_tmp.csv",

              index=False)
# Import preprocessed data

df_tmp = pd.read_csv("../train_tmp.csv",

                     low_memory=False)

df_tmp.head().T
df_tmp.isna().sum()
for label, content in df_tmp.items():

    if pd.api.types.is_numeric_dtype(content):

        print(label)
# Check for which numeric columns have null values

for label, content in df_tmp.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            print(label)
# Fill numeric rows with the median

for label, content in df_tmp.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            # Add a binary column which tells us if the data was missing or not

            df_tmp[label+"_is_missing"] = pd.isnull(content)

            # Fill missing numeric values with median

            df_tmp[label] = content.fillna(content.median())
# Check if there's any null numeric values

for label, content in df_tmp.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            print(label)




# Check to see how many examples were missing

df_tmp.auctioneerID_is_missing.value_counts()
pd.Categorical(df_tmp["state"])
pd.Categorical(df_tmp["state"]).codes
# Check for columns which aren't numeric

for label, content in df_tmp.items():

    if not pd.api.types.is_numeric_dtype(content):

        print(label)
# Turn categorical variables into numbers and fill missing

for label, content in df_tmp.items():

    if not pd.api.types.is_numeric_dtype(content):

        # Add binary column to indicate whether sample had missing value

        df_tmp[label+"_is_missing"] = pd.isnull(content)

        # Turn categories into numbers and add +1

        df_tmp[label] = pd.Categorical(content).codes+1
pd.Categorical(df_tmp["state"]).codes+1
pd.Categorical(df["UsageBand"]).codes # Adding 1 to represent null values with 0 not with -1
df_tmp.head().T
df_tmp.isna().sum()[:]
%%time

from sklearn.ensemble import RandomForestRegressor

# Instantiate model

model = RandomForestRegressor(n_jobs=-1,

                              random_state=42,

                              max_samples=50000)



# Fit the model

model.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])
# Score the model

model.score(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])
df_tmp.saleYear
df_tmp.saleYear.value_counts()
# Split data into training and validation

df_val = df_tmp[df_tmp.saleYear == 2012]

df_train = df_tmp[df_tmp.saleYear != 2012]



len(df_val), len(df_train)
# Split data into X & y

X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice

X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
y_train
# Create evaluation function (the competition uses RMSLE)

from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score



def rmsle(y_test, y_preds):

    """

    Caculates root mean squared log error between predictions and

    true labels.

    """

    return np.sqrt(mean_squared_log_error(y_test, y_preds))



# Create function to evaluate model on a few different levels

def show_scores(model):

    train_preds = model.predict(X_train)

    val_preds = model.predict(X_valid)

    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),

              "Valid MAE": mean_absolute_error(y_valid, val_preds),

              "Training RMSLE": rmsle(y_train, train_preds),

              "Valid RMSLE": rmsle(y_valid, val_preds),

              "Training R^2": r2_score(y_train, train_preds),

              "Valid R^2": r2_score(y_valid, val_preds)}

    return scores
len(X_train)
# Change max_samples value

model = RandomForestRegressor(n_jobs=-1,

                              random_state=42,

                              max_samples=10000)
%%time

# Cutting down on the max number of samples each estimator can see improves training time

model.fit(X_train, y_train)
(X_train.shape[0] * 100) / 1000000
10000 * 100
show_scores(model)
%%time

from sklearn.model_selection import RandomizedSearchCV



# Different RandomForestRegressor hyperparameters

rf_grid = {"n_estimators": np.arange(10, 100, 10),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2),

           "max_features": [0.5, 1, "sqrt", "auto"],

           "max_samples": [10000]}



# Instantiate RandomizedSearchCV model

rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,

                                                    random_state=42),

                              param_distributions=rf_grid,

                              n_iter=2,

                              cv=5,

                              verbose=True)



# Fit the RandomizedSearchCV model

rs_model.fit(X_train, y_train)
show_scores(rs_model)
rs_model.best_params_
%%time



# Most ideal hyperparamters

ideal_model = RandomForestRegressor(n_estimators=40,

                                    min_samples_leaf=1,

                                    min_samples_split=14,

                                    max_features=0.5,

                                    n_jobs=-1,

                                    max_samples=50000,

                                    random_state=42) # random state so our results are reproducible



# Fit the ideal model

ideal_model.fit(X_train, y_train)
# Scores on model (only trained on ~50,000 examples)

show_scores(model)
# Import the test data

df_test = pd.read_csv("../input/bluebook-for-bulldozers/Test.csv",

                      low_memory=False,

                      parse_dates=["saledate"])



df_test.head()
def preprocess_data(df):

    """

    Performs transformations on df and returns transformed df.

    """

    df["saleYear"] = df.saledate.dt.year

    df["saleMonth"] = df.saledate.dt.month

    df["saleDay"] = df.saledate.dt.day

    df["saleDayOfWeek"] = df.saledate.dt.dayofweek

    df["saleDayOfYear"] = df.saledate.dt.dayofyear

    

    df.drop("saledate", axis=1, inplace=True)

    

    # Fill the numeric rows with median

    for label, content in df.items():

        if pd.api.types.is_numeric_dtype(content):

            if pd.isnull(content).sum():

                # Add a binary column which tells us if the data was missing or not

                df[label+"_is_missing"] = pd.isnull(content)

                # Fill missing numeric values with median

                df[label] = content.fillna(content.median())

    

        # Filled categorical missing data and turn categories into numbers

        if not pd.api.types.is_numeric_dtype(content):

            df[label+"_is_missing"] = pd.isnull(content)

            # We add +1 to the category code because pandas encodes missing categories as -1

            df[label] = pd.Categorical(content).codes+1

    

    return df
# Process the test data 

df_test = preprocess_data(df_test)

df_test.head().T
# We can find how the columns differ using sets

set(X_train.columns) - set(df_test.columns)
# Manually adjust df_test to have auctioneerID_is_missing column

df_test["auctioneerID_is_missing"] = False

df_test.head()
# Make predictions on the test data

test_preds = ideal_model.predict(df_test)
test_preds
# Format predictions into the same format Kaggle is after

df_preds = pd.DataFrame()

df_preds["SalesID"] = df_test["SalesID"]

df_preds["SalesPrice"] = test_preds

df_preds
# Export prediction data

df_preds.to_csv("../test_predictions.csv", index=False)
# Find feature importance of our best model

ideal_model.feature_importances_
# Helper function for plotting feature importance

def plot_features(columns, importances, n=20):

    df = (pd.DataFrame({"features": columns,

                        "feature_importances": importances})

          .sort_values("feature_importances", ascending=False)

          .reset_index(drop=True))

    

    # Plot the dataframe

    fig, ax = plt.subplots()

    ax.barh(df["features"][:n], df["feature_importances"][:20])

    ax.set_ylabel("Features")

    ax.set_xlabel("Feature importance")

    ax.invert_yaxis()
plot_features(X_train.columns, ideal_model.feature_importances_)
df["Enclosure"].value_counts()