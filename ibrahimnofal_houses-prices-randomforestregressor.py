# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head(3)
train.info()
for label,content in train.items():

    if  pd.api.types.is_string_dtype(content):

        if pd.isnull(content).sum():

            print(label)
train.tail(3)
train.shape
train.SalePrice.hist();
plt.style.use('ggplot')

plt.figure(figsize=(10, 5))

sns.countplot(train['YrSold'], palette = 'pink')

plt.title('Comparison of each Year Total Houses Sold ', fontweight = 30, fontsize = 20)

plt.xlabel('Years')

plt.ylabel('Houses Sold');


plt.style.use('ggplot')

plt.figure(figsize=(10, 5))

sns.countplot(train['MoSold'], palette = 'pink')

plt.title('Comparison of each Month Total Houses Sold ', fontweight = 30, fontsize = 20)

plt.xlabel('Month')

plt.ylabel('Houses Sold');
plt.style.use('fivethirtyeight')

plt.figure(figsize=(15, 5))



sns.countplot(train.SaleCondition, palette = 'Blues')

plt.title('Comparison of Sales Condition', fontweight = 30, fontsize = 20)

plt.xlabel('Sale Condition')

plt.ylabel('Count');
plt.style.use('fivethirtyeight')

plt.figure(figsize=(15, 5))



sns.countplot(train.SaleType, palette = 'Blues')

plt.title('Comparison of Sales Type', fontweight = 30, fontsize = 20)

plt.xlabel('Sale Type')

plt.ylabel('Count');
# Sort DataFrame in Year order

train.sort_values(by=["YrSold"], inplace=True, ascending=True)

train.YrSold.head(5)
train.YrSold.tail(5)
# make a copy of our data

df_temp=train.copy()
df_temp.T
pd.api.types.is_string_dtype(df_temp["SaleType"])
# Find the columns which contain strings

for label, content in df_temp.items():

    if pd.api.types.is_string_dtype(content):

        print(label)
# This will turn all of the string value into category values

for label, content in df_temp.items():

    if pd.api.types.is_string_dtype(content):

        df_temp[label] = content.astype("category").cat.as_ordered()
df_temp.info()
df_temp.SaleCondition.cat.categories
df_temp.SaleCondition.cat.categories.value_counts()
# This will turn all of the string values into category values

for label,content in df_temp.items():

    if pd.api.types.is_string_dtype(content):

        df_temp[label]=content.astype("category").cat.as_ordered()
df_temp.info()
df_temp.SaleType.cat.categories
df_temp.LotShape.cat.codes
# Check missing data

df_temp.isnull().sum()/len(df_temp)
df_temp.isna().sum()
# Check for which numeric columns have null values

for label,content in train.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

           print(label)
# Fill numeric rows with the median

for label, content in df_temp.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

             

            # Fill missing numeric values with median

            df_temp[label] = content.fillna(content.median())
# Check for which numeric columns have null values

for label,content in df_temp.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

           print(label)
# Check for columns which aren't numeric

for label, content in df_temp.items():

    if not pd.api.types.is_numeric_dtype(content):

        print(label)
# Turn categorical variables into numbers and fill missing



for label,content in df_temp.items():

    if not pd.api.types.is_numeric_dtype(content):

        df_temp[label]=pd.Categorical(content).codes+1
df_temp.info()
df_temp.isna().sum()
len(df_temp)
from sklearn.ensemble import RandomForestRegressor



# Instantiate model

model = RandomForestRegressor(n_jobs=-1,

                              random_state=42)



# Fit the model

model.fit(df_temp.drop("SalePrice", axis=1), df_temp["SalePrice"])
# Score the model

model.score(df_temp.drop("SalePrice", axis=1), df_temp["SalePrice"])
df_temp.head(2)
# Split data into training and validation

df_val = df_temp[df_temp.YrSold == 2010]

df_train = df_temp[df_temp.YrSold != 2010]



len(df_val), len(df_train)
# Split data into X & y

X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice

X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
# Create evaluation function (RMSLE)



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
# Change max_samples value

model = RandomForestRegressor(n_jobs=-1,

                              random_state=42)
model.fit(X_train,y_train)
show_scores(model)
%%time

from sklearn.model_selection import RandomizedSearchCV



# Different RandomForestRegressor hyperparameters

rf_grid = {"n_estimators": np.arange(10, 100, 10),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2),

           "max_features": [0.5, 1, "sqrt", "auto"]

          }



# Instantiate RandomizedSearchCV model

rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,

                                                    random_state=42),

                              param_distributions=rf_grid,

                              n_iter=2,

                              cv=5,

                              verbose=True)



# Fit the RandomizedSearchCV model

rs_model.fit(X_train, y_train)
# Find the best model hyperparameters

rs_model.best_params_
show_scores(rs_model)
%%time



# Most ideal hyperparamters

ideal_model = RandomForestRegressor(n_estimators=50,

                                    min_samples_leaf=1,

                                    min_samples_split=2,

                                    max_features='sqrt',

                                    max_depth=3,

                                    n_jobs=-1,

                                    max_samples=None,

                                    random_state=42) # random state so our results are reproducible



# Fit the ideal model

ideal_model.fit(X_train, y_train)
show_scores(ideal_model)
test.shape
test.head(4)
def preprocess_data(df):

    """

    Performs transformations on df and returns transformed df.

    """

    # Fill the numeric rows with median

    for label, content in df.items():

        if pd.api.types.is_numeric_dtype(content):

            if pd.isnull(content).sum():

                  

                # Fill missing numeric values with median

                df[label] = content.fillna(content.median())

    

        # Filled categorical missing data and turn categories into numbers

        if not pd.api.types.is_numeric_dtype(content):

            # We add +1 to the category code because pandas encodes missing categories as -1

            df[label] = pd.Categorical(content).codes+1

    return df        

            
# Process the test data 

df_test = preprocess_data(test)

df_test.head()
# Make predictions on updated test data

test_preds = ideal_model.predict(df_test)
test_preds
# Format predictions into the same format Kaggle is after

df_preds = pd.DataFrame()

df_preds["Id"] = df_test["Id"]

df_preds["SalePrice"] = test_preds

df_preds
df_preds.to_csv("submission.csv", index=False)
ideal_model.feature_importances_
# Helper function for plotting feature importance

def plot_features(columns, importances, n=20):

    plt.style.use('ggplot')



    df = (pd.DataFrame({"features": columns,

                        "feature_importances": importances})

          .sort_values("feature_importances", ascending=False)

          .reset_index(drop=True))

    

    # Plot the dataframe

    fig, ax = plt.subplots(figsize=(15,10))

    ax.barh(df["features"][:n], df["feature_importances"][:20])

    ax.set_ylabel("Features")

    ax.set_xlabel("Feature importance")

    ax.invert_yaxis()
plot_features(X_train.columns, ideal_model.feature_importances_)