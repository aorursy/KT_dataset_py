# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df.head()
df.info()
fig, ax = plt.subplots()

ax.scatter(df['YrSold'], df['SalePrice'])
df.SalePrice.plot.hist()
df.tail().T
df.sort_values(by=['YrSold'], inplace=True, ascending=True)
df.YrSold.tail(20)
df_original = df.copy()
df.info()
df.isna().sum()
i=0

for label, content in df.items():

    if pd.api.types.is_string_dtype(content):

        print(label)

        i+=1

print('--------\n',i)
for label, content in df.items():

    if pd.api.types.is_string_dtype(content):

        df[label] = content.astype('category').cat.as_ordered()
df.info()
df.isnull().sum()/len(df)
for label, content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            print(label, pd.isnull(content).sum()/len(df))
for label, content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            df[label+'_is_missing'] = pd.isnull(content)

            df[label] = content.fillna(content.median())
for label, content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            print(label, pd.isnull(content).sum()/len(df))
df.LotFrontage_is_missing.value_counts()
for label, content in df.items():

    if not pd.api.types.is_numeric_dtype(content):

        #if pd.isnull(content).sum():

        print(label, pd.isnull(content).sum()/len(df))
for label, content in df.items():

    if not pd.api.types.is_numeric_dtype(content):

        df[label+'_is_missing'] = pd.isnull(content)

        df[label] = pd.Categorical(content).codes+1
df.info()
df.isna().sum()
df.head().T
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1)



model.fit(df.drop('SalePrice', axis=1), df.SalePrice)
model.score(df.drop('SalePrice', axis=1), df.SalePrice)
X_train, y_train = df.drop('SalePrice', axis=1), df.SalePrice
from sklearn.metrics import mean_squared_log_error, mean_absolute_error



def rmsle1(y_test, y_preds):

    return np.sqrt(mean_squared_log_error(y_test, y_preds))



def show_scores1(model):

    train_preds = model.predict(X_train)

    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),

              "Training RMSLE": rmsle1(y_train, train_preds),

              "Training R^2": model.score(X_train, y_train)}

    return scores
show_scores1(model)
model_no_tune = model
df.head()
df.YrSold.value_counts()
df_val = df[df.YrSold == 2010]

df_train = df[df.YrSold != 2010]



len(df_val), len(df_train)
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice

X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
def rmsle(y_test, y_preds):

    return np.sqrt(mean_squared_log_error(y_test, y_preds))



def show_scores(model):

    train_preds = model.predict(X_train)

    val_preds = model.predict(X_valid)

    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),

              "Valid MAE": mean_absolute_error(y_valid, val_preds),

              "Training RMSLE": rmsle(y_train, train_preds),

              "Valid RMSLE": rmsle(y_valid, val_preds),

              "Training R^2": model.score(X_train, y_train),

              "Valid R^2": model.score(X_valid, y_valid)}

    return scores
len(X_train), len(X_valid)
model1 = RandomForestRegressor(n_jobs=-1)
model1.fit(X_train, y_train)
show_scores(model1)
model_train_val = model1
from sklearn.model_selection import RandomizedSearchCV



# Different RandomForestClassifier hyperparameters

rf_grid = {"n_estimators": np.arange(10, 100, 10),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2),

           "max_features": [0.5, 1, "sqrt", "auto"]}



rs_model = RandomizedSearchCV(RandomForestRegressor(),

                              param_distributions=rf_grid,

                              n_iter=20,

                              cv=5,

                              verbose=True)



rs_model.fit(X_train, y_train)
rs_model.best_params_
show_scores(rs_model)
show_scores(model_train_val)
models = {'model_no_tune': model_no_tune,

          'model_train_val': model_train_val,

          'model_rs': rs_model}
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_test.head()
for label, content in df_test.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            df_test[label+"_is_missing"] = pd.isnull(content)

            df_test[label] = content.fillna(content.median())



    # Turn categorical variables into numbers

    if not pd.api.types.is_numeric_dtype(content):

        df_test[label+"_is_missing"] = pd.isnull(content)

        # We add the +1 because pandas encodes missing categories as -1

        df_test[label] = pd.Categorical(content).codes+1
df_test.head().T
for label in set(df.drop("SalePrice", axis=1).columns) - set(df_test.columns):

#      df_test[label] = False

    print(label)
for label in set(df.drop("SalePrice", axis=1).columns) - set(df_test.columns):

     df_test[label] = False

for label in set(df.drop("SalePrice", axis=1).columns) - set(df_test.columns):

#      df_test[label] = False

    print(label)
for label in set(df_test.columns) - set(df.drop("SalePrice", axis=1).columns):

#      df_test[label] = False

    print(label)
for label in set(df_test.columns) - set(df.drop("SalePrice", axis=1).columns):

#      df_test[label] = False

    df_test = df_test.drop(label, axis=1)
for label in set(df_test.columns) - set(df.drop("SalePrice", axis=1).columns):

#      df_test[label] = False

    print(label)
df_test.head().T
predictions = {}

outputs = {}

for label, model in models.items():

    predictions[label] = model.predict(df_test)

    outputs[label] = pd.DataFrame({'Id': df_test.Id,

                                   'SalePrice': predictions[label]})
for label, output in outputs.items():

    filename = 'my_submission_'+label+'.csv'

    output.to_csv(filename, index=False)
