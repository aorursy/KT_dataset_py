import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv', parse_dates = ['saledate'])
df.head()
df.head().T
df.info()
df.saledate.head(10)
df.sort_values(['saledate'], ascending=True, inplace=True)

df['saledate'].head(10)
data = df.copy()
data.head()
data['saleyear'] = data.saledate.dt.year

data['salemonth'] = data.saledate.dt.month

data['saleday'] = data.saledate.dt.day

data['saledayofweek'] = data.saledate.dt.dayofweek

data['saledayofyear'] = data.saledate.dt.dayofyear

data.head().T
data.drop('saledate', inplace=True, axis=1)
data.head().T
print(data['state'].value_counts())

#data['state'].value_counts().plot(kind='bar')
data.head()
data.info()
for label, content in data.items():

    if pd.api.types.is_string_dtype(content):

        data[label] = content.astype('category').cat.as_ordered()
data.info()
data['state'].cat.categories
data['state'].cat.codes
missing_values=(data.isnull().sum()/len(data)*100)
missing_values
#data.to_csv('new_data.csv', index=False)
#data = pd.read_csv('./new_data.csv')
for label, content in data.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

           print(label)
for label, content in data.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            #Add a binary column which will tell us if the data was missing

            data[label+'_is_missing'] = pd.isnull(content)

            #Fill missing numeric values with median.

            #Reason for choosing median over mean is, median is robust to outliers.

            data[label] = content.fillna(content.median())
for label, content in data.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

           print(label)
for label, content in data.items():

    if pd.api.types.is_categorical_dtype(content):

        if pd.isnull(content).sum():

            print(label)

len(label)
for label, content in data.items():

    if not pd.api.types.is_numeric_dtype(content):

            #Add a binary column which will tell us if the data was missing

            data[label+'_is_missing'] = pd.isnull(content)

            #Turn categories into numbers and then add 1.

            #Reason for adding 1 is, if there are missing values after converting

            #categories into numbers, it'll replace missing values(0) by -1.

            data[label] = pd.Categorical(content).codes+1
data.info()
data.head().T
data.isnull().sum()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
data_train = data[data.saleyear!=2012]

data_val = data[data.saleyear==2012]



len(data_train), len(data_val)
X_train, y_train = data_train.drop('SalePrice', axis=1), data_train['SalePrice']

X_valid, y_valid = data_val.drop('SalePrice', axis=1), data_val['SalePrice']
X_train.shape,y_train.shape, X_test.shape, y_test.shape
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score



def rmsle(y_test, y_pred):

    """

    Calculates Root mean squared lof error between predictions and true labels.

    """

    return np.sqrt(mean_squared_log_error(y_test, y_pred))



# Create function to evaluate model on a few different levels.



def show_scores(model):

    train_pred = model.predict(X_train)

    val_pred = model.predict(X_valid)

    #If model performs better on validation dataset, it means the model is overfitting.

    scores = {'Training MAE':mean_absolute_error(y_train, train_pred),

             "Validation MAE": mean_absolute_error(y_valid, val_pred),

             "Training RMSLE": rmsle(y_train, train_pred),

             "Valid RMSLE": rmsle(y_valid, val_pred),

             "Training R^2": r2_score(y_train, train_pred),

             "Valid R^2": r2_score(y_valid, val_pred)}

    return scores
%%time

model = RandomForestRegressor(n_jobs=-1, random_state=42,

                             max_samples=10000)

model.fit(X_train, y_train)
show_scores(model)
%%time

from sklearn.model_selection import RandomizedSearchCV



#Different RandomForestRegressor hyperparameters

rf_grid = {"n_estimators": np.arange(10, 100, 10),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2),

           "max_features": [0.5, 1, "sqrt", "auto"],

           "max_samples": [10000]}



#Instantiating RandomizedSearchCV

rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, random_state=42),

                              param_distributions=rf_grid,

                              n_iter=20,

                              cv=5,

                              verbose=True)

#Fitting the RandomizedSearchCV model

rs_model.fit(X_train, y_train)
rs_model.best_params_
show_scores(rs_model)
%%time

# Most ideal hyperparameters

ideal_model = RandomForestRegressor(n_estimators=90,

                                    min_samples_leaf=1,

                                    min_samples_split=14,

                                    max_features=0.5,

                                    n_jobs=-1,

                                    max_samples=None,

                                   random_state=42)

ideal_model.fit(X_train, y_train)
show_scores(ideal_model)
df_test = pd.read_csv('../input/bluebook-for-bulldozers/Test.csv',

                     parse_dates = ['saledate'])
df_test.shape
def preprocess_data(df):

    # Add datetime parameters for saledate

    df["saleYear"] = df.saledate.dt.year

    df["saleMonth"] = df.saledate.dt.month

    df["saleDay"] = df.saledate.dt.day

    df["saleDayofweek"] = df.saledate.dt.dayofweek

    df["saleDayofyear"] = df.saledate.dt.dayofyear



    # Drop original saledate

    df.drop("saledate", axis=1, inplace=True)

    

    # Fill numeric rows with the median

    for label, content in df.items():

        if pd.api.types.is_numeric_dtype(content):

            if pd.isnull(content).sum():

                df[label+"_is_missing"] = pd.isnull(content)

                df[label] = content.fillna(content.median())

                

        # Turn categorical variables into numbers

        if not pd.api.types.is_numeric_dtype(content):

            df[label+"_is_missing"] = pd.isnull(content)

            # We add the +1 because pandas encodes missing categories as -1

            df[label] = pd.Categorical(content).codes+1        

    

    return df
preprocess_data(df_test)
df_test.shape, X_train.shape
set(X_train.columns)-set(df_test.columns)
df_test['auctioneerID_is_missing'] = False

df_test.head()
df_test.shape, X_train.shape
test_pred = ideal_model.predict(df_test)

test_pred
len(test_pred)
df_preds = pd.DataFrame()

df_preds['SaledID'] = df_test['SalesID']

df_preds['SalesPrice'] = test_pred

df_preds
# Find feature importance of our best model

ideal_model.feature_importances_




import seaborn as sns



# Helper function for plotting feature importance

def plot_features(columns, importances, n=20):

    df = (pd.DataFrame({"features": columns,

                        "feature_importance": importances})

          .sort_values("feature_importance", ascending=False)

          .reset_index(drop=True))

    

    sns.barplot(x="feature_importance",

                y="features",

                data=df[:n],

                orient="h")
plot_features(X_train.columns, ideal_model.feature_importances_)
sum(ideal_model.feature_importances_)
df.ProductSize.isna().sum()
df.ProductSize.value_counts()
df.Turbocharged.value_counts()
df.Thumb.value_counts()