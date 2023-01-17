# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

!pip install  -U featuretools
df = home_data

import featuretools as ft

from sklearn.preprocessing import StandardScaler



def std(df):

    scaler = StandardScaler()

    scaler.fit(df)

    scaler.transform(df)

    return pd.DataFrame(scaler.transform(df), columns=df.columns)



def featureing(df):

    df.fillna(0,inplace=True)

    df_num = df.select_dtypes(include=['number'])

    dummies = pd.get_dummies( df.select_dtypes(exclude=['number']))

    df_train = pd.concat([df_num['Id'],std(df_num.drop(columns='Id'))

              ,dummies

    ],axis=1)

    dic = {}

    for k in dummies.columns:

        dic[k] = ft.variable_types.Categorical



    es = ft.EntitySet()

    es = es.entity_from_dataframe(entity_id='df',dataframe=df_train,

                                  variable_types=dic)

    feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'df',

                                          trans_primitives = ['add_numeric'])



    return feature_matrix
df_train = featureing(df.drop(columns='SalePrice'))

df_train.head()
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data  = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = featureing(test_data)
cls = test_X.columns & df_train.columns
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import lightgbm as lgb







def modeling():

    # To improve accuracy, create a new Random Forest model which you will train on all training data

    model = lgb.LGBMRegressor()

    return model

    # fit rf_model_on_full_data on all data from the training data



X_train, X_val, y_train, y_val = train_test_split(df_train[cls],  df['SalePrice'], test_size=0.7, random_state=42)

X_eval, X_test, y_eval, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)



model =  modeling().fit(X_train, y_train)



print(model.score(X_train,y_train))

print(model.score(X_test,y_test))

model = modeling().fit(df_train[cls], df['SalePrice'])

# make predictions which we will submit. 

test_preds = model.predict(test_X[cls])



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
dfv = df_train

dfv['SalePrice'] = df['SalePrice']

cols = dfv.corr().nlargest(20, 'SalePrice')['SalePrice']
cols
import matplotlib.pyplot as plt

import seaborn as sns

feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_,cls)), columns=['Value','Feature'])

plt.figure(figsize=(10, 6))

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False).head(20))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()
