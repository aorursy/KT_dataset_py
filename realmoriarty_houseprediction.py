# import required libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load the dataset

dataset_train = pd.read_csv("/kaggle/input/house-prices-dataset/train.csv",sep=',')

dataset_test = pd.read_csv("/kaggle/input/house-prices-dataset/test.csv",sep=',')

dataset_test["SalePrice"]=-99999

# Append the test data with train for final result

dataset = dataset_train.append(dataset_test)
# See the head of the data

dataset.head()
# See the info of dataset

dataset.info()
dataset.describe()
# Drop the ID column

dataset.drop("Id",axis=1,inplace=True)
# See the correlation heatmap of the features

plt.figure(figsize=(20,20))

sns.heatmap(dataset[dataset.columns[:-1]].corr(),annot=True)
# Drop the correlated columns

dataset.drop("GarageCars",axis=1,inplace=True)

dataset.drop("GarageYrBlt",axis=1,inplace=True)

dataset.drop("1stFlrSF",axis=1,inplace=True)

dataset.drop("2ndFlrSF",axis=1,inplace=True)

dataset.drop("BsmtFullBath",axis=1,inplace=True)

dataset.drop("FullBath",axis=1,inplace=True)

dataset.drop("HalfBath",axis=1,inplace=True)

dataset.drop("TotRmsAbvGrd",axis=1,inplace=True)
# Find columns which has null values and data type object

columns_missing_data = []

for column_name in dataset.columns:

    if dataset[column_name].isnull().any() and dataset[column_name].dtype==object:

        columns_missing_data.append(column_name)

        

print(columns_missing_data)
# Columns for which null value will be filled based on existing data ratio

col_names_ratio_fill = ['MSZoning','Alley','SaleType']



# Columns for which null value will be filled by NONE

col_names_blank_fill = ['Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', ]
for col_name in col_names_blank_fill:

    dataset[col_name].fillna("NONE",inplace=True)
# Fill missing Data based on existing data ratio

def fill_nan_cols(col_name,value_names,val_counts):

    """This method takes column name,existing values and existing value count 

    and fills the NaN columns with the existing values based on their ratio"""

    total_null_rows = dataset[dataset[col_name].isnull()].shape[0]

    total_nonnull_rows = val_counts.sum()

    

    np.random.seed(0)

    

    for name,cnt in zip(value_names[:-1],val_counts[:-1]):

        fill_cnt = int((cnt / total_nonnull_rows) * total_null_rows)

        na_ind = np.array(dataset[dataset[col_name].isnull()==True][col_name].index)

        

        if len(na_ind) > fill_cnt:

            ind = np.random.choice(na_ind,size = fill_cnt,replace=False)

            dataset.at[ind,col_name] = name

    

    if len(na_ind) > 0:

        dataset[col_name].fillna(value_names[-1],inplace=True)

# Find string columns and fill missing data based on existing data ratio

for column_name in col_names_ratio_fill:

    fill_nan_cols(column_name,np.array(dataset[column_name].value_counts().index),np.array(dataset[column_name].value_counts()))
# Fill missing values for numeric columns with Mean

for column_name in dataset.columns:

    if dataset[column_name].isnull().any() and dataset[column_name].dtype!=object:

        dataset[column_name].fillna(dataset[column_name].mean(),inplace=True)
dataset.info()
# Categorical columns for label encoding

categorical_columns = [dataset.columns.get_loc(column_name) for column_name in dataset.columns[:-1] if dataset[column_name].dtype == object]

categorical_columns
# Seperate the features and target

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values

columns = dataset.columns
X
# Categorical columns for OneHotEncoding

cat_cols_onehot = categorical_columns
# Do the label encoding for all categorical features

from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

for index in categorical_columns:

    X[:,index] = label_encoder.fit_transform(X[:,index])

# Do OneHotEncoding for categorical features

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



ct = ColumnTransformer([("hot", OneHotEncoder(), categorical_columns)], remainder = 'passthrough')

X = ct.fit_transform(X).toarray()

from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()

X = robust_scaler.fit_transform(X)
train_X = X[y!=-99999]

train_y = y[y!=-99999]

test_X = X[y==-99999]
# Prepare train and test sets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_X,train_y,test_size=0.2,random_state=0)
# Create instance of models

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor





logisregress_regressor = LinearRegression()

kneighbors_regressor = KNeighborsRegressor()

decisiontree_regressor = DecisionTreeRegressor(max_depth=4)

randomforest_regressor = RandomForestRegressor()

svr_regressor = SVR()

ada_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))
# Train the models with the training data

logisregress_regressor.fit(X_train,y_train)

kneighbors_regressor.fit(X_train,y_train)

decisiontree_regressor.fit(X_train,y_train)

randomforest_regressor.fit(X_train,y_train)

svr_regressor.fit(X_train,y_train)

ada_regressor.fit(X_train,y_train)
# Find out predicted value on test data

y_pred_logisregress = logisregress_regressor.predict(X_test)

y_pred_kneighbors = kneighbors_regressor.predict(X_test)

y_pred_decisiontree = decisiontree_regressor.predict(X_test)

y_pred_randomforest = randomforest_regressor.predict(X_test)

y_pred_svr = svr_regressor.predict(X_test)

y_pred_ada = ada_regressor.predict(X_test)
# Find the scored based on different evaluation metric

from sklearn.metrics import mean_squared_error

scores_df = pd.DataFrame(data=["Logistic Regression","K Nearest Neighbor Classifier","Decision Tree Classifier","Random Forest Classifier","Support Vector Classifier","AdaBoost Regressor"],columns=["Model Names"])

scores_df["Root_Mean_Squared_Error"] = pd.DataFrame([mean_squared_error(y_test,y_pred_logisregress)**0.5,mean_squared_error(y_test,y_pred_kneighbors),mean_squared_error(y_test,y_pred_decisiontree)**0.5,mean_squared_error(y_test,y_pred_randomforest)**0.5,mean_squared_error(y_test,y_pred_svr)**0.5,mean_squared_error(y_test,y_pred_ada)**0.5])
# See the evaluation metrics output

scores_df
randomforest_regressor.fit(train_X,train_y)
test_y_pred = randomforest_regressor.predict(test_X)
pred_result = pd.DataFrame(dataset_test["Id"])

pred_result["SalePrice"] = pd.DataFrame(test_y_pred)

pred_result.to_csv("test_pred.csv",index=False)