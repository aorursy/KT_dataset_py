# Imported all required libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

import statsmodels.api as sm

import warnings

warnings.filterwarnings("ignore")
# Getting our train and test datasets from csv to dataframe.

df_train = pd.read_csv("../input/train.csv")

X_test = pd.read_csv("../input/test.csv")
# Checking shape of both train and test dataset.

print('Shape of training dataset: {}'.format(df_train.shape))

print('Shape of test datast: {}'.format(X_test.shape))



# Seaprating dependent variable from the independent ones because we do not want to mess with the target variable.

y_train = df_train.SalePrice

X_train = df_train.drop('SalePrice',1)



# Storing IDs of test datapoints to make final submissions of predicted Sale Price.

test_ids = X_test['Id']



X_train.head()
# Checking if there are any null values in the target variable.

y_train.isnull().sum()
# Dropping the ID column from train and test and all those columns having more than tolerable null values.

X_train.drop('Id',axis=1,inplace=True)

X_test.drop('Id',axis=1,inplace=True)

drop_cols = [col for col in list(X_train.columns) if X_train.isnull().sum()[col]>100]
drop_cols
X_train.drop(drop_cols,axis=1,inplace=True)

X_test.drop(drop_cols,axis=1,inplace=True)
X_test.shape
X_train.isnull().sum().sort_values(ascending=False)
X_test.isnull().sum().sort_values(ascending=False)
'''

Imputing remaining nulll values ->

    1. Numerical columns' null values with the mean value of the column.

    2. Categorical columns' null values with the mode value of the column.

'''

for col in list(X_train.columns):

    if ((X_train[col].dtype == np.int64) or (X_train[col].dtype == np.float64)):

        X_train[col].fillna(value=X_train[col].mean(),inplace=True)

        X_test[col].fillna(value=X_test[col].mean(),inplace=True)

    else:

        X_train[col].fillna(value=X_train[col].mode()[0],inplace=True)

        X_test[col].fillna(value=X_test[col].mode()[0],inplace=True)
X_train.isnull().sum().sort_values(ascending=False)
# Separating numerical and categorical datatypes into two different dataframes for train and test both.

train_numerical = X_train.select_dtypes(include=np.number)

train_categorical = X_train.select_dtypes(exclude=np.number)

test_numerical = X_test.select_dtypes(include=np.number)

test_categorical = X_test.select_dtypes(exclude=np.number)



# Storing column names into a list for future use.

train_categorical_cols = train_categorical.columns



# One-hot encoding the categorical variables.

onehot_encoder = OneHotEncoder(sparse=False)

train_categorical = pd.DataFrame(onehot_encoder.fit_transform(train_categorical))

test_categorical = pd.DataFrame(onehot_encoder.transform(test_categorical))

train_numerical_cols = train_numerical.columns
# MinMax Scaling all the numerical columns (except target variable, obviously!) between the 0 to 1 range.

scaler = MinMaxScaler(feature_range=(0,1))

train_numerical = pd.DataFrame(scaler.fit_transform(train_numerical),columns=train_numerical_cols)

test_numerical = pd.DataFrame(scaler.fit(train_numerical).transform(test_numerical),columns=train_numerical_cols)
print('Shape of training dataset:\n 1.Numerical: {}\n 2.Categorical: {}\n'.format(train_numerical.shape,train_categorical.shape))

print('Shape of test dataset:\n 1.Numerical: {}\n 2.Categorical: {}\n'.format(test_numerical.shape,test_categorical.shape))
# Concatenating both (numerical and categorical) dataframes for train adn test again for prediction.

X_train = pd.concat([train_numerical,train_categorical],1)

X_test = pd.concat([test_numerical,test_categorical],1)
year_cols = [col for col in list(train_numerical.columns) if (col.find('Year')!=-1 or col.find('year')!=-1 or col.find('Yr')!=-1 or col.find('yr')!=-1)]
year_cols
# Splitting the train dataset into train and validation.

X_train,X_val,y_train,y_val = tts(X_train,y_train,test_size=0.3,random_state=6)
# Using XFBRegressor to predict the SalePrice.

model = XGBRegressor(max_depth=80,learning_rate=0.01,n_estimators=1000)

y_pred_val = model.fit(X_train,y_train).predict(X_val)

print('R2 Score On Validation: {}'.format(r2_score(y_val,y_pred_val)))



features = pd.Series(model.feature_importances_, index = X_train.columns)

features = features.sort_values()
# Plotting horizontal bar graph to assess the check the value of all the importance of the features.

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

features[features>=0.01].plot(kind = "barh")

plt.title("Feature Importances in the XGBoost Model For Regression")
# Making predictions on the test set.

y_pred_test = model.predict(X_test)
# Creating dataframe for final submission by concatenating Test IDs and predited Sale Price.

y_pred_test = pd.Series(y_pred_test)

submission_df = pd.concat([test_ids,y_pred_test],axis=1,keys=['Id','SalePrice']).reset_index(drop=True)

submission_df.head()
# Storing the final submission dataframe to a CSV File.

submission_df.to_csv(index=False,path_or_buf='submission.csv')