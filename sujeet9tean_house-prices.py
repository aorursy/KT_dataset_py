# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# read the train and test files
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_train.head()
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df_test.head()
# Dropping rows where the target is missing
target = "SalePrice"
df_train.dropna(axis=0, subset=[target], inplace=True)

# Combine Test and Training sets to maintain consistancy.
df = pd.concat([df_train.iloc[:, :-1], df_test], axis=0)
df.head()

print('df_train has {} rows and {} features or columns'.format(df_train.shape[0], df_train.shape[1]))
print('df_test has {} rows and {} features or columns'.format(df_test.shape[0], df_test.shape[1]))
print('df has {} rows and {} features or columns'.format(df.shape[0], df.shape[1]))

# drop the unwanted colmns
df = df.drop(["Id"], axis=1)
df.head()
df.isnull().sum()
# loking for Missing Values
def missingValuesInfo(df):
    total_count = df.isnull().sum().sort_values(ascending = False) # total None value for each columns
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2) # percentage of none value for each columns
    temp = pd.concat([total_count, percent], axis = 1, keys= ['Total_Count', 'Percent'])
    return temp.loc[(temp['Total_Count'] > 0)] #return the values for each columns

missingValuesInfo(df_train)

# Handling the missing values
def HandleMissingValues(df):   
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
    values = {}
    for a in cat_cols:
        values[a] = 'UNKOWN' # for Object columns fill using 'UNKOWN'

    for a in num_cols:
        values[a] = df[a].median() # for Numeric columns fill using median
        
    df.fillna(value=values, inplace=True)

#call the function
HandleMissingValues(df)
df.head()

# Check for any missing values
df.isnull().sum().sum()
# return the List of columns
def getObjectColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype == "object"]

#Categorical Feature Encoding
def PerformOneHotEncoding(df,columnsToEncode):
    return pd.get_dummies(df, columns = columnsToEncode)

cat_cols = getObjectColumnsList(df)
df = PerformOneHotEncoding(df, cat_cols)
df.head()
#spliting the data into train and test datasets
train_data = df.iloc[:1460,:]
test_data = df.iloc[1460:,:]
print(train_data.shape)
test_data.shape
# Get X,y for modelling
X = train_data
y = df_train.loc[:,'SalePrice']
#RidgeCV
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
ridge_cv.fit(X, y)
ridge_cv_preds = ridge_cv.predict(test_data)
ridge_cv_preds
#xgboost model
import xgboost as xgb

model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
model_xgb.fit(X, y)
xgb_preds = model_xgb.predict(test_data)
predictions = ( ridge_cv_preds + xgb_preds )/2
predictions
#make the submission data frame
submission = {
    'Id': df_test.Id.values,
    'SalePrice': predictions
}

solution = pd.DataFrame(submission)

#make the submission file
solution.to_csv('submission_house_price_pridiction.csv', index=False)
solution.head()
