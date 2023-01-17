import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#from sklearn.selection import train_test_split
X_full = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
X_full.describe
print('Total number of columns with missing values: %d' %X_full.isnull().sum().sum() )

categorical_columns = [col for col in X_full.columns if X_full[col].dtypes == 'object']
print('Total number of categorical columns: %d' %(len(categorical_columns)))

numerical_columns = [col for col in X_full.columns if col not in categorical_columns]
print('Total number of categorical columns: %d\n' %(len(numerical_columns)))

print('Looking at numerical columns:\n')
print(X_full[numerical_columns].describe)

print('\nLooking at categorical columns:\n\n')
print(X_full[categorical_columns].describe)
#Converting the 'TotalCharges' column to 'float64'
#errors='coerce' will set empty string (ie: missing values) to NaN
TotalCharges = [charge for charge in pd.to_numeric(X_full.TotalCharges, errors='coerce')] 
X_full.TotalCharges = pd.Series(TotalCharges)
#print(X_full.TotalCharges.dtypes)


#adding TotalCharges to numerical_columns 
numerical_columns = numerical_columns + ['TotalCharges']
#removing it from categorical_columns
categorical_columns.remove('TotalCharges')


print('Total number of rows with missing values in "TotalCharges" column: %d' %(X_full.TotalCharges.isnull().sum()))

#Look for the number of unique values in each categorical columns
print('Number of unique values per columns:')
for col in X_full[categorical_columns]:
    print('%s: %d' %(col,X_full[col].nunique()))
#X_full.iloc[:,1:] #selecting all rows, and all categorical columns but 'customerID' (the 1st one)

for col in X_full[categorical_columns].iloc[:,1:]:
    print('%s:%s' %(col,X_full[col].unique()))


X_full[categorical_columns]
from sklearn.preprocessing import LabelEncoder

#Let's drop 'customerID' column, as it is not helpful
X_full.drop('customerID', axis=1, inplace=True)
categorical_columns.remove('customerID')

#X_full.drop('Churn', axis=1, inplace=True)
#categorical_columns.remove('Churn')
y = X_full.Churn
X_full.drop('Churn', axis=1, inplace=True)
categorical_columns.remove('Churn')


#preprocessing categorical columns, label encoding
le = LabelEncoder()
for col in X_full[categorical_columns]:
    X_full[col] = le.fit_transform(X_full[col])
X_full[categorical_columns]
from sklearn.model_selection import train_test_split


X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size = 0.2, random_state=1)
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error

model = XGBClassifier(n_estimators=1000, learning_rate=0.05,
                     subsample=0.8, colsample_bytree= 0.8, seed=42)

model.fit(X_train,y_train,
        early_stopping_rounds=100,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False)

preds = model.predict(X_valid)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_valid, preds)
print("Accuracy:", score)
