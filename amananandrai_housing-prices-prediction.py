import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline 

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor as xgbr
train=pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

test=pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
train.head()
test.head()
train.describe()
test.describe()
train.shape
test.shape
train.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train.SalePrice

train.drop(['SalePrice'], axis=1, inplace=True)
categorical_cols = [cname for cname in train.columns if

                   train[cname].nunique() < 10 and 

                  train[cname].dtype == "object"]





numerical_cols = [cname for cname in train.columns if 

               train[cname].dtype in ['int64', 'float64']]

numeric_data=train[numerical_cols].copy()

plt.figure(figsize=(14,12))

correlation = numeric_data.corr()

sns.heatmap(correlation, mask = correlation <0.75, linewidth=0.75, cmap='YlOrBr');
train.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)

test.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)
numerical_cols
print("Number of numerical features")

print(len(numerical_cols))
categorical_cols
print("Number of categorical features which are selected")

print(len(categorical_cols))
f=plt.figure(figsize=(25,8))

f.suptitle('Number of missing rows',fontsize=30);

missing_count = pd.DataFrame(train.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(20).reset_index()

missing_count.columns = ['features','sum']

sns.barplot(x='features',y='sum', data = missing_count);
train.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)

test.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)
new_num_cols=list(set(numerical_cols)-set(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars']))
new_cat_cols=list(set(categorical_cols)-set(['PoolQC','MiscFeature','Alley']))
X_train_full, X_valid_full, y_train, y_valid = train_test_split(train, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)
my_cols = new_cat_cols + new_num_cols



X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = test[my_cols].copy()
numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, new_num_cols),

        ('cat', categorical_transformer, new_cat_cols)

    ])
my_model = xgbr(random_state=42,n_estimators=2400,learning_rate=0.01 ) 



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', my_model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
preds_test = clf.predict(X_test) 



# Save test predictions to file

output = pd.DataFrame({'Id': X_test.Id,

                       'SalePrice': preds_test})

output.to_csv('submission1.csv', index=False)