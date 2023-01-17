from scipy import stats

import seaborn as sns

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score

import numpy as np

from xgboost import XGBRegressor

from sklearn.model_selection import KFold

import xgboost as xgb

import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import Normalizer

from sklearn.feature_selection import RFECV

import seaborn as sns

warnings.filterwarnings('ignore')



data = pd.read_csv('../input/train.csv')
data.head()
# Removing outliers

sns.scatterplot(data = data, x = 'LotFrontage', y = 'SalePrice')
x = data[data['LotFrontage']<200]

y = data[data['LotFrontage'].isnull()]

data = pd.concat([x,y])

data.shape

data = data.reset_index(drop=True)
y = data['SalePrice']

X = data.drop(axis=1, columns=['SalePrice','Id'])

X = X.select_dtypes(exclude = ['object'])
#data.isnull().sum()
#Simple Inputer

fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')

imputed_X = pd.DataFrame(fill_NaN.fit_transform(X))

imputed_X.columns = X.columns

imputed_X.index = X.index



print(imputed_X.shape)

#st = StandardScaler()

st = MinMaxScaler()

imputed_X = st.fit_transform(imputed_X)

imputed_X = pd.DataFrame(imputed_X, columns=X.columns)

#imputed_X.info()
col_categoricals = data.select_dtypes(include=['object'])

#col_categoricals.drop(inplace=True, columns=['PoolQC'])

for col in col_categoricals.columns:

    if col_categoricals[col].isnull().any():

        col_categoricals[col] = col_categoricals[col].fillna(col_categoricals[col].value_counts().index[0])

one_hot_categoricals = pd.get_dummies(col_categoricals)

imputed_X = pd.concat([imputed_X, one_hot_categoricals], axis = 1)

z = np.abs(stats.zscore(imputed_X))

print(z)
threshold = 3

print(np.where(z > 3))
print(len(np.where(z > 3)[0]))
#new_data.isnull().sum()

#imputed_X.info()

one_hot_categoricals.head(5)
'''def get_mae(X, y):

    return -1*cross_val_score(RandomForestRegressor(50), X, y, scoring='neg_mean_absolute_error').mean()

mae_one_hot_encoded = get_mae(imputed_X, y)

print(mae_one_hot_encoded)'''
#precessing of test data



# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



X_test = test_data.select_dtypes(exclude = ['object'])

X_test = X_test.drop(axis = 1, columns = ['Id'])

#Simple Inputer

fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')

test_imputed_X = pd.DataFrame(fill_NaN.fit_transform(X_test))

test_imputed_X.columns = X_test.columns

test_imputed_X.index = X_test.index

print(test_imputed_X.shape)

#Standardization

test_imputed_X = st.transform(test_imputed_X)

test_imputed_X = pd.DataFrame(test_imputed_X, columns=X_test.columns)

#test_imputed_X.head(5)



#one hot encoding

test_col_categoricals = test_data.select_dtypes(include=['object'])

#test_col_categoricals.drop(inplace=True, columns=['PoolQC'])

for col in test_col_categoricals.columns:

    if test_col_categoricals[col].isnull().any():

        test_col_categoricals[col] = test_col_categoricals[col].fillna(test_col_categoricals[col].value_counts().index[0])



test_one_hot_categoricals = pd.get_dummies(test_col_categoricals)



#concatenation

test_imputed_X = pd.concat([test_imputed_X, test_one_hot_categoricals], axis = 1)



final_train, final_test = imputed_X.align(test_imputed_X,join='inner', axis=1)
#Training

X_train, X_test, y_train, y_test = train_test_split(final_train, y, test_size=0.2, random_state=1)

# training with cross validation

"""from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(random_state=1)

scores = cross_val_score(model, imputed_X, y, scoring = 'neg_mean_absolute_error')

print(-1*scores.mean())"""



#modeling

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds = 5

                     , eval_set=[(X_test, y_test)], verbose = False)

#kfold = KFold(n_splits=10, random_state=7)

#results = cross_val_score(model, final_train, y, cv=kfold)

#print("The MAE would be: %f") %results.mean()

model.fit(X_train, y_train, early_stopping_rounds=5, 

         eval_set = [(X_test, y_test)], verbose = False)

predicted = model.predict(X_test)

print("The Mean Absolute Error is " + str(mean_absolute_error(y_test, predicted)))
#prediction



# make predictions which we will submit. 

test_preds = model.predict(final_test)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                    'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)