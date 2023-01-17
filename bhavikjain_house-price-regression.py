import pandas as pd

import numpy as np
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')
train.isnull().sum()
def fill_missing_values(df):

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    for column in list(missing.index):

        if df[column].dtype == 'object':

            df[column].fillna(df[column].value_counts().index[0], inplace=True)

        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':

            df[column].fillna(df[column].median(), inplace=True)

fill_missing_values(train)

fill_missing_values(test)

train.isnull().sum()
from sklearn.preprocessing import LabelEncoder

def impute_cats(df):

    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    object_cols_ind = []

    for col in object_cols:

        object_cols_ind.append(df.columns.get_loc(col))



    label_enc = LabelEncoder()

    for i in object_cols_ind:

        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])
impute_cats(train)

impute_cats(test)
X = train.drop('SalePrice', axis=1)

y = np.ravel(np.array(train[['SalePrice']]))

print(y.shape)
from sklearn.model_selection import train_test_split, cross_val_score, KFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators=1000,

                                      max_depth=15,

                                      min_samples_split=5,

                                      min_samples_leaf=5,

                                     )



kf = KFold(n_splits=5)

y_pred = cross_val_score(random_forest, X, y, cv=kf, n_jobs=-1)

y_pred.mean()
random_forest.fit(X, y)

rf_pred = random_forest.predict(test)
import xgboost as xgb

model = xgb.XGBRegressor()

model.fit(X,y)
p = model.predict(test)
submission1 = pd.DataFrame()



submission1['Id'] = np.array(test.index)

submission1['SalePrice'] = p
submission1.to_csv('submission1.csv', index=False)

submission1 = pd.read_csv('submission1.csv')
submission = pd.DataFrame()



submission['Id'] = np.array(test.index)

submission['SalePrice'] = rf_pred
submission.to_csv('submission.csv', index=False)

submission = pd.read_csv('submission.csv')