import pandas as pd

import numpy as np

import seaborn as sns



from sklearn.model_selection import train_test_split



from sklearn.utils import check_array



from sklearn.preprocessing import OneHotEncoder



from sklearn.metrics import mean_absolute_error



from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines=False)

df.head()
# missing_val_count_by_column = (df.isnull().sum())

# print(missing_val_count_by_column[missing_val_count_by_column > 0])

df.isnull().sum()
df.duplicated().unique()
df.isna().sum()
s = (df.dtypes == 'object')

object_cols = list(s[s].index) 

# s returns True or False for each column, 

# s[s] returns only Trues, 

# s[s].index returns 'Index' list of columns with dtype = object

# list(s[s].index) returns normal list of columns with dtype = object

object_cols

for i in object_cols:

    print(f'Unique values for "{i}" column: {len(df[i].unique())}')
df.columns
y = df.average_rating

X = df[['title', 'authors', '  num_pages', 'ratings_count', 'text_reviews_count', 'publisher']] # 'language_code', 'publication_date'
corr = df.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
df.describe()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
print(f'Length of training set: {len(train_X)}')

print(f'Length of testing set: {len(val_X)}')
s = (X.dtypes == 'object')

object_cols = list(s[s].index) 

object_cols
print(f'Columns number for train_X: {len(train_X.columns)}')

print(f'Columns number for val_X: {len(val_X.columns)}')
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = train_X.index

OH_cols_valid.index = val_X.index



# One-hot encoding didn't assign column names for new added columns; put them using get_feature_names

OH_cols_train.columns = OH_encoder.get_feature_names(object_cols)

OH_cols_valid.columns = OH_encoder.get_feature_names(object_cols)



# Remove categorical columns (will replace with one-hot encoding)

num_train_X = train_X.drop(object_cols, axis=1)

num_val_X = val_X.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_train_X = pd.concat([num_train_X, OH_cols_train], axis=1)

OH_val_X = pd.concat([num_val_X, OH_cols_valid], axis=1)
def mean_absolute_percentage_error(y_true, y_pred):

    y_true = y_true.copy()

    y_true[y_true == 0] = 0.00000000000001

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs(np.divide((y_true - y_pred), y_true)))*100
regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')

regressor.fit(OH_train_X, train_y)
prediction = regressor.predict(OH_val_X)
mae_regressor = mean_absolute_error(prediction, val_y)

#print(f'KNeighborsRegressor MAPE: {mean_absolute_percentage_error(val_y, prediction)}')
pd.DataFrame(data={'predicted average rate': prediction, 'average rate': val_y})
mlr = LinearRegression()

mlr.fit(OH_train_X, train_y) 
prediction = mlr.predict(OH_val_X)

mae_mlr = mean_absolute_error(prediction, val_y)

#print(f'KNeighborsRegressor MAPE: {mean_absolute_percentage_error(val_y, prediction)}')
pd.DataFrame(data={'predicted average rate': prediction, 'average rate': val_y})
rand = RandomForestRegressor(n_estimators=50, random_state=0)

rand.fit(OH_train_X, train_y) 
prediction = mlr.predict(OH_val_X)

mae_rand = mean_absolute_error(prediction, val_y)
print(f'KNeighborsRegressor MAE: {mae_regressor}')

print(f'Multiple Linear Regression MAE: {mae_mlr}')

print(f'Random Forest Regressor MAE: {mae_rand}')