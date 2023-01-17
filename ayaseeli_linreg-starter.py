import numpy as np

import pandas as pd

# import xgboost as xgb

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])



df_train.head()
y_train = df_train['price_doc'].values

y_train_log = np.log(y_train)

id_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)
# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

print ("sebelum concat", df_all.shape)

df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')

print ("setelah merge", df_all.shape)
# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp'], axis=1, inplace=True)
# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_numeric_train = df_numeric[:num_train]

df_numeric_test = df_numeric[num_train:]



df_categoric = df_all.select_dtypes(include=['object']).copy()

df_categoric_train = df_categoric[:num_train]

df_categoric_test = df_categoric[num_train:]
df_numeric_train = df_numeric_train.replace(np.inf, np.nan)

df_numeric_train = df_numeric_train.astype(np.float64)

df_numeric_test = df_numeric_test.replace(np.inf, np.nan)

df_numeric_test = df_numeric_test.astype(np.float64)

print (df_numeric_test.shape)

print (df_numeric_train.shape)
from sklearn.preprocessing import Imputer

imput_numerical = Imputer(missing_values = 'NaN', strategy = 'median' )

numeric_column = list(df_numeric.columns)



def numericalImputation(data):

    imput_numerical.fit(data)

    

    numerical_data_imputed = pd.DataFrame(imput_numerical.transform(data))

    numerical_data_imputed.columns = data.columns

    numerical_data_imputed.index = data.index



    return  numerical_data_imputed, imput_numerical
x_train_numerical, imput_numerical = numericalImputation(df_numeric_train)

numerical_data_imputed = pd.DataFrame(imput_numerical.transform(df_numeric_test))

numerical_data_imputed.columns = df_numeric_test.columns

numerical_data_imputed.index = df_numeric_test.index

x_test_numerical = numerical_data_imputed

print (x_train_numerical.shape)

print (x_test_numerical.shape)
def categoricalImputation(data):

    categorical_data = data.fillna(value="KOSONG")

    return categorical_data

 

x_train_categorical = categoricalImputation(df_categoric_train)

x_test_categorical = categoricalImputation(df_categoric_test)
categorical_columns = x_train_categorical.columns

for i in categorical_columns:

    print (i, x_train_categorical[i].value_counts(normalize = True))

    print (i, x_test_categorical[i].value_counts(normalize = True))
x_train_categorical_dummies = pd.get_dummies(x_train_categorical)

x_test_categorical_dummies = pd.get_dummies(x_test_categorical)

print (x_train_categorical_dummies.shape)

print (x_test_categorical_dummies.shape)
x_train = pd.concat([x_train_numerical, x_train_categorical_dummies], axis =1)

x_test = pd.concat([x_test_numerical, x_test_categorical_dummies], axis =1)

print (x_train.shape)

print (x_test.shape)
print (np.all(np.isnan(x_train)))

print (np.all(np.isnan(x_test)))
from sklearn.preprocessing import StandardScaler



def standardizer(data):

    data_columns = data.columns  # agar nama column tidak hilang

    data_index = data.index # agar index tidak hilang

    normalize = StandardScaler()

    normalize.fit(data)

    

    normalize_x = pd.DataFrame(normalize.transform(data))

    normalize_x.columns = data_columns

    return normalize_x, normalize



x_train_normalized, normalize = standardizer(x_train)

x_test_normalized, normalize = standardizer(x_test)

print (x_train_normalized.shape)

print (x_test_normalized.shape)
common_cols = list(set(x_train_normalized.columns).intersection(x_test_normalized.columns))

x_train_normalized = x_train_normalized[common_cols]

x_test_normalized = x_test_normalized[common_cols]
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from sklearn.metrics import mean_squared_error, r2_score, make_scorer

from sklearn.externals import joblib # import function untuk save object

linreg = LinearRegression()

linreg.fit(x_train_normalized, y_train_log)

linreg.score(x_train_normalized, y_train_log)



linreg_cv = cross_val_score(linreg, x_train_normalized, y_train_log, cv = 5, scoring= 'r2'

                           )

linreg_cv_mean = linreg_cv.mean()

linreg_cv_mean_root = linreg_cv_mean ** 1/2

linreg_cv_std = linreg_cv.std()
predtest = linreg.predict(x_test_normalized)

sub = pd.DataFrame({"id": id_test,"price_doc":predtest})

sub.to_csv('submission.csv', index=False,header=True)
print ("CV Mean", linreg_cv_mean)

print ("CV Mean", linreg_cv_mean_root)

print ("CV Std Dev", linreg_cv_std)