# Code you have previously used to load data

import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import make_column_selector, make_column_transformer

from sklearn.pipeline import make_pipeline





# Path of the file to read

melb_housing = pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')
melb_housing.head()
melb_housing.describe()
features = ['Rooms', 'Bathroom', 'YearBuilt', 'Landsize']

X = melb_housing[features]



null_columns = X.columns[X.isnull().any()]

X[null_columns].isnull().sum()
# melb_housing.Price.dropna()

# melb_housing.Bathroom.dropna()

# melb_housing.YearBuilt.dropna()

# melb_housing.Landsize.dropna()

X = X.dropna()

y = y.dropna()
# Create predicted variable and features

features = ['Rooms', 'Bathroom', 'YearBuilt', 'Landsize', 'Price']

drop = melb_housing[features]

drop = drop.dropna()

features = ['Rooms', 'Bathroom', 'YearBuilt', 'Landsize']

y = drop.Price

X = drop[features]



# Split data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Group numeric and non-numeric into lists

numeric = make_column_selector(dtype_include = "number")

non_numeric = make_column_selector(dtype_exclude = 'number')



# Create preprocessor for OH encoder and Imputer for missing numeric values

preprocessor = make_column_transformer(

    (make_pipeline(SimpleImputer(strategy = "mean"), StandardScaler()), numeric), # select all numeric columns, impute mean and scale the data

    (make_pipeline(SimpleImputer(strategy = "constant"), OneHotEncoder(handle_unknown='ignore')), non_numeric) # select all non numeric columns, impute most frequent value and OneHotEncode

) 

# Make pipeline for with model

pipe = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=0))



pipe.fit(train_X, train_y)

y_pred = pipe.predict(val_X)

val_mae = mean_absolute_error(y_pred, val_y)

val_mae