import pandas as pd
from sklearn.model_selection import train_test_split

# Read Data
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/housetrain.csv')
cols_to_use = ['LotArea',
                   'YearBuilt',
                   '1stFlrSF',
                   '2ndFlrSF',
                   'FullBath',
                   'BedroomAbvGr',
                   'TotRmsAbvGrd']
X = data[cols_to_use]
y = data.SalePrice
train_X, test_X, train_y, test_y = train_test_split(X, y)


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(predictions)
my_imputer = Imputer()
my_model = RandomForestRegressor()

imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)