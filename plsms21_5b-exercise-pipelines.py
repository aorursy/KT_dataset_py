import pandas as pd
from sklearn.model_selection import train_test_split

# Read Data
data = pd.read_csv('../input/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
train_X, test_X, train_y, test_y = train_test_split(X, y)


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer as Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor(n_estimators = 100))
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)
my_imputer = Imputer()
my_model = RandomForestRegressor(n_estimators = 100)

imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)