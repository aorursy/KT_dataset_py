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
from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)
my_imputer = Imputer()
my_model = RandomForestRegressor()

imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)
X_one_hots = data.copy()
X_one_hots = X_one_hots.drop(['Price'], axis=1)
X_one_hots = pd.get_dummies(X_one_hots)
train_X, test_X, train_y, test_y = train_test_split(X_one_hots, y)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error

pipeline = make_pipeline(Imputer(), RandomForestRegressor())
pipeline.fit(train_X, train_y)
predictions = pipeline.predict(test_X)
print(mean_absolute_error(predictions, test_y))
