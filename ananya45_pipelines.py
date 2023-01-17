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
import pandas as pd
original_data = pd.read_csv('../input/melb_data.csv')
target = original_data['Price']
predictors = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'Type', 'Suburb']
x_data = original_data[predictors]
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x_data, target)
categorical_trainx = pd.get_dummies(train_x)
categorical_testx = pd.get_dummies(test_x)
train_x, test_x = categorical_trainx.align(categorical_testx, join='left', axis=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_x, train_y)
predictions = my_pipeline.predict(test_x)
mean_absolute_error(test_y, predictions)
