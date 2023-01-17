from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Of course, we're gonna have to do the regular test-train split and OHE.
# Apparently you can use pipeline for OHE, but the tutorial does not cover it.
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
ohe_data = pd.get_dummies(data)

my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = ohe_data.drop(['SalePrice'], axis=1)
y = data['SalePrice']
train_X, test_X, train_y, test_y = train_test_split(X, y)

my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

print(mean_absolute_error(test_y, predictions))