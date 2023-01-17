import pandas as pd

data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
my_model = RandomForestRegressor()
my_imputer = Imputer()
my_pipeline = make_pipeline(my_imputer, my_model)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error %2f' %(-1 * scores.mean()))

my_pipeline.fit(X,y)
##Submission

# Read the test data
test = pd.read_csv('../input/test.csv')
test_X = test.select_dtypes(exclude=['object'])

# Use the model to make predictions
predicted_prices = my_pipeline.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)