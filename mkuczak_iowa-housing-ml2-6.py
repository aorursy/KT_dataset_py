import pandas as pd
data = pd.read_csv('../input/train.csv')
# Let's just choose a few values that should be significant to house cost. COMMENTED OUT DUE TO BAD SCORE
#cols_to_use = ['LotArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'YrSold']
#X = data[cols_to_use]
X = data.select_dtypes(include='number')
X = X.drop(['SalePrice'], axis=1)
y = data['SalePrice']
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())
from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error %2f' %(-1 * scores.mean()))