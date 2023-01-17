import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

### import trainint set 
house_data = pd.read_csv('../input/train.csv')

## read data from train.csv

house_data.columns
## show columns
#print(house_data.describe())
y = house_data.SalePrice
house_predictors = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'] 
X = house_data[house_predictors]
house_model = DecisionTreeRegressor()
house_model.fit(X,y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(house_model.predict(X.head()))
##house_data['SalePrice'].describe()

print(house_data['SalePrice'].head(8))