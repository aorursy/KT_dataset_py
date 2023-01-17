import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import Imputer
data = pd.read_csv('../input/train.csv')
data.head()
data = pd.read_csv('../input/train.csv')
# if no saleprice, drop the row
data.dropna(axis=0, subset=['SalePrice'], inplace=True)

cols_to_use = ['LotArea', 'OverallQual', 'YearBuilt']

y = data.SalePrice
X = data[cols_to_use]

my_imputer = Imputer()
X = my_imputer.fit_transform(X)
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model,
                                  features = [0, 1], # column numbers of plots we want to show
                                  X=X,
                                  feature_names = ['LotArea', 'OverallQual', 'YearBuilt'],#labels on graphs
                                  grid_resolution = 10) # no of values to plot on x axis
