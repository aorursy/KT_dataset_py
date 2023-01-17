import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error

# Load data
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
iowa_data = pd.read_csv(iowa_file_path) 
print(list(iowa_data))
cols_to_use = ['LotArea', 'BedroomAbvGr', 'PoolArea']


##Separating data in feature predictors, and target to predict
y = iowa_data.SalePrice
# Selecting 3 features to plot the partial dependence. 
X = iowa_data[cols_to_use]

### Generating training and test data
#train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.3)

## imputing NaN values
my_imputer = Imputer()
imputed_X = my_imputer.fit_transform(X)

### building XGBoost model 
from sklearn.ensemble import GradientBoostingRegressor

my_model = GradientBoostingRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y)

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
# Here we make the plot
my_plots = plot_partial_dependence(my_model,       
                                   features=[0,2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['LotArea', 'BedroomAbvGr', 'PoolArea'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
