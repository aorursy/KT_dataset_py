import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)

# Keep things simple.  Drop rows with NaN
data_NoNaN = data.dropna(axis=1)

X = data_NoNaN.drop(['SalePrice'], axis=1)
# We don't want to complicate things yet, so let's just stick to numeric data.
X = X.select_dtypes(include='number')
y = data_NoNaN['SalePrice']

print(X.head())
print(y.head())
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
# this was due to an implementation detail, and a future release will support all model types.
my_model = GradientBoostingRegressor()
# fit the model as usual
my_model.fit(X, y)
# THE ACTUAL PLOT IS CAUSING ME A LOT OF TROUBLE
# LOOK INTO THIS LATER FOR SURE
# MADE BIGGER TO NOT IGNORE!!!@@#!@$!@$!!%%
my_plots = plot_partial_dependence(my_model,       
                                   features=[0],
                                   X=X,
                                   feature_names=['LotArea'],
                                   grid_resolution=30) # number of values to plot on x axis
# Something is horribly wrong.  LotArea never dips as low as 250, and goes up to like 10000.
# Also, price should go up as Lot Area goes up.