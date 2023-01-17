import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing  import Imputer
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
data= pd.read_csv('../input/house-prices-advanced-regression-techniques/housetrain.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y= data.SalePrice
X= data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
cols = ['LotArea','1stFlrSF','FullBath','TotRmsAbvGrd']
X=X[cols]
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix() , test_size=0.25)

myImputer = Imputer()
train_X = myImputer.fit_transform(train_X)
test_X = myImputer.transform(test_X)



from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
my_model =  GradientBoostingRegressor()

my_model.fit(train_X,train_y)

my_plots = plot_partial_dependence(my_model,       
                                   features=[0, 3], # column numbers of plots we want to show
                                   X=train_X,            # raw predictors data.
                                   feature_names=['LotArea','1stFlrSF','FullBath','TotRmsAbvGrd'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis

