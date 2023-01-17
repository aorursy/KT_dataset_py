import pandas as pd 
from sklearn.preprocessing  import Imputer
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
data= pd.read_csv('../input/housetrain.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y= data.SalePrice
X= data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
cols = ['LotArea','1stFlrSF','FullBath','TotRmsAbvGrd']
X=X[cols]
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix() , test_size=0.25)
#Building pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())


#predictions
from sklearn.metrics import mean_absolute_error

my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)
mean_absolute_error(test_y,predictions)
