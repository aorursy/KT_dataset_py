import pandas as pd 

train = pd.read_csv('../input/train.csv')
train_df = train.drop('SalePrice', axis=1)
test_df = pd.read_csv('../input/test.csv')
housing_labels = train['SalePrice']
data_full = pd.concat([train_df, test_df])

data_full.head()
data_NaN = data_full.isnull().sum()
data_NaN  = data_NaN[data_NaN>0]
data_NaN.sort_values(ascending=False)
data_full['PoolQC'].fillna(0, inplace=True)        
data_full['MiscFeature'].fillna(0, inplace=True)
data_full['Alley'].fillna(0, inplace=True)   
data_full['Fence'].fillna(0, inplace=True)         
data_full['FireplaceQu'].fillna(0, inplace=True)     
data_full['LotFrontage'].fillna(0, inplace=True)      
data_full['GarageFinish'].fillna(0, inplace=True)     
data_full['GarageYrBlt'].fillna(0, inplace=True)     
data_full['GarageQual'].fillna(0, inplace=True)       
data_full['GarageCond'].fillna(0, inplace=True)      
data_full['GarageType'].fillna(0, inplace=True)       
data_full['BsmtExposure'].fillna(0, inplace=True)     
data_full['BsmtCond'].fillna(0, inplace=True)         
data_full['BsmtQual'].fillna(0, inplace=True)          
data_full['BsmtFinType2'].fillna(0, inplace=True)     
data_full['BsmtFinType1'].fillna(0, inplace=True)      
data_full['MasVnrType'].fillna(0, inplace=True)        
data_full['MasVnrArea'].fillna(0, inplace=True)        
data_full['MSZoning'].fillna(0, inplace=True)           
data_full['BsmtFullBath'].fillna(0, inplace=True)       
data_full['BsmtHalfBath'].fillna(0, inplace=True)       
data_full['Utilities'].fillna(0, inplace=True)          
data_full['Functional'].fillna(0, inplace=True)         
data_full['Exterior2nd'].fillna(0, inplace=True)        
data_full['Exterior1st'].fillna(0, inplace=True)        
data_full['SaleType'].fillna(0, inplace=True)           
data_full['BsmtFinSF1'].fillna(0, inplace=True)         
data_full['BsmtFinSF2'].fillna(0, inplace=True)         
data_full['BsmtUnfSF'].fillna(0, inplace=True)          
data_full['Electrical'].fillna(0, inplace=True)         
data_full['KitchenQual'].fillna(0, inplace=True)        
data_full['GarageCars'].fillna(0, inplace=True)         
data_full['GarageArea'].fillna(0, inplace=True)         
data_full['TotalBsmtSF'].fillna(0, inplace=True)        

data_full.isnull().sum()
corr_matrix = train.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from warnings import filterwarnings
filterwarnings('ignore') #Use this to get rid of the DataConversion warning concerning converting int64 data to float64 data.

class DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

features = data_full[['LotArea', '1stFlrSF', '2ndFlrSF']]
features_selected = list(features)

pipeline = Pipeline([
    ('selected_features', DataSelector(features_selected)),
    ('scaler', StandardScaler())
])

housing_prepared = pipeline.fit_transform(train_df)
housing_prepared
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

housing_prepared_train, housing_prepared_test, housing_labels_train, housing_labels_test = train_test_split(housing_prepared, housing_labels, test_size=0.2, random_state=33)

#LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared_train, housing_labels_train)
lin_housing_predictions = lin_reg.predict(housing_prepared_test)
lin_mse = mean_squared_error(housing_labels_test, lin_housing_predictions)
lin_rmse = np.sqrt(lin_mse)

#DecisionTreeRegression
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared_train, housing_labels_train)
tree_reg_predictions = tree_reg.predict(housing_prepared_test)
tree_mse = mean_squared_error(housing_labels_test, tree_reg_predictions)
tree_rmse = np.sqrt(tree_mse)

#RandomForestRegression
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared_train, housing_labels_train)
forest_reg_predictions = forest_reg.predict(housing_prepared_test)
forest_mse = mean_squared_error(housing_labels_test, forest_reg_predictions)
forest_rmse = np.sqrt(forest_mse)

print(lin_rmse)
print(tree_rmse)
print(forest_rmse)
test_prepared = pipeline.fit_transform(test_df)

final_model = RandomForestRegressor()
final_model.fit(housing_prepared, housing_labels)
final_predictions = final_model.predict(test_prepared)

my_submission = pd.DataFrame({'Id': test_df.Id, 'SalePrice': final_predictions})
my_submission.head()

my_submission.to_csv('submission.csv', index=False)