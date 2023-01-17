!pip install BorutaShap
import pandas  as pd



#===========================================================================

# read in the House Prices data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# select some features (These are all 'integer' fields for today).

#===========================================================================

features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 

            'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 

            '1stFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 

            'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr',  'Fireplaces', 

            'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 

            'EnclosedPorch',  'PoolArea', 'YrSold']



#===========================================================================

#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]
from BorutaShap import BorutaShap



# If no model is selected default is the Random Forest

# If classification is True it is a classification problem

Feature_Selector = BorutaShap(importance_measure='shap', classification=False)



Feature_Selector.fit(X=X_train, y=y_train, n_trials=50, random_state=0)
Feature_Selector.plot(which_features='all', figsize=(16,12))
# Return a subset of the original data with the selected features

Feature_Selector.Subset()