!pip install rgf_python
import pandas  as pd

import numpy   as np



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/titanic/train.csv')

test_data  = pd.read_csv('../input/titanic/test.csv')

solution   = pd.read_csv('../input/submission-solution/submission_solution.csv')



#===========================================================================

# select some features

#===========================================================================

features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]



#===========================================================================

# for the features that are categorical we use pd.get_dummies:

# "Convert categorical variable into dummy/indicator variables."

#===========================================================================

X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]

final_X_test  = pd.get_dummies(test_data[features])



#===========================================================================

# perform the classification 

#===========================================================================

from rgf.sklearn import RGFClassifier



classifier = RGFClassifier(max_leaf=300, algorithm="RGF_Sib", test_interval=100)



#===========================================================================

# and the fit 

#===========================================================================

classifier.fit(X_train, y_train)



#===========================================================================

# use the model to predict 'Survived' for the test data

#===========================================================================

predictions = classifier.predict(final_X_test)



#===========================================================================

# now calculate our score

#===========================================================================

from sklearn.metrics import accuracy_score

print("The score is %.5f" % accuracy_score( solution['Survived'] , predictions ) )
#===========================================================================

# read in the competition data 

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

solution   = pd.read_csv('../input/house-prices-advanced-regression-solution-file/solution.csv')

                         

#===========================================================================

# select some features

#===========================================================================

features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 

        'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 

        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 

        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 

        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 

        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']



#===========================================================================

#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]

final_X_test  = test_data[features]

y_true        = solution["SalePrice"]



#===========================================================================

# essential preprocessing: imputation; substitute any 'NaN' with mean value

#===========================================================================

X_train      = X_train.fillna(X_train.mean())

final_X_test = final_X_test.fillna(final_X_test.mean())



#===========================================================================

# perform the regression

#===========================================================================

from rgf.sklearn import RGFRegressor



regressor = RGFRegressor(max_leaf=300, algorithm="RGF_Sib", test_interval=100, loss="LS")



#===========================================================================

# and the fit 

#===========================================================================

regressor.fit(X_train, y_train)



#===========================================================================

# use the model to predict the prices for the test data

#===========================================================================

y_pred = regressor.predict(final_X_test)



#===========================================================================

# compare your predictions with the 'solution' using the 

# root of the mean_squared_log_error

#===========================================================================

from sklearn.metrics import mean_squared_log_error

RMSLE = np.sqrt( mean_squared_log_error(y_true, y_pred) )

print("The score is %.5f" % RMSLE )