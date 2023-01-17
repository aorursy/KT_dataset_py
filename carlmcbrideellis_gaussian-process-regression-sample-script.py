import pandas  as pd



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# select some features of interest ("ay, there's the rub", Shakespeare)

#===========================================================================

features = ['OverallQual' , 'GrLivArea' , 'TotalBsmtSF' , 'BsmtFinSF1' ,

            '2ndFlrSF'    , 'GarageArea', '1stFlrSF'    , 'YearBuilt'  ]



#===========================================================================

#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]

final_X_test  = test_data[features]



#===========================================================================

# pre-processing: imputation; substitute any 'NaN' with mean value

#===========================================================================

X_train      = X_train.fillna(X_train.mean())

final_X_test = final_X_test.fillna(final_X_test.mean())



#===========================================================================

# create the kernel 

#===========================================================================

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

kernel = DotProduct() + WhiteKernel()



#===========================================================================

# perform the regression 

#===========================================================================

from sklearn.gaussian_process import GaussianProcessRegressor

regressor = GaussianProcessRegressor(kernel=kernel)



#===========================================================================

# and the fit 

#===========================================================================

regressor.fit(X_train, y_train)



#===========================================================================

# now use the model to predict the prices for the test data

#===========================================================================

predictions = regressor.predict(final_X_test)



#===========================================================================

# and finally write out CSV submission file

#===========================================================================

output = pd.DataFrame({"Id":test_data.Id, "SalePrice":predictions})

output.to_csv('submission.csv', index=False)