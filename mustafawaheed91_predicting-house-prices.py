import numpy as np 

import pandas as pd 

import webbrowser

import os





from sklearn.model_selection import train_test_split

from sklearn import ensemble

from sklearn.metrics import mean_absolute_error

from sklearn.externals import joblib



%matplotlib inline
def show_in_browser(data_table):

    

    """Create a web page view of the data for easy viewing"""

    

    html = data_table.to_html()

    with open("data.html", "w") as f:

        f.write(html)

    full_filename = os.path.abspath("data.html")

    webbrowser.open("{}".format(full_filename))



    

def PercentageMissin(Dataset):

    

    """this function will return the percentage of missing values in a dataset """

    

    if isinstance(Dataset,pd.DataFrame):

        adict={} #a dictionary conatin keys columns names and values percentage of missin value in the columns

        for col in Dataset.columns:

            adict[col]=(np.count_nonzero(Dataset[col].isnull())*100)/len(Dataset[col])

        return pd.DataFrame(adict,index=['% of missing'],columns=adict.keys())

    else:

        raise TypeError("can only be used with panda dataframe")
df = pd.read_csv("../input/train.csv")

df.set_index(['Id'], inplace=True)
# For this model I am removing a lot of the features that I believe will add to the noise

# They are individually put here since it was through trial and error of checking values and context

del df['Alley']

del df['LotFrontage']

del df['PoolQC']

del df['Fence']

del df['MoSold']

del df['Neighborhood']

del df['Condition1']

del df['Condition2']

del df['LotConfig']

del df['MiscVal']

del df['HouseStyle']

del df['Fireplaces']

del df['FireplaceQu']

del df['KitchenQual']

del df['HeatingQC']

del df['3SsnPorch']

del df['GarageCond']

del df['GarageQual']

del df['GarageType']

del df['GarageYrBlt']

del df['GarageCars']

del df['MiscFeature']

del df['LandContour']

del df['LandSlope']

del df['Street']

del df['RoofStyle']

del df['YearRemodAdd']

del df['RoofMatl']

del df['Exterior1st']

del df['Exterior2nd']

del df['MasVnrType']

del df['ExterCond']

del df['ExterQual']

del df['BsmtQual']

del df['BsmtCond']

del df['BsmtExposure']

del df['BsmtFinType1']

del df['BsmtFinSF1']

del df['BsmtFinType2']

del df['BsmtFinSF2']

del df['BsmtUnfSF']

del df['BsmtHalfBath']

del df['TotRmsAbvGrd']

del df['Functional']

del df['PavedDrive']

del df['LowQualFinSF']

del df['GarageFinish']

del df['Heating']

del df['GrLivArea']

del df['OverallQual']

del df['WoodDeckSF']

del df['OpenPorchSF']

del df['EnclosedPorch']

del df['ScreenPorch']
# Impute Catagorical Variable with value of catagory with highest frequecy 

missing_ids = list(df.loc[ df['Electrical'].isnull() == True, ['Electrical'] ].index[:])

df['Electrical'].loc[ missing_ids , ] = str(df['Electrical'].value_counts().idxmax())



# Impute Continous numerical observations with median for that feature 

missing_ids = list(df.loc[ df['MasVnrArea'].isnull() == True, ['MasVnrArea'] ].index[:])

df['MasVnrArea'].loc[ missing_ids , ] = float(df['MasVnrArea'].median())



## Un-comment and run the following tocheck any other Nulls or NaN's

#df.isnull().any()
features_df = pd.get_dummies(df, columns=['MSZoning', 'LotShape', 'Utilities'

                                          , 'BldgType','Foundation', 'CentralAir'

                                          , 'Electrical', 'SaleType', 'SaleCondition'

                                         ])
del features_df['SalePrice']



## features that are not present in the test "features_df" that we will import below

del features_df['Utilities_NoSeWa'] 

del features_df['Electrical_Mix']
X = features_df.as_matrix()

y = df['SalePrice'].as_matrix()
from sklearn.model_selection import train_test_split



# Change test_size to 0 for training on complete model and then re-run program

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn import ensemble

from sklearn.metrics import mean_absolute_error

from sklearn.externals import joblib
model = ensemble.GradientBoostingRegressor(n_estimators=1000

                                           , learning_rate=0.1

                                           , max_depth=6

                                           , min_samples_leaf=9

                                           , max_features=0.1

                                           , loss='huber'

                                          )



model.fit(X_train, y_train)
## Uncomment the code below if trying to save a .pkl dump of the model to be reused later

#joblib.dump(model, 'trained_house_classifier_model.pkl')
mse = mean_absolute_error(y_train, model.predict(X_train))

print("Training Set Mean Absolute Error: $ %.4f"% mse)
mse = mean_absolute_error(y_test, model.predict(X_test))

print("Test Set Mean Absolute Error: $ %.4f"% mse)        
from sklearn.model_selection import GridSearchCV



param_grid = {

      'n_estimators'    :[500, 1000, 3000]

    , 'learning_rate'   :[1.0, 0.05, 0.02, 0.01]

    , 'max_depth'       :[4, 6]

    , 'min_samples_leaf':[3, 5, 9, 17]

    , 'max_features'    :[1.0, 0.3, 0.1]

    , 'loss'            :['lad', 'ls', 'huber']

}
gs_cv = GridSearchCV(model, param_grid=param_grid, n_jobs=8)
## Uncomment the code below to run since this process takes a very long time to complete

#gs_cv.fit(X_train, y_train)
## Uncomment Code to get the best parameter tunings for the model

#print(gs_cv.best_params_)
## Notice that we use the entire train set to train the model and remove some 

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.0, random_state=0)





model = ensemble.GradientBoostingRegressor(n_estimators=1000

                                           , learning_rate=0.1

                                           , max_depth=6

                                           , min_samples_leaf=9

                                           , max_features=0.1

                                           , loss='huber'

                                          )



model.fit(X_train, y_train)



mse = mean_absolute_error(y_train, model.predict(X_train))

print("Training Set Mean Absolute Error: $ %.4f"% mse)
df = pd.read_csv("../input/test.csv")

df.set_index(['Id'])
del df['Alley']

del df['LotFrontage']

del df['PoolQC']

del df['Fence']

del df['MoSold']

del df['Neighborhood']

del df['Condition1']

del df['Condition2']

del df['LotConfig']

del df['MiscVal']

del df['HouseStyle']

del df['Fireplaces']

del df['FireplaceQu']

del df['KitchenQual']

del df['HeatingQC']

del df['3SsnPorch']

del df['GarageCond']

del df['GarageQual']

del df['GarageType']

del df['GarageYrBlt']

del df['GarageCars']

del df['MiscFeature']

del df['LandContour']

del df['LandSlope']

del df['Street']

del df['RoofStyle']

del df['YearRemodAdd']

del df['RoofMatl']

del df['Exterior1st']

del df['Exterior2nd']

del df['MasVnrType']

del df['ExterCond']

del df['ExterQual']

del df['BsmtQual']

del df['BsmtCond']

del df['BsmtExposure']

del df['BsmtFinType1']

del df['BsmtFinSF1']

del df['BsmtFinType2']

del df['BsmtFinSF2']

del df['BsmtUnfSF']

del df['BsmtHalfBath']

del df['TotRmsAbvGrd']

del df['Functional']

del df['PavedDrive']

del df['LowQualFinSF']

del df['GarageFinish']

del df['Heating']

del df['GrLivArea']

del df['OverallQual']

del df['WoodDeckSF']

del df['OpenPorchSF']

del df['EnclosedPorch']

del df['ScreenPorch']
## Impute Catagorical Variable with value of catagory with highest frequecy 

missing_ids = list(df.loc[ df['Utilities'].isnull() == True, ['Utilities'] ].index[:])

df['Utilities'].loc[ missing_ids , ] = str(df['Utilities'].value_counts().idxmax())



missing_ids = list(df.loc[ df['MSZoning'].isnull() == True, ['MSZoning'] ].index[:])

df['MSZoning'].loc[ missing_ids , ] = str(df['MSZoning'].value_counts().idxmax())



missing_ids = list(df.loc[ df['SaleType'].isnull() == True, ['SaleType'] ].index[:])

df['SaleType'].loc[ missing_ids , ] = str(df['SaleType'].value_counts().idxmax())









## Impute Continous numerical observations with median for that feature 

missing_ids = list(df.loc[ df['MasVnrArea'].isnull() == True, ['MasVnrArea'] ].index[:])

df['MasVnrArea'].loc[ missing_ids , ] = float(df['MasVnrArea'].median())



missing_ids = list(df.loc[ df['TotalBsmtSF'].isnull() == True, ['TotalBsmtSF'] ].index[:])

df['TotalBsmtSF'].loc[ missing_ids , ] = float(df['TotalBsmtSF'].median())





missing_ids = list(df.loc[ df['BsmtFullBath'].isnull() == True, ['BsmtFullBath'] ].index[:])

df['BsmtFullBath'].loc[ missing_ids , ] = float(df['BsmtFullBath'].median())





missing_ids = list(df.loc[ df['GarageArea'].isnull() == True, ['GarageArea'] ].index[:])

df['GarageArea'].loc[ missing_ids , ] = float(df['GarageArea'].median())







## Un-comment and run the following tocheck any other Nulls or NaN's

#df.isnull().any()
features_df = pd.get_dummies(df, columns=['MSZoning', 'LotShape', 'Utilities'

                                          , 'BldgType','Foundation', 'CentralAir'

                                          , 'Electrical', 'SaleType', 'SaleCondition'

                                         ])
Id = features_df.loc[ : , 'Id' ]



del features_df['Id']



X_test = features_df.as_matrix()



SalePrice = pd.DataFrame(model.predict(X_test))

SalePrice = pd.Series( model.predict(X_test) )

output = pd.concat( [ Id, SalePrice.rename("SalePrice")] , axis=1)

output