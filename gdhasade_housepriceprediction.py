#Dataset link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
#pip install pandas
#pip install numpy
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
TrainDataSet = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
TestDataSet = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



TrainDataSet
TrainDataSet.shape

TestDataSet.shape
TrainDataSet.dtypes
TestDataSet.dtypes
cols = TrainDataSet.columns

cols
TrainDataSet.isnull().sum()

TestDataSet.isnull().sum()
#Replace null values of each columns

TrainDataSet['MSSubClass'].fillna((TrainDataSet['MSSubClass'].mean()), inplace=True)
TrainDataSet['MSZoning'].fillna("Unknown", inplace=True)
TrainDataSet['LotFrontage'].fillna((TrainDataSet['LotFrontage'].mean()), inplace=True)
TrainDataSet['LotArea'].fillna((TrainDataSet['LotArea'].mean()), inplace=True)
TrainDataSet['Street'].fillna("Unknown", inplace=True)
TrainDataSet['Alley'].fillna("Unknown", inplace=True)
TrainDataSet['LotShape'].fillna("Unknown", inplace=True)
TrainDataSet['LandContour'].fillna("Unknown", inplace=True)
TrainDataSet['Utilities'].fillna("Unknown", inplace=True)
TrainDataSet['LotConfig'].fillna("Unknown", inplace=True)
TrainDataSet['LandSlope'].fillna("Unknown", inplace=True)
TrainDataSet['Neighborhood'].fillna("Unknown", inplace=True)
TrainDataSet['Condition1'].fillna("Unknown", inplace=True)
TrainDataSet['Condition2'].fillna("Unknown", inplace=True)
TrainDataSet['BldgType'].fillna("Unknown", inplace=True)
TrainDataSet['HouseStyle'].fillna("Unknown", inplace=True)
TrainDataSet['OverallQual'].fillna((TrainDataSet['OverallQual'].mean()), inplace=True)
TrainDataSet['OverallCond'].fillna((TrainDataSet['OverallCond'].mean()), inplace=True)
TrainDataSet['RoofStyle'].fillna("Unknown", inplace=True)
TrainDataSet['RoofMatl'].fillna("Unknown", inplace=True)
TrainDataSet['Exterior1st'].fillna("Unknown", inplace=True)
TrainDataSet['Exterior2nd'].fillna("Unknown", inplace=True)
TrainDataSet['MasVnrType'].fillna("Unknown", inplace=True)
TrainDataSet['MasVnrArea'].fillna((TrainDataSet['MasVnrArea'].mean()), inplace=True)
TrainDataSet['ExterQual'].fillna("Unknown", inplace=True)
TrainDataSet['ExterCond'].fillna("Unknown", inplace=True)
TrainDataSet['Foundation'].fillna("Unknown", inplace=True)
TrainDataSet['BsmtQual'].fillna("Unknown", inplace=True)
TrainDataSet['BsmtCond'].fillna("Unknown", inplace=True)
TrainDataSet['BsmtExposure'].fillna("Unknown", inplace=True)
TrainDataSet['BsmtFinType1'].fillna("Unknown", inplace=True)
TrainDataSet['BsmtFinSF1'].fillna((TrainDataSet['BsmtFinSF1'].mean()), inplace=True)
TrainDataSet['BsmtFinType2'].fillna("Unknown", inplace=True)
TrainDataSet['BsmtFinSF2'].fillna((TrainDataSet['BsmtFinSF2'].mean()), inplace=True)
TrainDataSet['BsmtUnfSF'].fillna((TrainDataSet['BsmtUnfSF'].mean()), inplace=True)
TrainDataSet['TotalBsmtSF'].fillna((TrainDataSet['TotalBsmtSF'].mean()), inplace=True)
TrainDataSet['Heating'].fillna("Unknown", inplace=True)
TrainDataSet['HeatingQC'].fillna("Unknown", inplace=True)
TrainDataSet['CentralAir'].fillna("Unknown", inplace=True)
TrainDataSet['Electrical'].fillna("Unknown", inplace=True)
TrainDataSet['1stFlrSF'].fillna((TrainDataSet['1stFlrSF'].mean()), inplace=True)
TrainDataSet['2ndFlrSF'].fillna((TrainDataSet['2ndFlrSF'].mean()), inplace=True)
TrainDataSet['LowQualFinSF'].fillna((TrainDataSet['LowQualFinSF'].mean()), inplace=True)
TrainDataSet['GrLivArea'].fillna((TrainDataSet['GrLivArea'].mean()), inplace=True)
TrainDataSet['BsmtFullBath'].fillna((TrainDataSet['BsmtFullBath'].mean()), inplace=True)
TrainDataSet['BsmtHalfBath'].fillna((TrainDataSet['BsmtHalfBath'].mean()), inplace=True)
TrainDataSet['FullBath'].fillna((TrainDataSet['FullBath'].mean()), inplace=True)
TrainDataSet['HalfBath'].fillna((TrainDataSet['HalfBath'].mean()), inplace=True)
TrainDataSet['BedroomAbvGr'].fillna((TrainDataSet['BedroomAbvGr'].mean()), inplace=True)
TrainDataSet['KitchenAbvGr'].fillna((TrainDataSet['KitchenAbvGr'].mean()), inplace=True)
TrainDataSet['KitchenQual'].fillna("Unknown", inplace=True)
TrainDataSet['TotRmsAbvGrd'].fillna((TrainDataSet['TotRmsAbvGrd'].mean()), inplace=True)
TrainDataSet['Functional'].fillna("Unknown", inplace=True)
TrainDataSet['Fireplaces'].fillna((TrainDataSet['Fireplaces'].mean()), inplace=True)
TrainDataSet['FireplaceQu'].fillna("Unknown", inplace=True)
TrainDataSet['GarageType'].fillna("Unknown", inplace=True)
TrainDataSet['GarageYrBlt'].fillna((TrainDataSet['GarageYrBlt'].mean()), inplace=True)
TrainDataSet['GarageFinish'].fillna("Unknown", inplace=True)
TrainDataSet['GarageCars'].fillna((TrainDataSet['GarageCars'].mean()), inplace=True)
TrainDataSet['GarageArea'].fillna((TrainDataSet['GarageArea'].mean()), inplace=True)
TrainDataSet['GarageQual'].fillna("Unknown", inplace=True)
TrainDataSet['GarageCond'].fillna("Unknown", inplace=True)
TrainDataSet['PavedDrive'].fillna("Unknown", inplace=True)
TrainDataSet['WoodDeckSF'].fillna((TrainDataSet['WoodDeckSF'].mean()), inplace=True)
TrainDataSet['OpenPorchSF'].fillna((TrainDataSet['OpenPorchSF'].mean()), inplace=True)
TrainDataSet['EnclosedPorch'].fillna((TrainDataSet['EnclosedPorch'].mean()), inplace=True)
TrainDataSet['3SsnPorch'].fillna((TrainDataSet['3SsnPorch'].mean()), inplace=True)
TrainDataSet['ScreenPorch'].fillna((TrainDataSet['ScreenPorch'].mean()), inplace=True)
TrainDataSet['PoolArea'].fillna((TrainDataSet['PoolArea'].mean()), inplace=True)
TrainDataSet['PoolQC'].fillna("Unknown", inplace=True)
TrainDataSet['Fence'].fillna("Unknown", inplace=True)
TrainDataSet['MiscFeature'].fillna("Unknown", inplace=True)
TrainDataSet['MiscVal'].fillna((TrainDataSet['MiscVal'].mean()), inplace=True)
TrainDataSet['SaleType'].fillna("Unknown", inplace=True)
TrainDataSet['SaleCondition'].fillna("Unknown", inplace=True)
TrainDataSet['SalePrice'].fillna((TrainDataSet['SalePrice'].mean()), inplace=True)
#Replace null values in test dataset
TestDataSet['MSSubClass'].fillna((TestDataSet['MSSubClass'].mean()), inplace=True)
TestDataSet['MSZoning'].fillna("Unknown", inplace=True)
TestDataSet['LotFrontage'].fillna((TestDataSet['LotFrontage'].mean()), inplace=True)
TestDataSet['LotArea'].fillna((TestDataSet['LotArea'].mean()), inplace=True)
TestDataSet['Street'].fillna("Unknown", inplace=True)
TestDataSet['Alley'].fillna("Unknown", inplace=True)
TestDataSet['LotShape'].fillna("Unknown", inplace=True)
TestDataSet['LandContour'].fillna("Unknown", inplace=True)
TestDataSet['Utilities'].fillna("Unknown", inplace=True)
TestDataSet['LotConfig'].fillna("Unknown", inplace=True)
TestDataSet['LandSlope'].fillna("Unknown", inplace=True)
TestDataSet['Neighborhood'].fillna("Unknown", inplace=True)
TestDataSet['Condition1'].fillna("Unknown", inplace=True)
TestDataSet['Condition2'].fillna("Unknown", inplace=True)
TestDataSet['BldgType'].fillna("Unknown", inplace=True)
TestDataSet['HouseStyle'].fillna("Unknown", inplace=True)
TestDataSet['OverallQual'].fillna((TestDataSet['OverallQual'].mean()), inplace=True)
TestDataSet['OverallCond'].fillna((TestDataSet['OverallCond'].mean()), inplace=True)
TestDataSet['RoofStyle'].fillna("Unknown", inplace=True)
TestDataSet['RoofMatl'].fillna("Unknown", inplace=True)
TestDataSet['Exterior1st'].fillna("Unknown", inplace=True)
TestDataSet['Exterior2nd'].fillna("Unknown", inplace=True)
TestDataSet['MasVnrType'].fillna("Unknown", inplace=True)
TestDataSet['MasVnrArea'].fillna((TestDataSet['MasVnrArea'].mean()), inplace=True)
TestDataSet['ExterQual'].fillna("Unknown", inplace=True)
TestDataSet['ExterCond'].fillna("Unknown", inplace=True)
TestDataSet['Foundation'].fillna("Unknown", inplace=True)
TestDataSet['BsmtQual'].fillna("Unknown", inplace=True)
TestDataSet['BsmtCond'].fillna("Unknown", inplace=True)
TestDataSet['BsmtExposure'].fillna("Unknown", inplace=True)
TestDataSet['BsmtFinType1'].fillna("Unknown", inplace=True)
TestDataSet['BsmtFinSF1'].fillna((TestDataSet['BsmtFinSF1'].mean()), inplace=True)
TestDataSet['BsmtFinType2'].fillna("Unknown", inplace=True)
TestDataSet['BsmtFinSF2'].fillna((TestDataSet['BsmtFinSF2'].mean()), inplace=True)
TestDataSet['BsmtUnfSF'].fillna((TestDataSet['BsmtUnfSF'].mean()), inplace=True)
TestDataSet['TotalBsmtSF'].fillna((TestDataSet['TotalBsmtSF'].mean()), inplace=True)
TestDataSet['Heating'].fillna("Unknown", inplace=True)
TestDataSet['HeatingQC'].fillna("Unknown", inplace=True)
TestDataSet['CentralAir'].fillna("Unknown", inplace=True)
TestDataSet['Electrical'].fillna("Unknown", inplace=True)
TestDataSet['1stFlrSF'].fillna((TestDataSet['1stFlrSF'].mean()), inplace=True)
TestDataSet['2ndFlrSF'].fillna((TestDataSet['2ndFlrSF'].mean()), inplace=True)
TestDataSet['LowQualFinSF'].fillna((TestDataSet['LowQualFinSF'].mean()), inplace=True)
TestDataSet['GrLivArea'].fillna((TestDataSet['GrLivArea'].mean()), inplace=True)
TestDataSet['BsmtFullBath'].fillna((TestDataSet['BsmtFullBath'].mean()), inplace=True)
TestDataSet['BsmtHalfBath'].fillna((TestDataSet['BsmtHalfBath'].mean()), inplace=True)
TestDataSet['FullBath'].fillna((TestDataSet['FullBath'].mean()), inplace=True)
TestDataSet['HalfBath'].fillna((TestDataSet['HalfBath'].mean()), inplace=True)
TestDataSet['BedroomAbvGr'].fillna((TestDataSet['BedroomAbvGr'].mean()), inplace=True)
TestDataSet['KitchenAbvGr'].fillna((TestDataSet['KitchenAbvGr'].mean()), inplace=True)
TestDataSet['KitchenQual'].fillna("Unknown", inplace=True)
TestDataSet['TotRmsAbvGrd'].fillna((TestDataSet['TotRmsAbvGrd'].mean()), inplace=True)
TestDataSet['Functional'].fillna("Unknown", inplace=True)
TestDataSet['Fireplaces'].fillna((TestDataSet['Fireplaces'].mean()), inplace=True)
TestDataSet['FireplaceQu'].fillna("Unknown", inplace=True)
TestDataSet['GarageType'].fillna("Unknown", inplace=True)
TestDataSet['GarageYrBlt'].fillna((TestDataSet['GarageYrBlt'].mean()), inplace=True)
TestDataSet['GarageFinish'].fillna("Unknown", inplace=True)
TestDataSet['GarageCars'].fillna((TestDataSet['GarageCars'].mean()), inplace=True)
TestDataSet['GarageArea'].fillna((TestDataSet['GarageArea'].mean()), inplace=True)
TestDataSet['GarageQual'].fillna("Unknown", inplace=True)
TestDataSet['GarageCond'].fillna("Unknown", inplace=True)
TestDataSet['PavedDrive'].fillna("Unknown", inplace=True)
TestDataSet['WoodDeckSF'].fillna((TestDataSet['WoodDeckSF'].mean()), inplace=True)
TestDataSet['OpenPorchSF'].fillna((TestDataSet['OpenPorchSF'].mean()), inplace=True)
TestDataSet['EnclosedPorch'].fillna((TestDataSet['EnclosedPorch'].mean()), inplace=True)
TestDataSet['3SsnPorch'].fillna((TestDataSet['3SsnPorch'].mean()), inplace=True)
TestDataSet['ScreenPorch'].fillna((TestDataSet['ScreenPorch'].mean()), inplace=True)
TestDataSet['PoolArea'].fillna((TestDataSet['PoolArea'].mean()), inplace=True)
TestDataSet['PoolQC'].fillna("Unknown", inplace=True)
TestDataSet['Fence'].fillna("Unknown", inplace=True)
TestDataSet['MiscFeature'].fillna("Unknown", inplace=True)
TestDataSet['MiscVal'].fillna((TestDataSet['MiscVal'].mean()), inplace=True)
TestDataSet['SaleType'].fillna("Unknown", inplace=True)
TestDataSet['SaleCondition'].fillna("Unknown", inplace=True)

#Check null values for date colums to calculate how old property is
TrainDataSet[['YearBuilt', 'YearRemodAdd','MoSold', 'YrSold','GarageYrBlt']].isnull().sum()
TestDataSet[['YearBuilt', 'YearRemodAdd','MoSold', 'YrSold','GarageYrBlt']].isnull().sum()
TrainDataSet[['YearBuilt', 'YearRemodAdd','MoSold', 'YrSold','GarageYrBlt']]
#Replace years date with number in TrainDataSet
#Add col to dataset as current year
TrainDataSet['Current Year']=2020
#Calculate how old property use YearBuilt col--> Property_Age
TrainDataSet['Property_Age']=TrainDataSet['Current Year']- TrainDataSet['YearBuilt']

#Calculate last renovated use YearRemodAdd--> Last_Renovation
TrainDataSet['Last_Renovation']=TrainDataSet['Current Year']- TrainDataSet['YearRemodAdd']

#Last sold time use Sold_Date col --> Last_Sold
TrainDataSet['Last_Sold']=TrainDataSet['Current Year']- TrainDataSet['YrSold']

#Garage age use GarageYrBlt --> Garage_Age
TrainDataSet['Garage_Age']=TrainDataSet['Current Year']- TrainDataSet['GarageYrBlt']
#Replace years date with number in TestDataSet
#Add col to dataset as current year
TestDataSet['Current Year']=2020
#Calculate how old property use YearBuilt col--> Property_Age
TestDataSet['Property_Age']=TestDataSet['Current Year']- TrainDataSet['YearBuilt']

#Calculate last renovated use YearRemodAdd--> Last_Renovation
TestDataSet['Last_Renovation']=TestDataSet['Current Year']- TestDataSet['YearRemodAdd']

#Last sold time use Sold_Date col --> Last_Sold
TestDataSet['Last_Sold']=TestDataSet['Current Year']- TestDataSet['YrSold']

#Garage age use GarageYrBlt --> Garage_Age
TestDataSet['Garage_Age']=TestDataSet['Current Year']- TestDataSet['GarageYrBlt']
TestDataSet[['Property_Age','Last_Renovation','Last_Sold','Garage_Age']]
#Now Drop those colums (YearBuilt, YearRemodAdd, YrSold, MoSold)
TrainDataSet.drop(['YearBuilt','YearRemodAdd','YrSold', 'MoSold','GarageYrBlt'],axis=1,inplace=True)
TestDataSet.drop(['YearBuilt','YearRemodAdd','YrSold', 'MoSold','GarageYrBlt'],axis=1,inplace=True)
TrainDataSet.shape
TestDataSet.shape
#Now create Dummies for categorical columns
TrainDataSet=pd.get_dummies(TrainDataSet,drop_first=True)
TestDataSet=pd.get_dummies(TestDataSet,drop_first=True)
TrainDataSet.shape 
TestDataSet.shape 
TrainDataSet.columns
TestDataSet.columns
#Let's check the correlation between columns
#TrainDataSet.corr()
#pip install sklearn
#Take dependent vairable in y
y = TrainDataSet['SalePrice']
X = TrainDataSet
X = X.drop(['SalePrice','Id'],axis=1)
TrainDataSet.shape
X.shape
y.shape
#------------------Feature Selection---------------------------
#-----ensemble- ExtraTreeRegressor algo----
### Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feat_importances.nlargest(30))
ETR_IV= X[['ExterQual_TA','OverallQual','GarageCars','GrLivArea','1stFlrSF','FireplaceQu_Unknown','FullBath','GarageArea','2ndFlrSF','TotalBsmtSF','BsmtFinSF1','TotRmsAbvGrd','KitchenQual_Gd','Neighborhood_NoRidge','LotArea','Last_Renovation','Fireplaces','BsmtQual_Gd','BsmtFullBath','KitchenQual_TA','GarageType_Attchd','BsmtExposure_Gd','BedroomAbvGr','Property_Age','CentralAir_Y','ExterQual_Gd','MSZoning_RM','MasVnrArea','GarageType_Detchd','SaleType_WD']]
ETR_DV = TrainDataSet['SalePrice']

ETR_TestDataSet= TestDataSet[['ExterQual_TA','OverallQual','GarageCars','GrLivArea','1stFlrSF','FireplaceQu_Unknown','FullBath','GarageArea','2ndFlrSF','TotalBsmtSF','BsmtFinSF1','TotRmsAbvGrd','KitchenQual_Gd','Neighborhood_NoRidge','LotArea','Last_Renovation','Fireplaces','BsmtQual_Gd','BsmtFullBath','KitchenQual_TA','GarageType_Attchd','BsmtExposure_Gd','BedroomAbvGr','Property_Age','CentralAir_Y','ExterQual_Gd','MSZoning_RM','MasVnrArea','GarageType_Detchd','SaleType_WD']]


#-----install xgboost--- and use to predict the feature
#conda install -c anaconda py-xgboost
#Take almost 15 mins
from xgboost import XGBClassifier
# Create an object instance for XGBClassifier class
model = XGBClassifier()
# Fit/Train the model
model.fit(X,y)
# Get the Feature Importance Score for all the features between 0 and 1
col = pd.DataFrame(X.columns, columns = ['col'])
feature = pd.DataFrame(model.feature_importances_,columns = ['feature'])
Feature_Importance = pd.concat([col,feature],axis=1)
Feature_Importance.sort_values(by=['feature'], inplace=True, ascending = False)
Sorted_Features = Feature_Importance[Feature_Importance['feature'] > 0.008] 
Sorted_Features['col']
#Feature_Importance['feature']
Features = Sorted_Features['col'].tolist()


Features.remove('GarageQual_Fa')
Features

Features.remove('RoofMatl_CompShg')
TestDataSet.columns
XGB_IV= X[Features]
XGB_DV = TrainDataSet['SalePrice']
XGB_TestDataSet= TestDataSet[Features]
#--------------Checking Training Accuracy with above extracted features -----------
#--------------1. Creating models for each feature extracted algorithms ----------
from sklearn.model_selection import train_test_split 
# Spliting the dataset as 80% Training & 20% Testing data using sklearn selection train_test_split function
#X_ETRTrain, X_ETRTest, Y_ETRTrain, Y_ETRTest = train_test_split(ETR_IV, ETR_DV, test_size = 0.20, random_state = 0)
#X_XGBTrain, X_XGBTest, Y_XGBTrain, Y_XGBTest = train_test_split(XGB_IV, XGB_DV, test_size = 0.20, random_state = 0)

#--------------2. Now train those created models -----------------
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


rf_randomXGB = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(ETR_IV, ETR_DV)
rf_randomXGB.fit(XGB_IV, XGB_DV)
#Computing accuracy score of training Model-----------------
rf_random.best_params_
rf_randomXGB.best_params_
rf_random.best_score_
rf_randomXGB.best_score_
predictionsETR=rf_random.predict(ETR_TestDataSet)
predictionsXGB=rf_randomXGB.predict(XGB_TestDataSet)
predictionsETR
predictionsXGB
FinalResult = pd.DataFrame({"Id":TestDataSet.Id, "SalePrice":predictionsETR})
#FinalResult.to_csv('../output/kaggle/working/mysubmission2.csv', index=False)
#--------------SVM--------
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(ETR_IV, ETR_DV)
SVC = svclassifier.predict(TestDataSet)
