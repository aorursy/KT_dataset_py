import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
full_train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
full_train.shape
test.shape
# First, let's explore some data
full_train.describe(include=[object])
# How about corellation?
corr = full_train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)
# Let's build a heatmap for clarity
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(corr, vmax=.8, annot=True);
# Wow! That was big! It is better to find the most correlated features
most_corr_features = corr.index[abs(corr["SalePrice"])>0.6]
plt.figure(figsize=(15,15))
sns.heatmap(full_train[most_corr_features].corr(),annot=True)
# Let's explore the train and test data together so we get a more holistic picture
full_train = full_train.append(test, ignore_index=True)
full_train.drop('Id', axis=1, inplace=True)
# NaN values are always bad. Maybe we'll get lucky and they won't be in our data.
full_train.isnull().sum()
# Well, we're going to test a lot of models. Some models cope with missing values, some not.

# To keep things simple, I filled in the missing values without going into too much detail. 
# In other kernels, you can see how to do it better than me.

full_train["LotFrontage"].fillna(0.0, inplace=True)
full_train['PoolQC'].fillna('No', inplace=True)
full_train['Alley'].fillna('No', inplace=True)
full_train['BsmtCond'].fillna('No', inplace=True)
full_train['BsmtExposure'].fillna('No', inplace=True)
full_train['BsmtFinSF1'].fillna(0, inplace=True)
full_train['BsmtFinSF2'].fillna(0.0, inplace=True)
full_train['BsmtFinType1'].fillna('No', inplace=True)
full_train['BsmtFinType2'].fillna('No', inplace=True)
full_train['BsmtFullBath'].fillna(0, inplace=True)
full_train['BsmtHalfBath'].fillna(0, inplace=True)
full_train['BsmtQual'].fillna('No', inplace=True)
full_train['BsmtUnfSF'].fillna(0, inplace=True)
full_train['Electrical'].fillna('No', inplace=True)
full_train['Exterior1st'].fillna('No', inplace=True)
full_train['Exterior2nd'].fillna('No', inplace=True)
full_train['Fence'].fillna('No', inplace=True)
full_train['FireplaceQu'].fillna('No', inplace=True)
full_train['MSZoning'].fillna(full_train['MSZoning'].mode()[0], inplace=True)
full_train['MasVnrArea'].fillna(0, inplace=True)
full_train['MasVnrType'].fillna('No', inplace=True)
full_train['MiscFeature'].fillna('No', inplace=True)
full_train['SaleType'].fillna(full_train['SaleType'].mode()[0], inplace=True)
full_train['Utilities'].fillna(full_train['Utilities'].mode()[0], inplace=True)
full_train['TotalBsmtSF'].fillna(0, inplace=True)
full_train['Functional'].fillna('Typ', inplace=True)
full_train['KitchenQual'].fillna(full_train['KitchenQual'].mode()[0], inplace=True)
full_train['GarageArea'].fillna(0, inplace=True)
full_train['GarageCars'].fillna(0, inplace=True)
full_train['GarageYrBlt'].fillna(0, inplace=True)
full_train['GarageType'].fillna('No', inplace=True)
full_train['GarageQual'].fillna('No', inplace=True)
full_train['GarageFinish'].fillna('No', inplace=True)
full_train['GarageCond'].fillna('No', inplace=True)
# Are we done with NaN?
full_train.columns[full_train.isnull().any()].tolist()
y_train = full_train[:1460].SalePrice
full_train.drop('SalePrice', axis=1, inplace=True)
# Some models can only work with numeric values. What can we do with, for example, RoofStyle? 
# We can convert it to a number! And pd.get_dummies will help us.

non_numeric_predictors = full_train.select_dtypes(include=['object']).columns
full_train = pd.get_dummies(full_train)
full_train.head()
# When one parameter is 1000 and the other is 30, some models start to go crazy.
# We will make the values near zero, this is called normalization

mean = full_train.mean(axis=0)
std = full_train.std(axis=0)
full_train -= mean
full_train = full_train/std
full_train.head()
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)

y_train -= y_mean
y_train = y_train/y_std
X_train = full_train[:1460]
X_test = full_train[1460:]
# How do we test our models without making a submission every time? We can 
# just split our train set and test models on it

from sklearn.model_selection import train_test_split

X_train_valid, X_test_valid, y_train_valid, y_test_valid = train_test_split(
    X_train, y_train, random_state=42, shuffle=True)
from sklearn.dummy import DummyRegressor

#  I always use Dummy model like a benchmark. It's our starting point, which we will improve

dummy_majority = DummyRegressor(strategy = 'median')
dummy_majority.fit(X_train_valid, y_train_valid)

print("score on train: {}".format(dummy_majority.score(X_train_valid, y_train_valid)))
print("score on test: {}".format(dummy_majority.score(X_test_valid, y_test_valid)))
from sklearn.neighbors import KNeighborsRegressor

for neighbors in range(1,40):
    knn = KNeighborsRegressor(n_neighbors = neighbors)
    knn.fit(X_train_valid, y_train_valid)
    
    print("================================")
    print("neighbors number: {}".format(neighbors))
    print("model score on train: {}".format( knn.score(X_train_valid, y_train_valid)))
    print("model score on test: {}".format( knn.score(X_test_valid, y_test_valid)))
    print("================================")

print('Done fitting!')
# As we see, with 12 neighbors our model get the best test score
from sklearn.model_selection import GridSearchCV

# Now let's use another interesting thing - GridSearch. 
# This example is the same as above, but with a slight difference - 
# GridSearch use cross validation inside, what's why we fit it by X_train, not X_train_valid.
# Also, best scores show us n_neighbors = 7, not 12 as above. Interesting!
# I think it's because of the cross validation we didn't use in the last example.

knn = KNeighborsRegressor()

tuned_parameters = {'n_neighbors': range(1,40)}

grid_knn = GridSearchCV(cv=5, estimator=knn, param_grid=tuned_parameters, verbose=2)

grid_knn.fit(X_train, y_train)
print("best_scores:{}".format(grid_knn.best_params_))
print("================")
print("grid scores:{}".format(grid_knn.grid_scores_))
from sklearn.linear_model import Ridge

for alpha_sample in range(50, 100):
    linridge = Ridge(alpha=alpha_sample).fit(X_train_valid, y_train_valid)
    print("================")
    print("Alpha is: {}".format(alpha_sample))
    print("score on train: {}".format(linridge.score(X_train_valid, y_train_valid)))
    print("score on test: {}".format(linridge.score(X_test_valid, y_test_valid)))
    print("================")
    
print("Done fitting!")
from sklearn.svm import SVR

for C_sample in [0.001, 0.01, 0.1, 1]:

    svr = SVR(kernel = 'linear', C=C_sample).fit(X_train_valid, y_train_valid)   
    print("================================")
    print("C is : {}".format(C_sample))
    print("model score on train: {}".format( svr.score(X_train_valid, y_train_valid)))
    print("model score on test: {}".format( svr.score(X_test_valid, y_test_valid)))
    print("================================")
    
print("Done fitting!")
from sklearn.svm import LinearSVR

for C_sample in [0.001, 0.01, 0.1, 1, 10]:
    print("================================")
    print("C is : {}".format(C_sample))
    svc = LinearSVR(C_sample).fit(X_train_valid, y_train_valid)
    print("score on train: {}".format(svc.score(X_train_valid, y_train_valid)))
    print("score on test: {}".format(svc.score(X_test_valid, y_test_valid)))
    print("================================")
    
print("Done fitting!")
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressor.fit(X_train_valid, y_train_valid)

print("================================")
print("score on train:{}".format(regressor.score(X_train_valid, y_train_valid)))
print("score on test:{}".format(regressor.score(X_test_valid, y_test_valid)))
print("================================")
from sklearn.ensemble import RandomForestRegressor

for number in range(200, 500, 50):
    model_random_forest = RandomForestRegressor(n_estimators=number, n_jobs=-1,
                                            random_state = 42)
    model_random_forest.fit(X_train_valid, y_train_valid)
    
    print("===============")
    print("Trees number: {}".format(number))
    print("score on train: {}".format(model_random_forest.score(X_train_valid, y_train_valid)))
    print("score on test: {}".format(model_random_forest.score(X_test_valid, y_test_valid)))
    print("===============")

print("Done fitting!")
from sklearn.ensemble import GradientBoostingRegressor

for trees in range(200, 400, 100):
    model_gradient_boosting = GradientBoostingRegressor(n_estimators=trees, 
                                                        criterion="mae", random_state=42)
    
    model_gradient_boosting.fit(X_train_valid, y_train_valid)

    print("===============")
    print("trees_number: {}".format(trees))
    print("score on train: {}".format(model_gradient_boosting.score(X_train_valid, y_train_valid)))
    print("score on test: {}".format(model_gradient_boosting.score(X_test_valid, y_test_valid)))
    print("===============")

print("Done fitting!")
from sklearn.neural_network import MLPRegressor

mlpreg = MLPRegressor(hidden_layer_sizes = [100,100],
                             activation = 'relu',
                             alpha = 1.0,
                             solver = 'lbfgs', verbose=2, random_state=42)

mlpreg.fit(X_train_valid, y_train_valid)
print("score on train:{}".format(mlpreg.score(X_train_valid, y_train_valid)))
print("score on test:{}".format(mlpreg.score(X_test_valid, y_test_valid)))
from xgboost import XGBRegressor

for number in range(300, 600, 100):
    xgb_regressor = XGBRegressor(seed = 42, n_estimators=number)
    xgb_regressor.fit(X_train_valid, y_train_valid)
    
    print("===============")
    print("trees:{}".format(number))
    print("score on train:{}".format(xgb_regressor.score(X_train_valid, y_train_valid)))
    print("score on test:{}".format(xgb_regressor.score(X_test_valid, y_test_valid)))
    print("===============")
    
print('Done fitting!')
import xgboost
fig, ax = plt.subplots(figsize=(12,18))
xgboost.plot_importance(xgb_regressor, max_num_features=75, height=0.8, ax=ax)
plt.show()
from sklearn.feature_selection import RFECV

# How does this thing work? It trains a model on all the features and test it. 
# Then removes one feature and tests again. Compares the quality of the first and 
# second model - if it does not fall or deteriorate, this feature is considered noise,
# and more in the training will not participate.
# Then everything is repeated. At the end we have a list of features without noise.

xgb_feature_choosing = XGBRegressor(random_state = 42, n_estimators=2000, 
                             max_depth=3, learning_rate=0.1, 
                             early_stopping_rounds=5,
                             silent = True,
                            n_jobs=-1)


feature_eliminator = RFECV(xgb_feature_choosing,verbose = 1, n_jobs=-1)


# WARNING
# It will take a long time, about 2 hours, because we have a lot of parameters!

feature_eliminator.fit(X_train, y_train)

feature_array = feature_eliminator.support_

columns_array = X_test.columns

clean_array = columns_array[feature_array]
# Here I have a clean list in case you don't want to run the code at the top. 
# Clean_features_columns is the same as clean_array
clean_features_columns = ['1stFlrSF',
'2ndFlrSF',
'3SsnPorch',
'BedroomAbvGr',
'BsmtFinSF1',
'BsmtFinSF2',
'BsmtFullBath',
'BsmtHalfBath',
'BsmtUnfSF',
'EnclosedPorch',
'Fireplaces',
'FullBath',
'GarageArea',
'GarageCars',
'GarageYrBlt',
'GrLivArea',
'KitchenAbvGr',
'LotArea',
'LotFrontage',
'LowQualFinSF',
'MSSubClass',
'MasVnrArea',
'MiscVal',
'MoSold',
'OpenPorchSF',
'OverallCond',
'OverallQual',
'PoolArea',
'ScreenPorch',
'TotRmsAbvGrd',
'TotalBsmtSF',
'WoodDeckSF',
'YearBuilt',
'YearRemodAdd',
'YrSold',
'BsmtCond_Fa',
'BsmtCond_TA',
'BsmtExposure_Av',
'BsmtExposure_Gd',
'BsmtExposure_Mn',
'BsmtExposure_No',
'BsmtFinType1_ALQ',
'BsmtFinType1_GLQ',
'BsmtFinType1_LwQ',
'BsmtFinType1_Rec',
'BsmtQual_Fa',
'BsmtQual_Gd',
'Condition1_Artery',
'Condition1_Norm',
'Condition1_PosN',
'Condition1_RRAe',
'Condition2_Feedr',
'Electrical_SBrkr',
'ExterCond_Gd',
'ExterQual_Gd',
'Exterior1st_AsbShng',
'Exterior1st_BrkFace',
'Exterior1st_MetalSd',
'Exterior1st_Plywood',
'Exterior1st_VinylSd',
'Exterior1st_Wd Sdng',
'Exterior2nd_HdBoard',
'Exterior2nd_Stucco',
'Exterior2nd_Wd Sdng',
'Fence_MnPrv',
'Fence_No',
'FireplaceQu_Fa',
'FireplaceQu_Gd',
'FireplaceQu_TA',
'Functional_Min1',
'Functional_Typ',
'GarageCond_Fa',
'GarageFinish_Fin',
'GarageFinish_RFn',
'GarageFinish_Unf',
'GarageType_Attchd',
'GarageType_Detchd',
'HeatingQC_Ex',
'HeatingQC_Gd',
'HeatingQC_TA',
'HouseStyle_1.5Fin',
'HouseStyle_SLvl',
'KitchenQual_Ex',
'KitchenQual_Gd',
'LandContour_Lvl',
'LotConfig_Corner',
'LotConfig_CulDSac',
'LotConfig_FR2',
'LotConfig_Inside',
'LotShape_IR1',
'LotShape_Reg',
'MSZoning_C (all)',
'MSZoning_RH',
'MasVnrType_BrkFace',
'Neighborhood_BrkSide',
'Neighborhood_CollgCr',
'Neighborhood_Crawfor',
'Neighborhood_Edwards',
'Neighborhood_Gilbert',
'Neighborhood_Mitchel',
'Neighborhood_NAmes',
'Neighborhood_NoRidge',
'Neighborhood_NridgHt',
'Neighborhood_OldTown',
'Neighborhood_Sawyer',
'Neighborhood_SawyerW',
'Neighborhood_Somerst',
'Neighborhood_StoneBr',
'Neighborhood_Timber',
'PavedDrive_Y',
'RoofStyle_Gable',
'SaleCondition_Abnorml',
'SaleCondition_Family',
'SaleCondition_Normal',
'SaleType_New',
'SaleType_WD']
X_train_clean = X_train[clean_features_columns]
X_test_clean = X_test[clean_features_columns]
# There are lots of models up there! I'll just take one of them for making a prediction.

xgb_regressor = XGBRegressor(seed = 42, n_estimators=400)
xgb_regressor.fit(X_train_clean, y_train)
predicted_prices = xgb_regressor.predict(X_test_clean)
predicted_prices = predicted_prices * y_std
predicted_prices += y_mean
predicted_prices
submission = pd.DataFrame({"Id": test.Id, "SalePrice": predicted_prices})

submission.to_csv('submission.csv', index=False)

print("Submission Finished")