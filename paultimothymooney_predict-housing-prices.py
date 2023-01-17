import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
hpart_file_path = '../input/train.csv'
hpart_data = pd.read_csv(hpart_file_path) 
y = hpart_data.SalePrice
all_features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition','Id','SalePrice']
X=hpart_data[all_features]
plt.figure(figsize=(12, 6))
_ = sns.distplot(y)
train_data=X
target=y
test_data = pd.read_csv('../input/test.csv')                            
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'], axis=1)
candidate_test_predictors = test_data.drop(['Id'], axis=1)
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='left',axis=1)
train_data=final_train
test_data=final_test
my_imputer = Imputer(axis=1)
imputed_train_data = my_imputer.fit_transform(train_data)
imputed_test_data = my_imputer.fit_transform(test_data)
imputed_train_data=pd.DataFrame(imputed_train_data)
imputed_test_data=pd.DataFrame(imputed_test_data)
final_train=imputed_train_data
final_test=imputed_test_data
X2 = final_train
y = target
train_X, test_X, train_y, test_y = train_test_split(X2, y,random_state = 0)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
test = pd.read_csv('../input/test.csv')
test_X2 = final_test
predicted_prices = my_model.predict(test_X2)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('Submission_A.csv', index=False)
X2 = final_train
y = target
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))
my_pipeline.fit(X2, y)
scores = cross_val_score(my_pipeline, X2, y, scoring='neg_mean_absolute_error')
print(scores)
# test = pd.read_csv('../input/test.csv')
test_X2 = final_test
predicted_prices = my_pipeline.predict(test_X2)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('Submission_B.csv', index=False)
selected_features =  ['MSZoning','Street','Utilities', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'TotalBsmtSF', 'CentralAir', '1stFlrSF', '2ndFlrSF',
       'TotRmsAbvGrd', 'Fireplaces','GarageType','GarageCars', 'PavedDrive', 'WoodDeckSF',
       'EnclosedPorch', 'YrSold','BsmtFinSF1','FullBath',
       'SaleCondition']
selected_features =  ['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF',
       'TotRmsAbvGrd', 'Fireplaces','GarageCars']
train_predictors_small=train_predictors[selected_features]
test_predictors_small=test_predictors[selected_features]
one_hot_encoded_training_predictors2 = pd.get_dummies(train_predictors_small)
one_hot_encoded_test_predictors2 = pd.get_dummies(test_predictors_small)
final_train2, final_test2 = one_hot_encoded_training_predictors2.align(one_hot_encoded_test_predictors2,join='left',axis=1)
train_data2=final_train2
test_data2=final_test2
my_imputer = Imputer(axis=1)
imputed_train_data2 = my_imputer.fit_transform(train_data2)
imputed_test_data2 = my_imputer.fit_transform(test_data2)
imputed_train_data2=pd.DataFrame(imputed_train_data2)
imputed_test_data2=pd.DataFrame(imputed_test_data2)
final_train2=imputed_train_data2
final_test2=imputed_test_data2
X3 = final_train2
y = target
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))
my_pipeline.fit(X3, y)
scores = cross_val_score(my_pipeline, X3, y, scoring='neg_mean_absolute_error')
print(scores)
# test = pd.read_csv('../input/test.csv')
test_X3 = final_test2
predicted_prices = my_pipeline.predict(test_X3)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('Submission_C.csv', index=False)
X4 = final_train
xValuesScaled = preprocessing.scale(X4)
xValuesScaled = pd.DataFrame(xValuesScaled)
yValues = target
X5 = final_test
xValuesScaledTest = preprocessing.scale(X5)
xValuesScaledTest = pd.DataFrame(xValuesScaledTest)
yValues = target
variance_pct = .9999 # Minimum percentage of variance we want to be described by the resulting transformed components
pca = PCA(n_components=variance_pct) # Create PCA object
X_transformed = pca.fit_transform(xValuesScaled,yValues) # Transform the initial features
xValuesScaledPCA = pd.DataFrame(X_transformed) # Create a data frame from the PCA'd data
X_transformedTest = pca.transform(xValuesScaledTest,) 
xValuesScaledPCATest = pd.DataFrame(X_transformedTest) 
print("\nFeature Correlation Before PCA:\n")
g = sns.heatmap(xValuesScaled.corr(),cmap="BrBG",annot=False)
print("\nFeature Correlation After PCA:\n")
g = sns.heatmap(xValuesScaledPCA.corr(),cmap="BrBG",annot=False)
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))
my_pipeline.fit(xValuesScaledPCA, yValues)
scores = cross_val_score(my_pipeline, xValuesScaledPCA, yValues, scoring='neg_mean_absolute_error')
print(scores)
# test = pd.read_csv('../input/test.csv')
test_X3 = xValuesScaledPCATest
predicted_prices = my_pipeline.predict(test_X3)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('Submission_D.csv', index=False)
X_trainScaled, X_testScaled, Y_trainScaled, Y_testScaled = train_test_split(xValuesScaled, yValues, test_size=0.2)
X_trainScaledPCA, X_testScaledPCA, Y_trainScaledPCA, Y_testScaledPCA = train_test_split(xValuesScaledPCA, yValues, test_size=0.2)
def compareABunchOfRegressors(a, b, c, d):
    """
    compare performance of classifiers on X_train, X_test, Y_train, Y_test
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    """    
    print('\nCompare Multiple Classifiers: \n')
    print('neg_mean_absolute_error Score: \n')
    names = []
    models = []
    resultsAccuracy = []
    models.append(('XGB', XGBRegressor()))
    models.append(('ADA', AdaBoostRegressor()))
    models.append(('BR', BayesianRidge()))
    models.append(('DTR', DecisionTreeRegressor()))
    models.append(('EN', ElasticNet()))
    models.append(('GBR', GradientBoostingRegressor()))
    models.append(('KNR', KNeighborsRegressor()))
    #models.append(('L', Lasso()))
    #models.append(('LR', LinearRegression()))
    #models.append(('LSVR', LinearSVR()))
    models.append(('RFR', RandomForestRegressor()))
    models.append(('Ridge', Ridge()))
    #models.append(('SGD', SGDRegressor()))
    for name, model in models:
        model.fit(a, b)
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        accuracy_results = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='neg_mean_absolute_error')
        resultsAccuracy.append(accuracy_results)
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())
        print(accuracyMessage) 
    # Boxplot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: neg_mean_absolute_error Score')
    ax = fig.add_subplot(111)
    plt.boxplot(resultsAccuracy)
    ax.set_xticklabels(names)
    ax.set_ylabel('neg_mean_absolute_error Score')
    plt.show()

print("After Data Scaling:")
compareABunchOfRegressors(X_trainScaled, Y_trainScaled, X_testScaled, Y_testScaled)
print("After PCA:")
compareABunchOfRegressors(X_trainScaledPCA, Y_trainScaledPCA, X_testScaledPCA, Y_testScaledPCA)

def defineModels2():
    """
    This function just defines each abbreviation used in the previous function (e.g. LR = Logistic Regression)
    """
    print('\nADA = AdaBoostRegressor')
    print('BR = BayesianRidge')
    print('DTR = DecisionTreeRegressor')
    print('EN = ElasticNet')
    print('GBR = GradientBoostingRegressor')
    print('KNR = KNeighborsRegressor')
    #print('L = Lasso')
    #print('LR = LinearRegression \n\n')
    #print('LSCVR = LinearSVR')
    print('RFR = RandomForestRegressor')
    print('LSCVR = LinearSVR')
    print('R = Ridge')
    #print('SGD = SGDRegressor')
    print('XGB = XGBRegressor')
defineModels2()
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

clf = XGBRegressor(
    eval_metric = 'rmse',
    nthread = 4,
    eta = 0.1,
    num_boost_round = 80,
    max_depth = 5,
    subsample = 0.5,
    colsample_bytree = 1.0,
    silent = 1,
    )
parameters = {
    'num_boost_round': [5, 10, 50],
    'eta': [0.05, 0.1, 0.3],
    'max_depth': [3, 10, 15],
    'subsample': [0.9, 1.0],
    'colsample_bytree': [0.9, 1.0],
}

clf2 = RandomizedSearchCV(clf, parameters, n_jobs=1, cv=5)
clf2.fit(X_trainScaled, Y_trainScaled)
print("Optimized XGBoost Parameters:",clf2)
print('\nAccuracy of Optimized XGBoost Regressor on training data: {:.2f}'.format(clf2.score(X_trainScaled, Y_trainScaled)))
print('Accuracy of Optimized XGBoost Regressor on testing data: {:.2f}'.format(clf2.score(X_testScaled, Y_testScaled)))
clf = XGBRegressor(n_estimators=500, learning_rate=0.1, scoring='neg_mean_absolute_error',early_stopping_rounds=5,max_depth=5,min_child_weight=200)
clf.fit(X_trainScaled, Y_trainScaled)
print('Accuracy of XGBoost Regressor on training data: {:.2f}'.format(clf.score(X_trainScaled, Y_trainScaled)))
print('Accuracy of XGBoost Regressor Classifier on testing data: {:.2f}'.format(clf.score(X_testScaled, Y_testScaled)))