!pip install pycaret
!pip install pandas_profiling
import pandas as pd 
import pandas_profiling as pp
#read the data
train_titanic=pd.read_csv('/kaggle/input/titanic/train.csv')
test_titanic=pd.read_csv('/kaggle/input/titanic/test.csv')
train_house=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_house=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#data dimensions
print('Titanic train:',train_titanic.shape)
print('Titanic test:',test_titanic.shape)
print('House price train:',train_house.shape)
print('House price test:',test_house.shape)
#pandas profiling report for Titanic train data
pp.ProfileReport(train_titanic)
#import pycaret classification library
from pycaret.classification import * #for classification
#classification setup
classification_setup =setup(data = train_titanic, 
             target = 'Survived',
             numeric_imputation = 'mean', #fill missing value with mean for numeric features
             categorical_features = ['Sex','Embarked','Pclass','Ticket','Cabin'], #we know categorical features from pandas profiling report
             ignore_features = ['Name','PassengerId'],
             train_size=0.8, #0.7 as default
             high_cardinality_features=['Cabin'],
             normalize=True,
             normalize_method='minmax',
             handle_unknown_categorical=True,
             unknown_categorical_method='most_frequent',  #fill missing value with most frequent value for categorical features
             remove_outliers=True, #it automatically applies PCA for removing outliers,
             outliers_threshold=0.05, #By default, 0.05 is used which means 0.025 of the values on each side of the distribution’s tail are dropped from training data.
             silent=True,
             profile=True #a data profile for Exploratory Data Analysis will be displayed in an interactive HTML report. It also generates pandas profiling report
     )
#comparing models
blacklist_models = ['svm','rbfsvm','mlp']

compare_models(
    blacklist=blacklist_models, #blacklisted models won't work.
    fold = 5,
    sort = 'Accuracy', ## competition metric
    turbo = True
)
#creating model with selected estimator.
xgb=create_model(estimator='xgboost',fold=5)
#tune the model
tuned_xgb = tune_model('xgboost')
# ensembling a trained xgboost model
xgb_bagged = ensemble_model(xgb)
plot_model(xgb, plot = 'boundary')# Decision Boundary
plot_model(xgb, plot = 'pr')# Precision Recall Curve
plot_model(xgb, plot = 'vc')# Validation Curve
plot_model(xgb, plot='confusion_matrix') # Confusion Matrix
#Evaluating model is a good option. Because you don't need to plot different plots seperately. It provides all of them in the same cell.
evaluate_model(xgb)
#As you remember, we split 80% of the data for training. The rest of the data can be used for holdout prediction.
xgb_holdout_pred = predict_model(xgb)
#Or you can use your test data for prediction.
titanic_prediction =  predict_model(xgb, data=test_titanic)
titanic_prediction.head()
#prepare the submission file
titanic_prediction['Survived'] = round(titanic_prediction['Score']).astype(int)
submission=titanic_prediction[['PassengerId','Survived']]
submission.to_csv('submission.csv',index=False)
submission.head()
#pandas profiling report for House price train data
pp.ProfileReport(train_house)
from pycaret.regression import * #for regression

#regression setup
regression_setup =setup(data = train_house, 
             target = 'SalePrice',
             numeric_imputation = 'mean', #fill missing value with mean for numeric features
             categorical_features = ['MSZoning','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType',
                                     'Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',   
                                     'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',    
                                     'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',   
                                     'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',   
                                     'Electrical','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                                     'SaleCondition'], #we know categorical features from pandas profiling report
             ignore_features = ['Id'],
             train_size=0.8, #0.7 as default
             normalize=True,
             normalize_method='minmax',
             handle_unknown_categorical=True,
             unknown_categorical_method='most_frequent',  #fill missing value with most frequent value for categorical features
             remove_outliers=True, #it automatically applies PCA for removing outliers,
             outliers_threshold=0.05, #By default, 0.05 is used which means 0.025 of the values on each side of the distribution’s tail are dropped from training data.
             silent=True,
             profile=True #a data profile for Exploratory Data Analysis will be displayed in an interactive HTML report. It also generates pandas profiling report
     )
bl_models = ['ransac', 'tr', 'rf', 'et', 'ada', 'gbr']

compare_models(
    blacklist = bl_models,
    fold = 5,
    sort = 'MAE', ## competition metric
    turbo = True
)
#creating model.
lgbm = create_model(
    estimator='lightgbm',
    fold=5
)
#Evaluating model.
evaluate_model(lgbm)
# use rest of the training data for holdout prediction.
lgbm_holdout_pred = predict_model(lgbm)
#prediction with test data.
house_prediction =  predict_model(lgbm, data=test_house)
house_prediction.head()
#prepare the submission file
house_prediction.rename(columns={'Label':'SalePrice'}, inplace=True)
house_prediction[['Id','SalePrice']].to_csv('submission_house.csv', index=False)
from pycaret.clustering import * #for clustering
#clustering setup doesn't support silent True options. So, you need to hit ENTER manually.
#if silent is set True, it means that you approve data types which were inferred from PyCaret.

#clustering setup
clustering_setup =setup(data = train_titanic, 
             numeric_imputation = 'mean', #fill missing value with mean for numeric features
             categorical_features = ['Sex','Embarked','Pclass','Ticket','Cabin'], #we know categorical features from pandas profiling report
             ignore_features = ['Name','PassengerId'],
             high_cardinality_features=['Cabin'],
             normalize=True,
             normalize_method='minmax',
             handle_unknown_categorical=True,
             unknown_categorical_method='most_frequent',  #fill missing value with most frequent value for categorical features
             verbose=False        
    
     )
#use k-means for clustering. You can check other clustering algorithms from https://pycaret.org/clustering/
kmeans = create_model('kmeans')
#assigning data to clusters.
kmeans_df = assign_model(kmeans)
kmeans_df.head()
# PCA Plot
plot_model(kmeans) 
#Also you can plot Silhouette
plot_model(kmeans, plot='silhouette') 
# Or you can plot Elbow etc.
plot_model(kmeans, plot='elbow') 
# tunes the num_clusters model parameter using a predefined grid with the objective of optimizing a supervised learning metric as defined in the optimize param. 
tuned_kmeans = tune_model(model = 'kmeans', supervised_target = 'Survived')
# Also, you can specify estimator for tuning.
tuned_kmeans = tune_model(model = 'kmeans', supervised_target = 'Survived', estimator='xgboost')
