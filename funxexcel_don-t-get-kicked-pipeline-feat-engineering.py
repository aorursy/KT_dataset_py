import pandas as pd
import numpy as np
#from catboost import CatBoostClassifier
#from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
#from sklearn.metrics import accuracy_score

#Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

#For Missing Value and Feature Engineering
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import time

train = pd.read_csv("../input/DontGetKicked/training.csv")
test = pd.read_csv("../input/DontGetKicked/test.csv")
train.head()
#insert code
# Date
#PurchDate
train['mean_MMRAcquisitionAuctionAveragePrice_Make']=train.groupby(['Make'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_Model']=train.groupby(['Model'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_Trim']=train.groupby(['Trim'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_SubModel']=train.groupby(['SubModel'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_Color']=train.groupby(['Color'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_Transmission']=train.groupby(['Transmission'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Make']=test.groupby(['Make'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Model']=test.groupby(['Model'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Trim']=test.groupby(['Trim'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_SubModel']=test.groupby(['SubModel'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Color']=test.groupby(['Color'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Transmission']=test.groupby(['Transmission'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
#create X and y datasets for splitting 
X = train.drop(['RefId', 'IsBadBuy'], axis=1)
y = train['IsBadBuy']
all_features = X.columns
all_features = all_features.tolist()
numerical_features = [c for c, dtype in zip(X.columns, X.dtypes)
                     if dtype.kind in ['i','f'] and c !='PassengerId']
categorical_features = [c for c, dtype in zip(X.columns, X.dtypes)
                     if dtype.kind not in ['i','f']]
numerical_features
categorical_features
#import train_test_split library
from sklearn.model_selection import train_test_split

# create train test split
X_train, X_test, y_train, y_test = train_test_split( X,  y, test_size=0.3, random_state=0)  
preprocessor = make_column_transformer(
    
    (make_pipeline(
    SimpleImputer(strategy = 'median'),
    MinMaxScaler()), numerical_features),
    
    (make_pipeline(
    SimpleImputer(strategy = 'constant', fill_value = 'missing'),
    OneHotEncoder(categories = 'auto', handle_unknown = 'ignore')), categorical_features),
    
)
preprocessor_best = make_pipeline(preprocessor, SelectKBest(chi2, k = 50))
RF_Model = make_pipeline(preprocessor_best, RandomForestClassifier(n_estimators = 100))
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
#Maximum number of levels in tree
max_depth = [2,4,6,8]
# Minimum number of samples required to split a node
#min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
#bootstrap = [True, False]
# Create the param grid
param_grid = {'randomforestclassifier__n_estimators': n_estimators,
               'randomforestclassifier__max_features': max_features,
               'randomforestclassifier__max_depth': max_depth
               #'randomforestclassifier__min_samples_split': min_samples_split,
               #'randomforestclassifier__min_samples_leaf': min_samples_leaf,
               #'randomforestclassifier__bootstrap': bootstrap
             }
print(param_grid)
from sklearn.model_selection import RandomizedSearchCV
rf_RandomGrid = RandomizedSearchCV(estimator = RF_Model, param_distributions = param_grid, cv = 3, verbose=1, n_jobs = -1)
rf_RandomGrid.fit(X_train, y_train)
rf_RandomGrid.best_estimator_
print(f'Train : {rf_RandomGrid.score(X_train, y_train):.3f}')
print(f'Test : {rf_RandomGrid.score(X_test, y_test):.3f}')
def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
actual_train = y_train
pred_train = rf_RandomGrid.predict(X_train)
actual_test = y_test
pred_test = rf_RandomGrid.predict(X_test)
print(f'Gini Train : {gini(actual_train,pred_train):.3f}')
print(f'Gini Test : {gini(actual_test,pred_test):.3f}')
test_pred = rf_RandomGrid.predict_proba(test[X.columns])[:,1]
AllSub = pd.DataFrame({ 'RefId': test['RefId'],
                       'IsBadBuy' : test_pred
    
})
AllSub['IsBadBuy'] = AllSub['IsBadBuy'].apply(lambda x: 1 if x > 0.09 else 0)
AllSub.to_csv('PyCaret_RF_Pipe_Feat_2.csv', index = False)
