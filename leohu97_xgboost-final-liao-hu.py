# Import Necessary Libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Import Data
trainF = pd.read_csv("../input/trainFeatures.csv")
trainL = pd.read_csv("../input/trainLabels.csv")
testF = pd.read_csv("../input/testFeatures.csv")
# Drop columns with a large number of unique values, since they may become noise(just like ids)
dropcol = {'ids', 'RatingID','AccountabilityID', 'RatingYear','BaseYear', 'date_calculated', 'previousratingid', 'Rpt_Comp_Date','Rpt_Ap_Date', 'Rpt_Ver_Date'}

# Here are the functions relate to feature selection. Not all methods are used at last.
def conv_nan(df):
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column].fillna(mean_val, inplace=True)
    return df

def drop_percent_nan(df):
    for col in df.columns:
        if (df[col].isnull().sum())/len(df) > 0.95:
            df.drop(col,inplace=True,axis=1)
    df.drop(dropcol,inplace=True,axis=1)
    return df

def drop_all_nan(df):
    for col in df.columns:
        if (df[col].isnull().sum()) == df.shape[0]:
            df.drop(col,inplace=True,axis=1)
    df.drop(dropcol,inplace=True,axis=1)
    return df

def apply_pca(df):
    pca = PCA(copy=False, n_components=40, whiten=False)
    df = pca.fit_transform(df)
    return df
# There is a inherent bug with n_components = 'mle', thus Auto PCA is not used.

# Convert 'erkey' columnn to value
trainF['erkey'] = trainF['erkey'].str.slice_replace(0, 3, '')
trainF['erkey'] = pd.to_numeric(trainF['erkey'])

# Further data refining. 
trainData = pd.merge(trainF, trainL, on='ids')
trainData = drop_percent_nan(trainData)
trainData = conv_nan(trainData)
X_train = trainData.iloc[:,0:((trainData.shape[1]-2))]
y_train = trainData.iloc[:,(trainData.shape[1]-1):(trainData.shape[1])]
X_train = np.array(X_train)
y_train = np.array(y_train)

# Apply to the test data.
testF['erkey'] = testF['erkey'].str.slice_replace(0, 3, '')
testF['erkey'] = pd.to_numeric(testF['erkey'])
X_test = drop_percent_nan(testF)
testF = conv_nan(testF)
X_test = np.array(X_test)
# Various hyper-parameters to tune
# 'cv_params2' is a set of example candidate parameters, which need to be test sequentially. 
cv_params2 = {'n_estimators': [15000,20000,40000],
             'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
             'min_child_weight': [1, 2, 3, 4, 5, 6],
             'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
             'subsample': [0.6, 0.7, 0.8, 0.9],
             'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
             'reg_alpha': [0.05, 0.1, 1, 2, 3],
             'reg_lambda': [0.05, 0.1, 1, 2, 3],
             'learning_rate': [0.01, 0.05, 0.1, 1]}

# 'cv_params' is the parameter that is being tuned.
cv_params = {'learning_rate': [0.01]}

# 'other_params' is the default/tuned parameters that will be used in training.
other_params = {'learning_rate': 0.01,
                'n_estimators': 22500,
                'max_depth': 6,
                'min_child_weight': 5,
                'seed': 0,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'gamma': 0.5,
                'reg_alpha': 3,
                'reg_lambda': 1}

# Training is performed at Google Cloud Platform with 8 vCPU, thus n_jobs is 20.
# Personal Computer/Laptop may only works with 8.
# It took about 10 hours on GCP to tune all parameter sets.
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=2, n_jobs=20)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
print('Results from each iteration:{0}'.format(evalute_result))
print('Beat parameter set：{0}'.format(optimized_GBM.best_params_))
print('Best score:{0}'.format(optimized_GBM.best_score_))
# Write output to csv file
y_pred = optimized_GBM.predict(X_test)
ind = np.arange(1,len(y_pred)+1,1)
dataset = pd.DataFrame({'id':ind,'OverallScore':y_pred})
dataset.to_csv("XGBoost Final pre.csv", sep=',', index=False)
print("done")