import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline
import seaborn as sns
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
constant = train.nunique().reset_index()
constant.columns = ["col", "count"]
constant = constant.loc[constant["count"]==1]
train = train.drop(columns=constant.col,axis = 1)
test = test.drop(columns=constant.col,axis = 1)
y = train["target"]
train = train.drop(["ID","target"],axis=1)
test = test.drop("ID",axis=1)
train["ID"] = train.index
test["ID"] = test.index
def maxabs(train,test):
    scaler = MaxAbsScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return train,test
y_train=np.log1p(y)
##start to test RF and tsvd below
trainSVD = train.copy()
testSVD = test.copy()
def rowagg(train,test):
    ##
    train["sum"] = train.sum(axis=1)
    test["sum"] = test.sum(axis=1)
    train["var"] = train.var(axis=1)
    test["var"] = test.var(axis=1)
    train["median"] = train.median(axis=1)
    test["median"] = test.median(axis=1)
    train["mean"] = train.mean(axis=1)
    test["mean"] = test.mean(axis=1)
    train["std"] = train.std(axis=1)
    test["std"] = test.std(axis=1)
    train["max"] = train.max(axis=1)
    test["max"] = test.max(axis=1)
    train["min"] =train.min(axis=1)
    test["min"] = test.min(axis=1)
    train["skew"] = train.skew(axis=1)
    test["skew"] = test.skew(axis=1)
    print ("Null values in train: "+ str(np.sum(np.sum(pd.isnull(train)))))
    print ("NAN values in train: "+ str(np.sum(np.isnan(train.values))))
    print ("Null values in test: "+ str(np.sum(np.sum(pd.isnull(test)))))
    print ("NAN values in test: "+ str(np.sum(np.isnan(test.values))))
    return train,test
from sklearn.decomposition import TruncatedSVD
trainSVD,testSVD = rowagg(trainSVD,testSVD)
svd = TruncatedSVD(n_components=2000)
res = svd.fit(trainSVD)
print (np.sum(res.explained_variance_ratio_))
trainSVD = res.transform(trainSVD)
testSVD = res.transform(testSVD)
import lightgbm as lgb
def run_lgb(X_train, Y_train, X_valid, Y_valid, test):
    seed = 42
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "task": "train",
        "boosting type":'dart',
        "num_leaves" :500,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_frequency" : 5,
        "bagging_seed" : seed,
        "verbosity" : -1,
        "seed": seed
    }
    lgtrain = lgb.Dataset(X_train,label= Y_train)
    lgval = lgb.Dataset(X_valid,label =Y_valid)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                  valid_sets=[lgtrain, lgval], 
                  early_stopping_rounds=300, 
                  verbose_eval=100, 
                  evals_result=evals_result)
    lgb_prediction = np.expm1(model.predict(test, num_iteration=model.best_iteration))
    return lgb_prediction, model, evals_result
from sklearn.model_selection import train_test_split
trainSVD,testSVD = maxabs(trainSVD,testSVD)
X_train, X_test, Y_train, Y_test = train_test_split(trainSVD, y_train, test_size=0.1, random_state=0)
lgb_predSVD, model, evals_resultRF = run_lgb(X_train, Y_train, X_test, Y_test, testSVD)
##stage2 :test on Random Forest
trainRF = train.copy()
testRF = test.copy()
trainRF,testRF = rowagg(trainRF,testRF)
rf_clf=RandomForestRegressor(random_state=42,n_jobs=-1)
rf_clf.fit(trainRF,y_train)
rank = pd.DataFrame()
rank["importance"] = np.array(rf_clf.feature_importances_)
rank["feature"] = np.array(trainRF.columns).T
rank = rank.sort_values(by=['importance'], ascending=False)
col = rank[:2000]
trainRF=trainRF[col.feature]
testRF=testRF[col.feature]
trainRF,testRF = maxabs(trainRF,testRF)
X_train, X_test, Y_train, Y_test = train_test_split(trainRF, y_train, test_size=0.1, random_state=0)
lgb_predRF, model, evals_resultRF = run_lgb(X_train, Y_train, X_test, Y_test, testRF)
sub=pd.read_csv('../input/sample_submission.csv')
sub["target"] = lgb_predRF
sub.to_csv('sub.csv', index=False)
sub.head()