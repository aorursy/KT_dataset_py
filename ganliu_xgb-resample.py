import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn.metrics import confusion_matrix
# read train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# remove constant columns
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,train[c[j]].values):
            remove.append(c[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
unhappy = train.loc[train['TARGET']==1]
happy = train.loc[train['TARGET']==0]
np.random.seed(10)
train = pd.concat([unhappy.sample(len(happy), replace=True), happy], ignore_index=True)
train = train.reindex(np.random.permutation(train.index)).reset_index(drop=True)
def modelfit(alg, dtrain, predictors, performCV=True,cv_folds=5):
    X = dtrain[predictors].values
    Y = dtrain['TARGET'].values
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(
            alg, X, Y, cv=cv_folds, scoring='roc_auc'
        )
    #Fit the algorithm on the data
    alg.fit(X,Y)
    #Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
    #Print model report:
    print("Model Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['TARGET'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['TARGET'], dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
base_model = xgb.XGBClassifier(
    objective='binary:logistic', n_estimators=300, learning_rate=0.04, 
    max_depth=5, nthread=4, subsample=0.7, colsample_bytree=0.5, 
    reg_lambda=6, reg_alpha=5, seed=10, silent=True
)
predictors = train.columns[1:-1]
modelfit(base_model, train, predictors)
import matplotlib.pyplot as plt
%matplotlib inline
def model_score_and_feature(clf, features, n_feature=10):
    sorted_index = np.argsort(clf.feature_importances_)[::-1]
    top_feature = sorted_index[:n_feature]
    top_feature_score = clf.feature_importances_[sorted_index[:n_feature]]
    plt.barh(range(n_feature), top_feature_score[::-1])
    ax = plt.gca()
    ax.set_yticks(np.arange(n_feature)+0.5)
    ax.set_yticklabels(features[top_feature][::-1])
    plt.show()
model_score_and_feature(base_model, train.columns[1:-1])