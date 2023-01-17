import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

from sklearn import metrics

%matplotlib inline
def modelfit(alg, dtrain, predictors,target,useTrainCV=False, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgb_param['num_class'] = 3

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    #Print model report:

    print("\nModel Report")

    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))

                    

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')

    

    return alg
testset = pd.read_csv("../input/test.csv")

trainset = pd.read_csv("../input/train.csv")

testset.drop("color", axis=1, inplace=True)

trainset.drop("color", axis=1, inplace=True)

trainset.loc[trainset['type'] == 'Ghoul', 'type'] = 0

trainset.loc[trainset['type'] == 'Ghost', 'type'] = 1

trainset.loc[trainset['type'] == 'Goblin', 'type'] = 2

trainset['type'] = pd.to_numeric(trainset['type'])
clf = xgb.sklearn.XGBClassifier(objective="multi:softmax");
predictors = [x for x in trainset.columns if x not in ["id","type"]]
clf = modelfit(clf, trainset, predictors, 'type')
preds = clf.predict(testset.drop("id", axis=1))
preds = preds.astype('O')

preds[preds == 0] = 'Ghoul'

preds[preds == 1] = 'Ghost'

preds[preds == 2] = 'Goblin'
sub = pd.DataFrame(preds)

pd.concat([testset["id"],sub], axis=1).rename(columns = {0: 'type'}).to_csv("submission.csv", index=False)