import pandas as pd

import numpy as np

from tqdm import tqdm

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

from sklearn import cross_validation, metrics   #Additional scklearn functions

from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

sample_submission = pd.read_csv("sample_submission.csv")



target = 'target'

IDcol = 'id'
## Over sampling 



pos = train[train.target == 1]

train = pd.concat([train] + [pos]*10 , axis = 0)
def modelfit(alg, dtrain, predictors, dtest , test_predictors, performCV=False , printFeatureImportance=True, cv_folds=10):

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain['target'])

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    test_pred = alg.predict_proba(dtest[test_predictors])[:,1]

    test_class = alg.predict(dtest[test_predictors])

    #Perform cross-validation:

    if performCV:

        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['target'], cv=cv_folds, scoring='roc_auc')

    

    #Print model report:

    print ("\nModel Report")

    print( "Accuracy : %.4g" % metrics.accuracy_score(dtrain['target'].values, dtrain_predictions))

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['target'], dtrain_predprob))

    if performCV:

        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    #Print Feature Importance:

    if printFeatureImportance:

        plt.rcParams['figure.figsize'] = (20.0, 20.0)

        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)

        feat_imp.plot(kind='bar', title='Feature Importances')

        plt.ylabel('Feature Importance Score')

        plt.show()

    return([test_pred,test_class])
#Choose all predictors except target & IDcols

predictors = [x for x in train.columns if x not in [target, IDcol]]

test_predictors = [x for x in test.columns if x not in [IDcol]]

gbm0 = GradientBoostingClassifier(random_state=10 , verbose = True , n_estimators = 100)

test_pred = modelfit(gbm0, train, predictors , test , test_predictors ,performCV = False)
class_ = test_pred[0] 

sol = pd.concat([test['id'] , pd.DataFrame(class_)], axis = 1)

sol.columns = ['id', 'target']

sol.to_csv("gbm_up_sampling20_submission.csv" , index=False)