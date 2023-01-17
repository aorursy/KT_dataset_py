import numpy as np

import pandas as pd

test = pd.read_csv("../input/catindat2encoded/test_encoded.csv")

train = pd.read_csv("../input/catindat2encoded/train_encoded.csv")

test_id = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")['id']
target = train['target']

train.drop(['target'], axis = 1, inplace = True)

# test.drop(['id'], axis = 1, inplace = True)
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate

from sklearn.linear_model import LogisticRegression, ElasticNet, SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
def generate_oof_trainset( train, test, target, strat_kfold, models,):



    oof_train = pd.DataFrame() # Initializing empty data frame

    

    count = 0

    print(train.shape, target.shape)



    for train_id, test_id in strat_kfold.split(train, target):

        count += 1

        print("Current fold number is :", count)

        xtrain, xtest = train.iloc[train_id], train.iloc[test_id]

        ytrain, ytest = target.iloc[train_id], target.iloc[test_id]

        

        curr_split = [None]*(len(models)+1) # Initializing list of lists to save all predictions for a split from all models for the current split

        

        for i in tqdm(range(len(models))):

            

            model = models[i]

            model.fit(xtrain, ytrain)

            

            curr_split[i] = model.predict_proba(xtest)[:,1]      

            

        curr_split[-1] = ytest

        oof_train = pd.concat([oof_train,pd.DataFrame(curr_split).T], ignore_index= True)

    

    oof_test = [None]*len(models)

    for i, model in enumerate(models):

        model.fit( train, target)

        oof_test[i] = model.predict_proba(test)[:,1]

    oof_test = pd.DataFrame(oof_test)

    return oof_train, oof_test

    
from tqdm import tqdm

strat_kfold = StratifiedKFold( n_splits =2, shuffle = 

              True)



log_reg = LogisticRegression(random_state = 0)

gbr = GradientBoostingClassifier(

        max_depth=6,

        n_estimators=35,

        warm_start=False,

        random_state=42)

adar = AdaBoostClassifier(n_estimators=100, random_state=0)



models = [ log_reg, gbr, adar ]

train_generated, test_generated = generate_oof_trainset( train, test, target, strat_kfold, models)
lr_clf = LogisticRegression()

target = train_generated[train_generated.columns[-1]]

train_generated.drop([train_generated.columns[-1]], axis = 1 , inplace = True)



cv_results = cross_validate(lr_clf,

                            train_generated.values,

                            target.values,

                            cv = 3,

                            scoring = 'roc_auc',

                            verbose = 1,

                            return_train_score = True,

                            return_estimator = True)



print("Fit time :", cv_results['fit_time'].sum(),"secs")

print("Score time :", cv_results['score_time'].sum(),"secs")

print("Train score :", cv_results['train_score'].mean())

print("Test score :", cv_results['test_score'].mean())   
lr_clf.fit(train_generated.values, target.values)

preds = lr_clf.predict(test_generated.T.values)