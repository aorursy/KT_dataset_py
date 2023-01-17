import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

from xgboost.sklearn import XGBClassifier

from sklearn import metrics   #Additional scklearn functions

from sklearn.grid_search import GridSearchCV   #Perforing grid search

from sklearn.feature_selection import SelectFromModel





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = pd.read_csv("../input/train_final.csv")

test = pd.read_csv("../input/test_final.csv")

test.drop('id', 1, inplace = True)
train_predict = train['Y']

predictor = train.drop('Y', 1)

predictor = predictor.drop('id', 1)

predicts = predictor
import xgboost as xgb

import boruta as b

from xgboost.sklearn import XGBClassifier
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain, label= train_predict)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    

    #Fit the algorithm on the data

    alg.fit(dtrain, train_predict,eval_metric='auc')

        



    #Predict training set:

    dtrain_predictions = alg.predict(dtrain)

    dtrain_predprob = alg.predict_proba(dtrain)[:,1]

    

    #Print model report:

    print("\nModel Report")

    print("Accuracy : %.4g" % metrics.accuracy_score(train_predict, dtrain_predictions))

    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_predict, dtrain_predprob))
dtrain = xgb.DMatrix(train, label = train_predict)

dtest = xgb.DMatrix(test)



params = {"max_depth":2, "eta":0.1}
xgb1 = XGBClassifier(

 learning_rate =0.01,

 n_estimators=5000,

 max_depth=4,

 min_child_weight=2,

 gamma=0,

 subsample=0.4,

 colsample_bytree=0.7,

 reg_alpha = 0.01,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)



modelfit(xgb1, predictor, predictor)



solution = xgb1.predict(test)

pd.DataFrame(solution).to_csv("sol.csv", index = False)
#trying Boruta selection algorithm

#predictor = predictor.fillna(-1)



#boruta_features = b.BorutaPy(xgb1, n_estimators='auto', verbose=2)

#boruta_features.fit(predictor.values, train_predict.values)

#dtrain_select = boruta_features.transform(predictor.values)

#modelfit(xgb1, dtrain_select, dtrain_select)
""" Amazon Access Challenge Code for ensemble

Marios Michaildis script for Amazon .

xgboost on input data

based on Paul Duan's Script.

"""

from __future__ import division

import numpy as np

import pandas as pd

from sklearn import  preprocessing

from sklearn.metrics import roc_auc_score

from sklearn.cross_validation import StratifiedKFold

from scipy import stats



SEED = 42  # always use a seed for randomized procedures







def save_results(predictions, filename):

    """Given a vector of predictions, save results in CSV format."""

    with open(filename, 'w') as f:

        f.write("id,ACTION\n")

        for i, pred in enumerate(predictions):

            f.write("%d,%f\n" % (i + 1, pred))





def bagged_set(X_t,y_c,model, seed, estimators, xt, update_seed=True):

    

   # create array object to hold predictions 

   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]

   #loop for as many times as we want bags

   for n in range (0, estimators):

        #shuff;e first, aids in increasing variance and forces different results

        #X_t,y_c=shuffle(Xs,ys, random_state=seed+n)

          

        if update_seed: # update seed if requested, to give a slightly different model

            model.set_params(random_state=seed + n)

        model.fit(X_t,y_c) # fit model0.0917411475506

        preds=model.predict_proba(xt)[:,1] # predict probabilities

        # update bag's array

        for j in range (0, (xt.shape[0])):           

                baggedpred[j]+=preds[j]

   # divide with number of bags to create an average estimate            

   for j in range (0, len(baggedpred)): 

                baggedpred[j]/=float(estimators)

   # return probabilities            

   return np.array(baggedpred) 

   

   

# using numpy to print results

def printfilcsve(X, filename):



    np.savetxt(filename,X) 

    



def main():

    """

    Fit models and make predictions.

    We'll use one-hot encoding to transform our categorical features

    into binary features.

    y and X will be numpy array objects.

    """

    

    #model = linear_model.LogisticRegression(C=3)  # the classifier we'll use

    

    #model=xg.XGBoostClassifier(num_round=1000 ,nthread=25,  eta=0.12, gamma=0.01,max_depth=12, min_child_weight=0.01, subsample=0.6, 

                                   #colsample_bytree=0.7,objective='binary:logistic',seed=1) 



    #creating new features which are multiplied together

    #m = pd.DataFrame()

    #for col in predictor:

        #for cols in predictor:

            #if col != cols:

                #m.append(predictor[cols] * predictor[col])

    train = pd.read_csv("../input/train_final.csv")

    test = pd.read_csv("../input/test_final.csv")

    test.drop('id', 1, inplace = True)         

    

    train_predict = train['Y']

    predictor = train.drop('Y', 1)

    predictor = predictor.drop('id', 1)

    

    #creating new features that are multiples of each other

    

    #for col in predictor:

        #for k in predictor:

            #n = pd.DataFrame() 

            #m = []

            #for j in range(0, len(predictor[k])):

                #m += predictor[k][j] * predictor[col][j] 

            #m = pd.DataFrame(m)

            #n.append(m)

    #predictor.append(n)

    print("done adding features...")

    

    #determining which cols are categorical and which are numerical

    cat = []

    num = []

    cat_test= []

    num_test = []

    

    for col in predictor:

        count = np.unique(predictor[col])

        if len(count) > 100:

            num.append(col)

        else:

            cat.append(col)

            

    for col in test:

        count = np.unique(predictor[col])

        if len(count) > 100:

            num_test.append(col)

        else:

            cat_test.append(col)

    

    #replacing NAN values

    for k in predictor[cat]:

        for j in predictor[k]:

            if np.isnan(j):

                j = stats.mode(predictor[cat])

    

    for k in predictor[num]:

        for j in predictor[k]:

            if np.isnan(j):

                j = np.mean(predictor[num])

     

    for k in test[cat_test]:

        for j in test[k]:

            if np.isnan(j):

                j = stats.mode(test[cat_test])

    

    for k in test[num_test]:

        for j in test[k]:

            if np.isnan(j):

                j = np.mean(test[num_test])

    print("done replacing NAN...")

    

    #transforming log of predictor data

    for k in predictor[num]:

        for j in predictor[k]:

            if np.isnan(j):

                j = np.log10(j)

    

    for k in test[num]:

        for j in test[k]:

            if np.isnan(j):

                j = np.log10(j)

    print("done logging data...")

    

    #transforming hot encoding of predictor data

    train_m = pd.DataFrame()

    test_m = pd.DataFrame()

    train_n = pd.DataFrame()

    test_n = pd.DataFrame()

    

    for k in predictor[cat]:

        train_m = train_m.assign(predictor[k])



    for k in test[cat]:

        test_m.append(test[k])

        

    for k in predictor[num]:

        train_n.append(predictor[k])



    for k in test[num]:

        test_n.append(test[k])

    

    print(train_m)

    # === one-hot encoding === #

    # we want to encode the category IDs encountered both in

    # the training and the test set, so we fit the encoder on both 

    print("encoding...")

    encoder = preprocessing.OneHotEncoder()

    encoder.fit(np.vstack((train_m, test_m)))

    train_m = encoder.transform(train_m)  # Returns a sparse matrix (see numpy.sparse)

    test_m = encoder.transform(test_m)

    

    test = test_n.append(test_m)

    predictor = train_n.append(train_m)

   

    # if you want to create new features, you'll need to compute them

    # before the encoding, and append them to your dataset after



    #create arrays to hold cv an dtest predictions

    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ] 



    # === training & metrics === #

    mean_auc = 0.0

    bagging=20 # number of models trained with different seeds

    n = 5  # number of folds in strattified cv

    kfolder=StratifiedKFold(predictor, n_folds= n,shuffle=True, random_state=SEED)     

    i=0

    for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object

        # creaning and validation sets

        

        X_train, X_cv = predictor[train_index], predictor[test_index]

        y_train, y_cv = np.array(predictor)[train_index], np.array(predictor)[test_index]

        #print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))



        # if you want to perform feature selection / hyperparameter

        # optimization, this is where you want to do it



        # train model and make predictions 

        preds=bagged_set(X_train,y_train, xgb1, SEED , bagging, X_cv, update_seed=True)   

        



        # compute AUC metric for this CV fold

        roc_auc = roc_auc_score(y_cv, preds)

        print ("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))

        mean_auc += roc_auc

        

        no=0

        for real_index in test_index:

                 train_stacker[real_index]=(preds[no])

                 no+=1

        i+=1

        



    mean_auc/=n

    print (" Average AUC: %f" % (mean_auc) )

    print (" printing train datasets ")

    printfilcsve(np.array(train_stacker), filename + ".train.csv")          



    # === Predictions === #

    # When making predictions, retrain the model on the whole training set

    preds=bagged_set(X, y,model, SEED, bagging, X_test, update_seed=True)  



    

    #create submission file 

    printfilcsve(np.array(preds), filename+ ".test.csv")  

    #save_results(preds, filename+"_submission_" +str(mean_auc) + ".csv")



if __name__ == '__main__':

    main()