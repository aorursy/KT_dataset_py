from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from catboost import Pool, CatBoostClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold



from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import xgboost as xgb

import pandas as pd

import numpy as np

import sklearn.metrics

from pathlib import Path

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import numpy as np

import lightgbm as lgb

    

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
root = Path("../input")

train = pd.read_csv(root.joinpath("kc_house_data.csv"))

train.head()
train.drop(columns=["id",'date'],inplace=True)



target='price'

X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['price']), train['price'], test_size=0.33, random_state=42)

X_train.reset_index(inplace =True,drop=True)

X_test.reset_index(inplace =True,drop=True)

y_train.reset_index(inplace =True,drop=True)

y_test.reset_index(inplace =True,drop=True)

X_train["price"]=y_train

X_test.head()

X_train.head()
def train_model_xgb(X_train,Y_train,X_val,Y_val,X_test,parmaters,features_name): 

    d_train = xgb.Dataset(X_train, Y_train,feature_names=features_name)

    d_valid = xgb.Dataset(X_val, Y_val,feature_names=features_name)

    d_test = xgb.DMatrix(X_test,feature_names=features_name)

    list_track = [(d_train, 'train'), (d_valid, 'valid')]

    model = xgb.train(parmaters, d_train, 2000,  list_track, maximize=False, verbose_eval=50, early_stopping_rounds=50)

    train_pred =model.predict(d_train)              

    valid_pred =model.predict(d_valid)   

    test_pred = model.predict(d_test)

    return train_pred ,valid_pred,test_pred,model

def train_model_LightGBM(X_train,Y_train,X_val,Y_val,X_test,parmaters): 

    d_train = lgb.Dataset(X_train,label=Y_train)

    d_valid = lgb.Dataset(X_val, label=Y_val)

    d_test = lgb.Dataset(X_test)

    

    model = lgb.train(params=parmaters, train_set=d_train, num_boost_round=2000,

                      valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=50)

    train_pred =model.predict(X_train)              

    valid_pred =model.predict(X_val)   

    test_pred = model.predict(X_test)

    return train_pred ,valid_pred,test_pred,model
def train_kfold_xgb(X_train,Y_train,X_test,parmaters,features_name,split=5):

    final_train_pred=np.zeros_like(Y_train)

    final_test_pred=np.zeros(len(X_test))

    

    kf = KFold(n_splits=split,random_state=2222)

    i=1

    for train_index, val_index in kf.split(X_train):

        print("fold:"+str(i))

        train_fold_features, val_fold_features = X_train.loc[train_index], X_train.loc[val_index]

        train_fold_target, val_fold_target = Y_train.loc[train_index], Y_train.loc[val_index] 

        train_pred ,valid_pred,test_pred,model=train_model_xgb( 

                                                        X_train=train_fold_features,

                                                        Y_train= train_fold_target,

                                                        X_val= val_fold_features,

                                                        Y_val= val_fold_target,

                                                        X_test= X_test,

                                                        parmaters=parmaters,

                                                        features_name=features_name 

                                                    )

        

        final_train_pred[val_index]=valid_pred

        final_test_pred=final_test_pred+test_pred/split

        i=i+1

    return final_train_pred,final_train_pred,model

def train_kfold_LightGBM(X_train,Y_train,X_test,parmaters,split=5):

    final_train_pred=np.zeros_like(Y_train)

    final_test_pred=np.zeros(len(X_test))

    

    kf = KFold(n_splits=split,random_state=2222)

    i=1

    for train_index, val_index in kf.split(X_train):

        print("fold:"+str(i))

        train_fold_features, val_fold_features = X_train.loc[train_index], X_train.loc[val_index]

        train_fold_target, val_fold_target = Y_train.loc[train_index], Y_train.loc[val_index] 

        train_pred ,valid_pred,test_pred,model=train_model_LightGBM( 

                                                        X_train=train_fold_features,

                                                        Y_train= train_fold_target,

                                                        X_val= val_fold_features,

                                                        Y_val= val_fold_target,

                                                        X_test= X_test,

                                                        parmaters=parmaters

                                                    )

        

        final_train_pred[val_index]=valid_pred

        final_test_pred=final_test_pred+test_pred/split

        i=i+1

    return final_train_pred,final_train_pred,model
#train_pred,test_pred,xgb_best_params,xgb_model=train_Xgboost(train_df=X_train,test_df=X_test,target='price',boosting_type='dart',metric='rmse')

#train_res,test_res=train_Xgboost(train_df=X_train,test_df=X_test,target='price',boosting_type='dart')

def train_Xgboost(train_df,test_df,target,Y,boosting_type='gbdt',metric='rmse') :

    

    import gc # garbej collector for mempry optimisation

    gc.enable()

    from sklearn.metrics import accuracy_score # to be changed in case of AUC,...

    from sklearn.metrics import roc_auc_score,mean_squared_error

    from sklearn.model_selection import train_test_split    

    from sklearn.metrics import mean_absolute_error



    X=train_df.drop(columns=[target])

    Y=train_df[target]

    #use this in case classification



    

    dtrain = xgb.DMatrix(X, label=Y)

    dtest = xgb.DMatrix(test_df)



  

    

    params = {

        # Parameters that we are going to tune.

        'booster': boosting_type,

        'max_depth':6,

        'min_child_weight': 1,

        'eta':.3,

        'subsample': 1,

        'colsample_bytree': 1,

        # Other parameters

        'objective':'reg:linear'

    }

    params['eval_metric'] = metric

    num_boost_round = 999

    

    

    

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics=metric,

        early_stopping_rounds=10

    )

    print('Best MAE with cv : '+str(cv_results['test-'+str(metric)+'-mean'].min()))

    

    

    

    print('--Tunning Parameters max_depth and min_child_weight--')

    # Define initial best params and MAE

    min_mae = float("Inf")

    best_params = None

    

    gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range(9,12)

    for min_child_weight in range(5,8)

    ]



    for max_depth, min_child_weight in gridsearch_params:

        print("CV with max_depth={}, min_child_weight={}".format(

                                 max_depth,

                                 min_child_weight))

        # Update our parameters

        params['max_depth'] = max_depth

        params['min_child_weight'] = min_child_weight

        # Run CV

        cv_results = xgb.cv(

            params,

            dtrain,

            num_boost_round=num_boost_round,

            seed=42,

            nfold=5,

            metrics=metric,

            early_stopping_rounds=10

        )

        # Update best MAE

        mean_mae = cv_results['test-'+str(metric)+'-mean'].min()

        boost_rounds = cv_results['test-'+str(metric)+'-mean'].argmin()

        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:

            min_mae = mean_mae

            best_params = (max_depth,min_child_weight)

        

    params['max_depth'] = best_params[0]

    params['min_child_weight'] = best_params[1]

    print('--Tunning Parameters subsample and colsample_bytree--')

    gridsearch_params = [

        (subsample, colsample)

        for subsample in [i/10. for i in range(7,11)]

        for colsample in [i/10. for i in range(7,11)]

    ]

    min_mae = float("Inf")

    best_params = None

    # We start by the largest values and go down to the smallest

    for subsample, colsample in reversed(gridsearch_params):

        print("CV with subsample={}, colsample={}".format(

                                 subsample,

                                 colsample))

        # We update our parameters

        params['subsample'] = subsample

        params['colsample_bytree'] = colsample

        # Run CV

        cv_results = xgb.cv(

            params,

            dtrain,

            num_boost_round=num_boost_round,

            seed=42,

            nfold=5,

            metrics=metric,

            early_stopping_rounds=10

        )

        # Update best score

        mean_mae = cv_results['test-'+str(metric)+'-mean'].min()

        boost_rounds = cv_results['test-'+str(metric)+'-mean'].argmin()

        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:

            min_mae = mean_mae

            best_params = (subsample,colsample)

    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

    params['subsample'] = best_params[0]

    params['colsample_bytree'] = best_params[1]

    best_params=0.1

    for eta in [.3, .2, .1, .05, .01, .005]:

        print("CV with eta={}".format(eta))

        # We update our parameters

        params['eta'] = eta

        # Run and time CV

        cv_results = xgb.cv(

            params,

            dtrain,

            num_boost_round=num_boost_round,

            seed=42,

            nfold=5,

            metrics=metric,

            early_stopping_rounds=10

              )

        # Update best score

        mean_mae = cv_results['test-'+str(metric)+'-mean'].min()

        boost_rounds = cv_results['test-'+str(metric)+'-mean'].argmin()

        print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:

            min_mae = mean_mae

            best_params = eta

    print("Best params: {}, MAE: {}".format(best_params, min_mae))

    params['eta'] = best_params

    print("Final Best params: {}".format(params))

    model = xgb.train(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        evals=[(dtest, "Test")],

        early_stopping_rounds=10

        )

    print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))



    final_train_pred,final_train_pred,model=train_kfold_xgb(X_train=X,Y_train=Y,X_test=test_df,parmaters=params,features_name=X.columns,split=5)



    return final_train_pred,final_train_pred,model



    
#train_res,test_res=train_Xgboost(train_df=X_train,test_df=X_test,Y=X_train["price"],target='price',boosting_type='dart')

def train_LightGBM(train_df,test_df,target,cat_features=[],num_boost_round_param=5000,

                   early_stopping_rounds_param=100,boosting_type='dart',metric='rmse') :

    

    import gc # garbej collector for mempry optimisation

    gc.enable()

    import lightgbm as lgb

    from sklearn.metrics import accuracy_score # to be changed in case of AUC,...

    from sklearn.metrics import roc_auc_score,mean_squared_error

    from sklearn.model_selection import train_test_split    

    from sklearn.metrics import mean_absolute_error



    X=train_df.drop(columns=[target])

    Y=train_df[target]

    #X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.20, random_state=42)

    #use this in case classification

    '''X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.33, stratify=Y,random_state=42)'''



    

    dtrain =lgb.Dataset(X, label=y_train)

    #dtest = lgb.Dataset(Y, label=y_test)



    

    # "Learn" the mean from the training data

    mean_train = np.mean(y_train)

    # Get predictions on the test set

    baseline_predictions = np.ones(y_test.shape) * mean_train

    # Compute MAE

    mae_baseline = mean_absolute_error(y_test, baseline_predictions)

    

    print("Baseline MAE is {:.2f}".format(mae_baseline))

    

    

  

    num_boost_round = num_boost_round_param

    params = {}

    params['learning_rate'] = 0.3

    params['boosting_type'] = boosting_type,

    params['objective'] = 'regression'

    params['metric'] = metric

 

   



    

    cv_results = lgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        stratified=False, # Default = TRUE !!!!

        categorical_feature=cat_features,

        nfold=5,

        metrics=metric,

        early_stopping_rounds=early_stopping_rounds_param

    )

    print('Best MAE with cv : '+str(np.min(cv_results[str(metric)+'-mean'])))    

    

    

    print('--Tunning Parameters max_depth and min_child_weight and num_leaves--')

    # Define initial best params and MAE

    min_mae = float("Inf")

    best_params = (-1,20,31)

    

    gridsearch_params = [

    (max_depth, min_data_in_leaf,num_leaves)

    for max_depth in range(7,13)

    for min_data_in_leaf in range(5,8)

    for num_leaves in  [5,10,20,50]

    ]



    for max_depth, min_data_in_leaf,num_leaves in gridsearch_params:

        print("CV with max_depth={}, min_data_in_leaf={},  num_leaves={}".format(

                                 max_depth,

                                 min_data_in_leaf,

                                num_leaves))

        # Update our parameters

        params['max_depth'] = max_depth

        params['min_data_in_leaf'] = min_data_in_leaf

        params['num_leaves'] = num_leaves

        # Run CV

        cv_results = lgb.cv(

                    params,

                    dtrain,

                    num_boost_round=num_boost_round,

                    seed=42,

                    stratified=False, # Default = TRUE 

                    categorical_feature=cat_features,

                    nfold=5,

                    metrics=metric,

                    early_stopping_rounds=early_stopping_rounds_param

        )

        # Update best MAE

        mean_mae = np.min(cv_results[str(metric)+'-mean'])

        boost_rounds = np.argmin(cv_results[str(metric)+'-mean'])

        print("\t"+str(metric)+" {} for {} rounds".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:

            min_mae = mean_mae

            best_params = (max_depth,min_data_in_leaf,num_leaves)

        

    params['max_depth'] = best_params[0]

    params['min_data_in_leaf'] = best_params[1]

    params['num_leaves'] = best_params[2]

    

    

    

    

    print('--Tunning Parameters feature_fraction and bagging_fraction--')

    

    

    # Define initial best params and MAE

    min_mae = float("Inf")

    best_params = (1.0,1.0)

    

    gridsearch_params = [

    (bagging_fraction, feature_fraction)

    for bagging_fraction in [0.6,0.7,0.8,0.9]

    for feature_fraction in [0.6,0.7,0.8,0.9]

    ]



    for feature_fraction, bagging_fraction in gridsearch_params:

        print("CV with bagging_fraction={}, feature_fraction={}, ".format(

                                 bagging_fraction,

                                 feature_fraction

                                ))

        # Update our parameters

        params['feature_fraction'] = feature_fraction

        params['bagging_fraction'] = bagging_fraction

        

        # Run CV

        cv_results = lgb.cv(

                    params,

                    dtrain,

                    num_boost_round=num_boost_round,

                    seed=42,

                    stratified=False, # Default = TRUE 

                    categorical_feature=cat_features,

                    nfold=5,

                    metrics=metric,

                    early_stopping_rounds=early_stopping_rounds_param

        )

        # Update best MAE

        mean_mae = np.min(cv_results[str(metric)+'-mean'])

        boost_rounds = np.argmin(cv_results[str(metric)+'-mean'])

        print("\t"+str(metric)+" {} for {} rounds".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:

            min_mae = mean_mae

            best_params = (feature_fraction,bagging_fraction)

        

    params['feature_fraction'] = best_params[0]

    params['bagging_fraction'] = best_params[1]

    

    

    

    

    

    print('--Tunning Parameter Learning rate --')

 

    

    # Define initial best params and MAE

    min_mae = float("Inf")

    best_params = 0.3



    for learning_rate in [ .2, .1, .05, .01, .005]:

        print("CV with Learning Rate={} ".format(

                                 learning_rate

                                ))

        # Update our parameters

        params['learning_rate'] = learning_rate

        

        # Run CV

        cv_results = lgb.cv(

                    params,

                    dtrain,

                    num_boost_round=num_boost_round,

                    seed=42,

                    stratified=False, # Default = TRUE 

                    categorical_feature=cat_features,

                    nfold=5,

                    metrics=metric,

                    early_stopping_rounds=early_stopping_rounds_param

        )

        # Update best MAE

        mean_mae = np.min(cv_results[str(metric)+'-mean'])

        boost_rounds = np.argmin(cv_results[str(metric)+'-mean'])

        print("\t"+str(metric)+" {} for {} rounds".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:

            min_mae = mean_mae

            best_params = learning_rate

        

    params['learning_rate'] = best_params

    

    

    

        

    print('--Tunning Parameter min_data_in_leaf and lambda_l1, lambda_l2  --')

 

    

    # Define initial best params and MAE

    min_mae = float("Inf")

    best_params = (20,0,0)



    gridsearch_params = [

    (min_data_in_leaf , lambda_l1,lambda_l2)

    for min_data_in_leaf in [30,40,50] 

    for lambda_l1 in [0.1,0.2]

    for lambda_l2 in  [0.1,0.2]

    ]



    for min_data_in_leaf, lambda_l1,lambda_l2 in gridsearch_params:

        print("CV with min_data_in_leaf={}, lambda_l1={},lambda_l2={}  ".format(

                                 bagging_fraction,

                                 lambda_l1,

                                 lambda_l2

                                ))

        # Update our parameters

        params['min_data_in_leaf'] = min_data_in_leaf

        params['lambda_l1'] = lambda_l1

        params['lambda_l2'] = lambda_l2

        

        # Run CV

        cv_results = lgb.cv(

                    params,

                    dtrain,

                    num_boost_round=num_boost_round,

                    seed=42,

                    stratified=False, # Default = TRUE 

                    categorical_feature=cat_features,

                    nfold=5,

                    metrics=metric,

                    early_stopping_rounds=early_stopping_rounds_param

        )

        # Update best MAE

        mean_mae = np.min(cv_results[str(metric)+'-mean'])

        boost_rounds = np.argmin(cv_results[str(metric)+'-mean'])

        print("\t"+str(metric)+" {} for {} rounds".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:

            min_mae = mean_mae

            best_params = (min_data_in_leaf, lambda_l1,lambda_l2)

        

    params['min_data_in_leaf'] = best_params[0]

    params['lambda_l1'] = best_params[1]

    params['lambda_l2'] = best_params[2]

    

    

    

    #last : max_bin

    print('--Tunning Parameters max_bin --')

    # Define initial best params and MAE

    min_mae = float("Inf")

    best_params = 63

    for max_bin in [63,128,256]:

        print("CV with max_bin={}".format(max_bin))

        # We update our parameters

        params['max_bin'] = max_bin

        # Run CV

        cv_results = lgb.cv(

                    params,

                    dtrain,

                    num_boost_round=num_boost_round,

                    seed=42,

                    stratified=False, # Default = TRUE 

                    categorical_feature=cat_features,

                    nfold=5,

                    metrics=metric,

                    early_stopping_rounds=early_stopping_rounds_param

        )

         # Update best MAE

        mean_mae = np.min(cv_results[str(metric)+'-mean'])

        boost_rounds = np.argmin(cv_results[str(metric)+'-mean'])

        print("\t"+str(metric)+" {} for {} rounds".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:

            min_mae = mean_mae

            best_params = max_bin

        

    params['max_bin'] = best_params

    

    print("Final Best params: {}".format(params))

    cv_results = lgb.cv(

                    params,

                    dtrain,

                    num_boost_round=num_boost_round,

                    seed=42,

                    stratified=False, # Default = TRUE 

                    categorical_feature=cat_features,

                    nfold=5,

                    metrics=metric,

                    early_stopping_rounds=early_stopping_rounds_param

        )

    # Update best MAE

    mean_mae = np.min(cv_results[str(metric)+'-mean'])

    boost_rounds = np.argmin(cv_results[str(metric)+'-mean'])

    print("\t"+str(metric)+" {} for {} rounds".format(mean_mae, boost_rounds))

    



    final_train_pred,final_train_pred,model=train_kfold_LightGBM(X_train=X,Y_train=Y,

                                                           X_test=test_df,parmaters=params,

                                                  split=5)



    return final_train_pred,final_train_pred,model

    



    
#train_res,test_res,lgb_model=train_LightGBM(train_df=X_train,test_df=X_test,target='price',boosting_type='gbdt')

lgb_model
def train_Catboost(train_df,test_df,target,cat_features=[],num_boost_round_param=5000,

                   early_stopping_rounds_param=100,boosting_type='dart',metric='rmse') :

    

    import gc # garbej collector for mempry optimisation

    gc.enable()

    import catboost as cat





    print(format('How to find optimal parameters for CatBoost using GridSearchCV for Regression','*^82'))    

    

    import warnings

    warnings.filterwarnings("ignore")

    

    # load libraries

    from sklearn import datasets

    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import GridSearchCV

    import catboost 



    X=train_df.drop(columns=[target])

    y=train_df[target]





    model = CatBoostRegressor(eval_metric=metric,

                              iterations=num_boost_round_param,

                              boosting_type=boosting_type,

                              categorical_feature=cat_features,

                              task_type = "GPU")

    parameters = {'depth'         : [6,8,10,13,16],

                  'learning_rate' : [0.01, 0.05, 0.1],

                  'l2_leaf_reg':[0,2,4,8],

                  'bagging_temperature':[0,1,5,10,20],

                  

                 }

    grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 5, n_jobs=-1)

    grid.fit(X_train, y_train)    



    # Results from Grid Search

    print("\n========================================================")

    print(" Results from Grid Search " )

    print("========================================================")    

    

    print("\n The best estimator across ALL searched params:\n",

          grid.best_estimator_)

    

    print("\n The best score across ALL searched params:\n",

          grid.best_score_)

    

    print("\n The best parameters across ALL searched params:\n",

          grid.best_params_)

    

    print("\n ========================================================")

    

    

    final_train_pred,final_train_pred,model=train_kfold_Catboost(X_train=X,Y_train=Y,

                                                           X_test=test_df,parmaters=best_params_,

                                                  split=5)



    return final_train_pred,final_train_pred,model

    

    



    
#train_res,test_res,lgb_model=train_Catboost(train_df=X_train,test_df=X_test,target='price',boosting_type='gbdt')
