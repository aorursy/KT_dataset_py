import pandas as pd

import numpy as np

import lightgbm as lgb

import eli5

from eli5.sklearn import PermutationImportance

from tsfresh.feature_extraction import feature_calculators

from sklearn.model_selection import train_test_split



params={'bagging_fraction': 0.6364049179265991,

        'bagging_freq': 17,

        'feature_fraction': 0.8780002461376601,

        'min_data_in_leaf': 100,

        'num_leaves': 65,

        'boost': 'gbdt',

        'learning_rate': 0.01,

        'max_depth': -1,

        'metric': 'mae',

        'num_threads': 4,

        'tree_learner': 'serial',

        'objective': 'huber',

        'n_estimators': 100000}
def cook_data(data):

    output=pd.Series()

    

    if "time_to_failure" in data.columns:

        output["target"]=data["time_to_failure"].iloc[-1]

    

    data=data["acoustic_data"].values

    output["std"]=data.std()

    

    #Limit the range.

    output["new_std"]=data[np.logical_and(0<=data,data<=10)].std()

    

    #This feature is from public kernel.

    output["numpeaks_10"]=feature_calculators.number_peaks(data,10)

    return output



def create_X_y():

    reader = pd.read_csv("../input/train.csv",chunksize=150000)

    train=pd.DataFrame( [cook_data(r) for r in reader] )

    y=train.pop("target")

    X=train

    return X,y



#80% for fit.

#10% for early stopping.

#10% for cv.

X,y=create_X_y()

(X_fit, _X,y_fit, _y) = train_test_split(X, y,train_size=0.8,test_size=0.2,random_state=0)

(X_cv,X_es,y_cv,y_es) = train_test_split(_X, _y,train_size=0.5,test_size=0.5,random_state=0)



model = lgb.LGBMRegressor(**params)

model.fit(X_fit,y_fit,eval_set = [(X_es,y_es)],verbose = 5000,early_stopping_rounds=1000)

perm = PermutationImportance(model, random_state=1).fit(X_cv,y_cv)

eli5.show_weights(perm, feature_names = X_cv.columns.tolist())
def create_prediction():

    model = lgb.LGBMRegressor(**params)

    model.fit(X_fit,y_fit,eval_set = [(X_es,y_es)],verbose = 5000,early_stopping_rounds=1000)

    submission=pd.read_csv('../input/sample_submission.csv')

    predictions=[cook_data(pd.read_csv('../input/test/'+s+'.csv')) for s in submission["seg_id"]]

    submission["time_to_failure"]=model.predict(pd.DataFrame(predictions),num_iteration=model.best_iteration_)

    submission.to_csv("submission.csv",index=False)



create_prediction()