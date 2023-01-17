import numpy as np 
import pandas as pd 

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#train = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
test = pd.read_csv('/kaggle/input/DontGetKicked/test.csv')
def create_Folds(myFolds):
    #Get Data
    df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
    
    #Assign default value
    df['kFold']  = -1
    
    #Get y 
    y = df['IsBadBuy']
    
    #initiate kfolds class 
    kf = StratifiedKFold(n_splits = myFolds)
    
    #fill kfolds column with value
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kFold'] = f
    
    #Export data with kfold column
    df.to_csv('train_kfold.csv')
create_Folds(5)
df = pd.read_csv('./train_kfold.csv')
#Folds have equal proportion of data
df[df.kFold == 3].IsBadBuy.value_counts()
def run_model(fold):
    #Load Data
    df = pd.read_csv('./train_kfold.csv')
    
    #Divide into Train and Validation 
    train = df[df.kFold != fold].reset_index(drop = True)
    validation = df[df.kFold == fold].reset_index(drop = True)
    
    #Extract Train
    y_train = train['IsBadBuy']
    X_train = train.drop(['IsBadBuy','RefId','kFold'], axis = 1)
    
    #Extract Validation
    y_valid = validation['IsBadBuy']
    X_valid = validation.drop(['IsBadBuy','RefId','kFold'], axis = 1)
    
    #Divide Features into Numercial and Categorical
    numerical_features = [c for c, dtype in zip(X_train.columns, X_train.dtypes)
                     if dtype.kind in ['i','f']]
    categorical_features = [c for c, dtype in zip(X_train.columns, X_train.dtypes)
                     if dtype.kind not in ['i','f']]
    
    #Create preproecessor Pipeline for Numericals and Categorical
    preprocessor = make_column_transformer(
    
    (make_pipeline(
    SimpleImputer(strategy = 'median'),
        StandardScaler(),
    KBinsDiscretizer(n_bins=3)), numerical_features),
    
    (make_pipeline(
    SimpleImputer(strategy = 'constant', fill_value = 'missing'),
    OneHotEncoder(categories = 'auto', handle_unknown = 'ignore')), categorical_features),
    
    )
    
    #Create Random Forest Pipeline
    RF_Model = make_pipeline(preprocessor, RandomForestClassifier())
    
    #Fit Model
    RF_Model.fit(X_train,y_train)
    
    #Predict Validation Scores
    train_preds = RF_Model.predict_proba(X_train)[:,1]
    valid_preds = RF_Model.predict_proba(X_valid)[:,1]
    
    #Get AUC
    train_auc = roc_auc_score(y_train,train_preds)
    valid_auc = roc_auc_score(y_valid,valid_preds)
    
    
    print('---------')
    print(f'Train Score : {train_auc:.3f}')
    print(f'Valid Score : {valid_auc:.3f}')
    
    print(f'Train AUC : {RF_Model.score(X_train, y_train):.3f}')
    print(f'Test AUC : {RF_Model.score(X_valid, y_valid):.3f}')
    
run_model(1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
grid_param =   {"randomforestclassifier__n_estimators": [10, 100, 200, 300, 400, 500, 600, 700],
                 "randomforestclassifier__max_depth":[5,8,15,25,30,None],
                 "randomforestclassifier__min_samples_leaf":[1,2,5,10,15,100],
                 "randomforestclassifier__max_leaf_nodes": [2, 5,10]}
RF_Model = make_pipeline(preprocessor, RandomForestClassifier())
randonSearch = RandomizedSearchCV(RF_Model, grid_param, cv=5, verbose=0,n_jobs=-1, scoring="accuracy") # Fit grid search
best_model = randonSearch.fit(X_train,y_train)
best_model
print(f'Train : {best_model.score(X_train, y_train):.3f}')
print(f'Test : {best_model.score(X_test, y_test):.3f}')
sub_test = test.drop(['RefId'], axis = 1)
sub_test_pred = best_model.predict(sub_test).astype(int)
AllSub = pd.DataFrame({ 'RefId': test['RefId'],
                       'IsBadBuy' : sub_test_pred
    
})

AllSub.to_csv("DGK_Pipeline_RF_RGS.csv", index = False)