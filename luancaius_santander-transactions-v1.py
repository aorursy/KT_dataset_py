import pandas as pd 

import numpy as np

from time import time

import seaborn as sns



inputTrain = pd.read_csv('../input/train.csv')

inputTest = pd.read_csv('../input/test.csv')
from imblearn.over_sampling import SMOTE

def oversampling(train):

    target = train['target']

    train = train.drop(['target'], axis=1)

    train['target'] = target

    features, target = SMOTE(n_jobs=-1, random_state=42).fit_sample(train.drop(['target'], axis=1), train['target'])

    target[target >= 0.5] = 1

    target[target < 0.5] = 0

    finalArray = np.column_stack((features, target))

    columns = train.columns.copy()

    train = pd.DataFrame(finalArray, columns=columns).reset_index(drop=True)

    return train



def replacingMissingValues(dataset):

    for i in dataset.columns:

        if dataset[i].isnull().sum() > 0:

            print(i," => ",dataset[i].isnull().sum())

            dataset[i].fillna(np.mean(dataset[i]), inplace=True)

    return dataset 

def dropColumns(data):

    dropColumns = []

    if 'ID_code' in data.columns:

        dropColumns.append('ID_code')

    if 'target' in data.columns:

        dropColumns.append('target')

    print('droped columns ',dropColumns)

    features = data.drop(dropColumns, axis = 1)   

    return features

def getCategories(features):

    categories = []

    for category in categories: 

        features[category] = features[category].astype("category").cat.codes

    return features
def getTarget(data):

    return data['target']

def getFeatures(data):

    features = dropColumns(data)

    features  = getCategories(features)

    replacingMissingValues(features)

    return features

def prepareData(data):

    target = getTarget(data)

    features = getFeatures(data)

    print(features.shape)

    return features,target
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import f1_score

   

def train_predict(clf, X_train, y_train, X_test, y_test):

    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    clf.fit(X_train, y_train)

    y_pred = clf.predict(features)

    return clf.score(features, y_pred)     

def split_data(features):

    n_splits = 15

    folds = KFold(n_splits=n_splits, random_state=42)

    return enumerate(folds.split(features))

def training(clf, features, target):

    score = 0

    start_fold = time()

    for fold, (train_idx, test_idx) in split_data(features):

        print("\nFold ", fold)

        X_train = features.iloc[train_idx]

        y_train = target.iloc[train_idx]

        X_test = features.iloc[test_idx]

        y_test = target.iloc[test_idx]            

        score = train_predict(clf, X_train, y_train, X_test, y_test)

        print(score)        

    end_fold = time()

    print('Training folds in {:.4f}'.format(end_fold - start_fold))

    return score 

def training_lgm( features, target):

    param = {

        'bagging_freq': 5,

        'bagging_fraction': 0.38,

        'boost_from_average':'false',

        'boost': 'gbdt',

        'feature_fraction': 0.045,

        'learning_rate': 0.01,

        'max_depth': -1,  

        'metric':'auc',

        'min_data_in_leaf': 80,

        'min_sum_hessian_in_leaf': 10.0,

        'num_leaves': 13,

        'num_threads': 8,

        'tree_learner': 'serial',

        'objective': 'binary', 

        'verbosity': 1

    }

    start_fold = time()

    for fold, (train_idx, test_idx) in split_data(features):

        print("\nFold ", fold)

        X_train = features.iloc[train_idx]

        y_train = target.iloc[train_idx]

        X_test = features.iloc[test_idx]

        y_test = target.iloc[test_idx]            

        train_data = lightgbm.Dataset(X_train, label=y_train)

        test_data = lightgbm.Dataset(X_test, label=y_test)

        model = lightgbm.train(param,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=5000,

                       early_stopping_rounds=100)        

    end_fold = time()

    print('Training folds in {:.4f}'.format(end_fold - start_fold))        

    return model
def submissionFile(clf):

    print('Creating submission file')

    columnId = 'ID_code'

    columnTarget = 'target'

    sub = pd.DataFrame(inputTest[columnId], columns=[columnId,columnTarget])

    features=getFeatures(inputTest)

    pred = clf.predict(features) 

    sub[columnTarget]=pred

    print('submit_{}.csv'.format(clf.__class__.__name__))

    sub.to_csv('submit_{}.csv'.format(clf.__class__.__name__), index=False)

def submission2(predictions):

    submission = pd.DataFrame({"ID_code": test_df.ID_code.values})

    submission["target"] = predictions

    submission.to_csv("submission_lightgbm.csv", index=False)
from sklearn.metrics import accuracy_score, log_loss

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from xgboost import XGBClassifier

import lightgbm



clfs = [

    lightgbm

#    LogisticRegression(),

#    XGBClassifier(silent=False,scale_pos_weight=1,learning_rate=0.01,colsample_bytree = 0.4,subsample = 0.8,objective='binary:logistic',n_estimators=1000,reg_alpha = 0.3,max_depth=4,gamma=10)

]

print("Starting")

start_init = time()



train = inputTrain.drop('ID_code', axis=1)

print("Starting oversampling")

train2 = oversampling(train)

end_over = time()

print("Ending oversampling in {}",(end_over - start_init))



features, target = prepareData(train2)

for clf in clfs:

   model=training_lgm(features, target)

   submissionFile(model)

   #clf=tuning(clf,features, target, score)    

   #submissionFile(clf)

end_init = time()

print("Finished in {:.4f} seconds".format(end_init - start_init))