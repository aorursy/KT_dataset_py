# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
train.info()
train["Age"] = train["Age"].fillna(int(np.mean(train["Age"])))
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
train["FamilySize"] = train['SibSp'] + train['Parch']
train["IsAlone"] = train["FamilySize"].apply(lambda x: 1 if x == 0 else 0)
train["Title"] = train["Name"].apply(lambda x: x.split(",")[1].split(".")[0])
train["Title"] = train["Title"].str.strip()
imp_titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev']
train["Title"] = train["Title"].apply(lambda x: x if x in imp_titles else "misc")
train["actualFare"] = train["Fare"]/train["FamilySize"].replace(0,1)

def get_age_bin(x):
    if x<=10:
        return "kids"
    elif x>10  and x<=20:
        return "teens"
    elif x>20 and x<=40:
        return "Adults"
    elif x>40 and x<=65:
        return "Mid Age"
    else:
        return "Old"
    
train["AgeGroup"] = train["Age"].apply(get_age_bin)
import seaborn as sns
plt = train.groupby("AgeGroup")["Survived"].mean().reset_index()
sns.lineplot(x = plt["AgeGroup"], y = plt["Survived"])

grpby = train.groupby(["Pclass", "IsAlone", "AgeGroup"])["Survived"].agg(["count", "sum"]).reset_index()
grpby["per"] = grpby["sum"]/grpby["count"]
grpby = grpby.sort_values(by = "per", ascending=False)
grpby
grpby["dict_key"] = grpby["Pclass"].astype(str) + "#" + grpby["IsAlone"].astype(str) + "#" + grpby["AgeGroup"].astype(str)
grpby_dict = {i:j for i,j in zip(grpby["dict_key"], grpby["per"])}
def get_combi(x):
    global grpby_dict
    comb = str(x["Pclass"]) + "#" + str(x["IsAlone"]) + "#" + str(x["AgeGroup"])
    if grpby_dict[comb]<=0.70:
        return 0
#    elif grpby_dict[comb]>0.30 and grpby_dict[comb]<=0.50:
#        return 1
#    elif grpby_dict[comb]>0.50 and grpby_dict[comb]<=0.80:
#        return 2
    else:
        return 1
    
train["combi"] = train[["Pclass", "IsAlone", "AgeGroup"]].apply(lambda x: get_combi(x), axis=1)

pd.crosstab(train["Survived"], [train["Pclass"], train["IsAlone"]])
!pip install sweetviz
import sweetviz as sv
report = sv.analyze(train, "Survived")
report.show_html("/kaggle/working/EDA.html")
#cols = ['Pclass', 'Sex', 'Age', 'Fare', "FamilySize", "HaveCabin", "Title", "actualFare"]
cols = ['Pclass', 'Sex', 'AgeGroup', "Title", "actualFare", "combi"]
cols_to_encode = train[cols].select_dtypes(include=['object']).copy()
cols_to_encode = list(cols_to_encode.columns)
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test["Age"] = test["Age"].fillna(int(np.mean(test["Age"])))
test["Fare"] = test["Fare"].fillna(np.mean(test["Fare"]))
test["FamilySize"] = test['SibSp'] + test['Parch']
test["IsAlone"] = test["FamilySize"].apply(lambda x: 1 if x == 0 else 0)
test["HaveCabin"] = test["Cabin"].fillna("").apply(lambda x: 0 if x == "" else 1)
test["Title"] = test["Name"].apply(lambda x: x.split(",")[1].split(".")[0])
test["Title"] = test["Title"].str.strip()
test["Title"] = test["Title"].apply(lambda x: x if x in imp_titles else "misc")
test["actualFare"] = test["Fare"]/test["FamilySize"].replace(0,1)
test["AgeGroup"] = test["Age"].apply(get_age_bin)
test["combi"] = test[["Pclass", "IsAlone", "AgeGroup"]].apply(lambda x: get_combi(x), axis=1)
from category_encoders.m_estimate import MEstimateEncoder
from sklearn.model_selection import train_test_split
MEE_encoder = MEstimateEncoder()

def get_x_and_y(train, test, y_col="Survived"):
    global cols, MEE_encoder
    y = train[y_col]
    train = train[cols].copy()
    test = test[cols].copy()
    train_mee = MEE_encoder.fit_transform(train, y) 
    test_mee = MEE_encoder.transform(test)
    return train_mee, test_mee


train1, test1 = get_x_and_y(train, test)
X = train1.copy()
Y = train['Survived'].copy()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
from sklearn import svm
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
np.mean(y_pred == y_test)
from sklearn.model_selection import GridSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import catboost as cat
from lightgbm import LGBMClassifier


def random_forest(train_x, train_y, test_x, test_y):
    rf = RandomForestClassifier()

    # Create the parameter grid based on the results of random search 
    rf_grid = {
        'max_depth': [50, 70, 110],
        'max_features': [3, 4, 6],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [3,4,6],
        'n_estimators': [100, 300, 2000, 1000]
    }


    grid_search = GridSearchCV(estimator = rf, param_grid = rf_grid,
                              cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(train_x, train_y)

    parameters = grid_search.best_params_
    
    print(parameters)
    rf = RandomForestClassifier(**parameters)
    rf.fit(train_x, train_y)
    y_pred = rf.predict(test_x)
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    sns.barplot(x = np.arange(len(indices)), y = importances[indices])
    return np.mean(y_pred == test_y),


def xg_boost(train_x, train_y, test_x, test_y):
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, objective = 'binary:logistic',
                        colsample_bytree = 0.5, max_depth = 10, alpha = 1, gamma = 0, booster = 'gbtree')

    xgb_grid = {'n_estimators':[50, 100, 200, 500],
                'learning_rate':[0.01, 0.1, 0.25,  0.75],
                'colsample_bytree':[0.1, 0.3, 0.5],
                'max_depth':[8, 12, 14],
                'alpha': [1, 2, 3],
                'objective': ['binary:logistic'],
                'gamma':[0],
                'booster': ['gbtree']}

    grid_search = GridSearchCV(estimator = xgb, param_grid = xgb_grid,
                                  cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(train_x, train_y)

    parameters = grid_search.best_params_
    
    print(parameters)
    xgb = XGBClassifier(**parameters)
    xgb.fit(train_x, train_y)
    y_pred = xgb.predict(test_x)
    return np.mean(y_pred == test_y)

def cat_boost(train_x, train_y, test_x, test_y):
    ctb = cat.CatBoostClassifier(learning_rate=0.75, n_estimators=100,
                                 subsample=0.5, loss_function='Logloss',
                                 depth = 8, l2_leaf_reg = 2, bagging_temperature = 1.0,
                                 )
    
    cat_grid = {'n_estimators': [100, 500, 1000],
                'depth': [1, 3, 7],
                'learning_rate': [0.1, 0.01, 0.05],
                'bagging_temperature': [1.0],
                'l2_leaf_reg': [2, 30],
                'scale_pos_weight': [0.01, 1.0],
                'subsample':[0.1, 0.3, 0.5],
                'loss_function' : ['Logloss']}


    grid_search = GridSearchCV(estimator = ctb, param_grid = cat_grid,
                              cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(train_x, train_y)
    
    parameters = grid_search.best_params_
    
    print(parameters)
    ctb = cat.CatBoostClassifier(**parameters)
    ctb.fit(train_x, train_y)
    y_pred = ctb.predict(test_x)
    return np.mean(y_pred == test_y)

def light_gbm(train_x, train_y, test_x, test_y):
    lgbm = LGBMClassifier()
    lgbm_grid = {'n_estimators': [100, 500, 1000],
                 'boosting_type':['rf', 'gbdt'],
                'max_depth': [1, 3, 7],
                'learning_rate': [0.01, 0.1,0.25],
                'objective': [ 'binary'],
                'reg_alpha': [0.1, 0.5],
                'reg_lambda': [0.1, 0.5],
                'subsample':[0.1, 0.3, 0.5]
                }


    grid_search = GridSearchCV(estimator = lgbm, param_grid = lgbm_grid,
                              cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(train_x, train_y)
    
    parameters = grid_search.best_params_
    print(parameters)
    
    lgbm = LGBMClassifier(**parameters)
    lgbm.fit(x_train, y_train)
    y_pred = lgbm.predict(x_test)
    return np.mean(y_pred == y_test)

random_forest(x_train, y_train, x_test, y_test)
importances = rf.feature_importances_
indices = np.argsort(importances)
xg_boost(x_train, y_train, x_test, y_test)
cat_boost(x_train, y_train, x_test, y_test)
light_gbm(x_train, y_train, x_test, y_test)
train_x.columns
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Sequential()
model.add(Dense(27, input_dim=8, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200)
y_pred = model.predict(test_x)
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy:', test_acc)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
np.mean(y_pred == y_test)
from vecstack import stacking
from sklearn.metrics import accuracy_score

seed = 0

base_models = [ctb, rf, lr]
S_train, S_test = stacking(base_models,                # list of base models
                           train_x, train_y, test_x,   # data
                           regression = False,          # We need regression - set to True)
                                                       
                           mode = 'oof_pred_bag',      # mode: oof for train set, predict test 
                                                       # set in each fold and vote
                           needs_proba = False,        # predict class labels (if you need 
                                                       # probabilities - set to True) 
                           save_dir = None,            # do not save result and log (to save 
                                                       # in current dir - set to '.')
                           metric = accuracy_score,# metric: callable
                           n_folds = 5,               # number of folds
                           stratified = False,         # stratified split for folds
                           shuffle = True,             # shuffle the data
                           random_state =  seed,       # ensure reproducibility
                           verbose = 1)    

super_learner = xgb
super_learner.fit(S_train, train_y)
Stack_pred = super_learner.predict(S_test)
test1.info()
S_train, S_test = stacking(base_models,                # list of base models
                           train1, train["Survived"], test1,   # data
                           regression = False,          # We need regression - set to True)
                                                       
                           mode = 'oof_pred_bag',      # mode: oof for train set, predict test 
                                                       # set in each fold and vote
                           needs_proba = False,        # predict class labels (if you need 
                                                       # probabilities - set to True) 
                           save_dir = None,            # do not save result and log (to save 
                                                       # in current dir - set to '.')
                           metric = accuracy_score,# metric: callable
                           n_folds = 5,               # number of folds
                           stratified = False,         # stratified split for folds
                           shuffle = True,             # shuffle the data
                           random_state =  seed,       # ensure reproducibility
                           verbose = 1)    

super_learner = xgb
super_learner.fit(train1, train["Survived"])
Stack_pred = super_learner.predict(test1)
from sklearn.model_selection import GridSearchCV
from skopt.space import Real, Categorical, Integer
# Create the parameter grid based on the results of random search 


xgb_grid = {'n_estimators':[50, 100, 150, 300],
            'learning_rate':[0.1, 0.01, 0.05],
            'colsample_bytree':[0.1, 0.3, 0.5],
            'max_depth':[10, 12, 14],
            'alpha': [1, 2, 3]}

cat_grid = {'n_estimators': [100, 500, 1000],
            'depth': [1, 7, 8],
            'learning_rate': [0.1, 0.01, 0.05],
            'bagging_temperature': [0.0, 1.0],
            'border_count': [1, 255],
            'l2_leaf_reg': [2, 30],
            'scale_pos_weight': [0.01, 1.0]}


grid_search = GridSearchCV(estimator = ctb, param_grid = cat_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(train_x, train_y)

grid_search.best_params_
test["predictions"] = pd.Series(Stack_pred)
test["predictions"].isnull().sum()

submissionFile = test[["PassengerId", "predictions"]].copy()
submissionFile = submissionFile.rename(columns = {"predictions": "Survived"})
submissionFile.to_csv("/kaggle/working/predictions.csv", index = False)
