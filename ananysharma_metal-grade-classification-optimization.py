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
df_train = pd.read_csv("../input/metal-furnace-dataset/Train.csv")
df_test = pd.read_csv("../input/metal-furnace-dataset/Test.csv")
df_train[:5]
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,8)
sns.countplot(x=df_train["f0"],hue=df_train['grade'])
plt.xticks(rotation = 90)
plt.show()
df_train.drop(['f9'],inplace = True,axis=1)
df_test.drop(['f9'],inplace=True,axis=1)
cols = df_train.columns
n_rows = 9
n_cols = 3
# plt.xlabel(fontsize=12
for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols = n_cols,figsize = (16,8))
    for j in range(n_cols):
        sns.violinplot(y = cols[i*n_cols+j],data  = df_train,ax = ax[j])
cols = df_train.columns
n_cols  = 2
n_rows = 14
for i in range(n_rows):
    fg,ax = plt.subplots(nrows = 1,ncols = n_cols,figsize = (17,6))
    for j in range(n_cols):
        sns.countplot(x = cols[i*n_cols+j],hue = 'grade',data = df_train,ax = ax[j])
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb
import lightgbm as lgb
df_train.shape,df_test.shape
df_train.grade.value_counts()
df_train.isna().sum()
plt.rcParams['figure.figsize'] = (14,8)
sns.countplot(df_train["grade"],hue = df_train["grade"],palette = 'dark')
plt.title("Grade Distribution",fontsize = 20)
plt.xlabel("Grade",fontsize = 15)
plt.ylabel("Count",fontsize = 15)
plt.show()
features = list(set(df_train.columns)-set(['grade','f9']))
target = 'grade'
len(features)
def metric(y,y0):
    return log_loss(y,y0)
def cross_valid(model,train,features,target,cv):
    results = cross_val_predict(model, train[features], train[target], method="predict_proba",cv=cv)
    return metric(train[target],results)

models = [lgb.LGBMClassifier(), xgb.XGBClassifier(), GradientBoostingClassifier(), LogisticRegression(), 
              RandomForestClassifier(), AdaBoostClassifier()
             ]
for i in models:
    error =  cross_valid(i,df_train,features,target,5)
    print(str(i).split("(")[0], error)
def xgb_model(train, features, target, plot=True):    
    evals_result = {}
    trainX, validX, trainY, validY = train_test_split(train[features], train[target], test_size=0.2, random_state=13)
    print("XGB Model")
    
    dtrain = xgb.DMatrix(trainX, label=trainY)
    dvalid = xgb.DMatrix(validX, label=validY)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    MAX_ROUNDS=2000
    early_stopping_rounds=100
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'learning_rate': 0.01,
        'num_round': MAX_ROUNDS,
        'max_depth': 8,
        'seed': 25,
        'nthread': -1,
        'num_class':5
    }
    
    model = xgb.train(
        params,
        dtrain,
        evals=watchlist,
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50
        #feval=metric_xgb
    
    )
    
    print("Best Iteration :: {} \n".format(model.best_iteration))
    
    
    if plot:
        # Plotting Importances
        fig, ax = plt.subplots(figsize=(24, 24))
        xgb.plot_importance(model, height=0.4, ax=ax)
xgb_model(df_train, features, target, plot=True)

xgb1 = xgb.XGBClassifier(
    booster='dart',
    objective='multi:softprob',
    learning_rate= 0.01,
    num_round= 775,
    max_depth=8,
    seed=25,
    nthread=3,
    eval_metric='mlogloss',
    num_class=5

)
trainX, validX, trainY, validY = train_test_split(df_train[features], df_train[target], test_size=0.2,stratify=df_train[target], random_state=13)
model  = xgb1
cross_valid(model,df_train,features,target,5)
model = xgb1
model.fit(trainX[features],trainY)
y_pred_valid = model.predict_proba(validX[features])
print("Validation Score:",metric(validY,y_pred_valid))
y_pred_test = model.predict(df_test[features])
result1 = pd.DataFrame(y_pred_test)
result1[:5]
y_pred_test = model.predict_proba(df_test[features])
result = pd.DataFrame(y_pred_test)
result
result1
X= df_train[features]
y = df_train[target]
trainX, validX, trainY, validY = train_test_split(df_train[features], df_train[target], test_size=0.2,stratify=df_train[target], random_state=13)
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import decomposition
from sklearn import pipeline


classifier = ensemble.RandomForestClassifier(n_jobs = -1)

param_grid = {
    "n_estimators":[100,200,300,400],
    "max_depth":[1,3,5,7],
    "criterion":["gini","entropy"]
}
model = model_selection.GridSearchCV(
    estimator = classifier,
    param_grid  = param_grid,
    scoring = "accuracy",
    verbose = 10,
    n_jobs = 1,
    cv=5
)

model.fit(trainX,trainY)
print(model.best_score_)
print(model.best_estimator_.get_params())


from sklearn.metrics import classification_report
preds = model.predict(validX)
print(metrics.accuracy_score(preds,validY))
print(classification_report(preds,validY))
from sklearn import preprocessing
pca = decomposition.PCA() 
rf  = ensemble.RandomForestClassifier(n_jobs = -1)
scl = preprocessing.StandardScaler()
classifier1 = pipeline.Pipeline([("scaling",scl),("pca",pca),("rf",rf)])

param_grid = {
    "pca__n_components":np.arange(10,15),
    "rf__n_estimators":np.arange(100,1500,100),
    "rf__max_depth":np.arange(1,20),
    "rf__criterion": ["gini","entropy"]
}

model = model_selection.RandomizedSearchCV(
    estimator = classifier1,
    param_distributions = param_grid,
    n_iter = 10,
    scoring = 'accuracy',
    verbose = 10,
    n_jobs = 1,
    cv = 5
)

model.fit(trainX,trainY)
print(model.best_score_)
print(model.best_estimator_.get_params())
from sklearn.metrics import classification_report
preds = model.predict(validX)
print(metrics.accuracy_score(preds,validY))
print(classification_report(preds,validY))
def optimize(params,param_name,x,y):
    params = dict(zip(param_name,params))
    model = ensemble.RandomForestClassifier(**params)
    kf =model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x,y=y):
        train_idx , test_idx = idx[0],idx[1]
        xtrain = X[train_idx]
        ytrain = y[train_idx]
        
        xtest = X[test_idx]
        ytest = y[test_idx]
        
        model.fit(xtrain,ytrain)
        preds = model.predict(xtest)
        accuracies.append(metrics.accuracy_score(preds,ytest))
    return -1.0 * np.mean(accuracies)  

from functools import partial
from skopt import space
from skopt import gp_minimize

trainX, validX, trainY, validY = train_test_split(df_train[features], df_train[target], test_size=0.2,stratify=df_train[target], random_state=13)

param_space = [
    space.Integer(3,15,name = "max_depth"),
    space.Integer(100,600,name = "n_estimators"),
    space.Categorical(["gini","entropy"],name = "criterion"),
    space.Real(0.01,1,prior = 'uniform', name = "max_features")
    
]
param_names = [
    "max_depth",
    "n_estimators",
    "criterion",
    "max_features"
]
optim_func = partial(
    optimize,
    param_name = param_names,
    x=trainX,
    y = trainY
)


result = gp_minimize(
            optim_func,
            dimensions = param_space,
            n_calls = 15,
            n_random_starts = 10,
            verbose = 10,
)

print(dict(zip(param_names,result.x)))
