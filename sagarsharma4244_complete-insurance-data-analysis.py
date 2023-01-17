# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
sns.set(style='whitegrid')

train = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")
test = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/test.csv")
train.head()
# WITH ONE COMMAND
# Get column dtypes
# Contains isnull or not
# Total Entries
train.info()
train.shape
train.isnull().sum()
sns.countplot(train.Response)
sns.catplot(x="Gender", y="Annual_Premium",hue="Response", data=train)
sns.scatterplot(x="Vehicle_Damage", y="Policy_Sales_Channel",hue="Response", data=train)
sns.scatterplot(x="Previously_Insured", y="Policy_Sales_Channel",hue="Response", data=train)
sns.scatterplot(x="Annual_Premium", y="Policy_Sales_Channel",hue="Response", data=train)
# sns.swarmplot(x="Annual_Premium", y="Policy_Sales_Channel",hue="Response", data=train)
sns.distplot(train.Age)
sns.scatterplot(x="Age", y="Annual_Premium",hue="Response", data=train)
# sns.swarmplot(x="Age", y="Annual_Premium",hue="Response", data=train)
sns.scatterplot(x="Age", y="Previously_Insured",hue="Response", data=train)
sns.scatterplot(x="Vehicle_Age", y="Policy_Sales_Channel",hue="Response", data=train)

sns.countplot(x="Vehicle_Age", hue="Response", data=train)
sns.countplot(x="Age", hue="Response", data=train)
sns.countplot(x="Vehicle_Age", hue="Response", data=train)
sns.countplot(x="Previously_Insured", hue="Response", data=train)
sns.countplot(x="Driving_License", hue="Response", data=train)
sns.countplot(x="Region_Code", hue="Response", data=train)
sns.scatterplot(x="Vintage",y="Region_Code" , hue="Response", data=train)
sns.pairplot(data=train)

train_os=RandomOverSampler(random_state=42)
X=train.drop(['Response'],axis=1)
y=train['Response']

X_os,y_os=train_os.fit_sample(X,y)
from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_os))) 
X_os['Gender'] = X_os['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
# Convert categorical variable into dummy/indicator variables
# Learn More https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html?highlight=dummies#pandas.get_dummies
X_os=pd.get_dummies(X_os,drop_first=True)
from sklearn.model_selection import train_test_split
x = X_os.drop(labels= ['id','Region_Code',"Driving_License"], axis = 1)

y=y_os

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)

x_train.shape
x_train.info()
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score,confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 
x_train.shape
x_train
random_search = {'criterion': ['entropy', 'gini'],
               'max_depth': [2,3,4,5,6,7,10],
               'min_samples_leaf': [4, 6, 8],
               'min_samples_split': [5, 7,10],
               'n_estimators': [300]}

clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 10, 
                               cv = 4, verbose= 2, random_state= 101, n_jobs = -1)
model.fit(x_train,y_train)
filename = 'rf_model.sav'
pickle.dump(model, open(filename, 'wb'))
filename = 'rf_model.sav'
rf_load = pickle.load(open(filename, 'rb'))
y_pred=model.predict(x_test)
print (classification_report(y_test, y_pred))

y_score = model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

title('Random Forest ROC curve: CC Fraud')
xlabel('FPR (Precision)')
ylabel('TPR (Recall)')

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))

roc_auc_score(y_test, y_score)
space={ 'max_depth': hp.quniform("max_depth", 3,18,1),
        'gamma': hp.uniform ('gamma', 1,11),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 300,
        'seed': 0
    }
def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']),learning_rate=0.01,gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [(x_train, y_train), (x_test, y_test)]
    
    clf.fit(x_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(x_test)
    y_score = model.predict_proba(x_test)[:,1]
    accuracy = accuracy_score(y_test, pred>0.5)
    Roc_Auc_Score = roc_auc_score(y_test, y_score)
    print ("ROC-AUC Score:",Roc_Auc_Score)
    print ("SCORE:", accuracy)
    return {'loss': -Roc_Auc_Score, 'status': STATUS_OK }
x_train=x_train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
x_test=x_test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)
print("The best hyperparameters are : ","\n")
print(best_hyperparams)
xgb_model=xgb.XGBClassifier(n_estimators = space['n_estimators'], max_depth = 7, gamma = 4.0388607178326605, reg_lambda = 0.26955899476862166,
                            reg_alpha = 66.0, min_child_weight=4.0,colsample_bytree = 0.8844758548525424 )
    

xgb_model.fit(x_train,y_train)
filename = 'xgboost_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))
y_score = xgb_model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

title('XGBoost ROC curve')
xlabel('FPR (Precision)')
ylabel('TPR (Recall)')

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
x_test['Vehicle_Age_lt_1_Year']=x_test['Vehicle_Age_lt_1_Year'].astype('int')
x_test['Vehicle_Age_gt_2_Years']=x_test['Vehicle_Age_gt_2_Years'].astype('int')
x_test['Vehicle_Damage_Yes']=x_test['Vehicle_Damage_Yes'].astype('int')

random_state=42
n_iter=50
num_folds=2
kf = KFold(n_splits=num_folds, random_state=random_state,shuffle=True)
def gb_mse_cv(params, random_state=random_state, cv=kf, X=x_train, y=y_train):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']), 
              'max_depth': int(params['max_depth']), 
             'learning_rate': params['learning_rate']}
    
    # we use this params to create a new LGBM Regressor
    model = lgb.LGBMClassifier(random_state=42, **params)
    
    # and then conduct the cross validation with the same folds as before
    score = -cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    return score

%%time

# possible values of parameters
space={'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
       'max_depth' : hp.quniform('max_depth', 2, 20, 1),
       'learning_rate': hp.loguniform('learning_rate', -5, 0)
      }

# trials will contain logging information
trials = Trials()

best=fmin(fn=gb_mse_cv, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=n_iter, # maximum number of iterations
          trials=trials, # logging
          rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility
         )

# computing the score on the test set
model = lgb.LGBMClassifier(random_state=random_state, n_estimators=int(best['n_estimators']),
                      max_depth=int(best['max_depth']),learning_rate=best['learning_rate'])
model.fit(x_train,y_train)

preds = [pred[1] for pred in model.predict_proba(x_test)]
score = roc_auc_score(y_test, preds, average = 'weighted')
print("Best auc-roc score",score)

y_score = model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

title('LGBM ROC curve: CC Fraud')
xlabel('FPR (Precision)')
ylabel('TPR (Recall)')

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
