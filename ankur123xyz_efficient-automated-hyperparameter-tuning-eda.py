!pip install -U scikit-learn==0.23
!pip install scikit-optimize==0.8.1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from skopt import gp_minimize,space
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score
from skopt.utils import use_named_args
from hyperopt import hp,Trials,tpe,fmin
from hyperopt.pyll.base import scope
from hyperopt.plotting import main_plot_history
import optuna
import warnings
warnings.filterwarnings("ignore")
dataset = pd.read_csv("../input/heart-disease-uci/heart.csv")
dataset.shape
dataset.head()
dataset.isnull().sum()
f , ax = plt.subplots()
plt.pie(dataset["sex"].value_counts(),explode=[0,.1],labels=["Male","Female"],startangle=90,shadow=True,autopct = '%1.1f%%')
f,ax = plt.subplots(figsize=(10,7))
sns.countplot("sex",hue="target",data=dataset)
bars = ax.patches
half = int(len(bars)/2)
ax.set_xticklabels(["female","male"])
ax.legend(["absence","presence"])
for first,second in (zip(bars[:half],bars[half:])):
    height1= first.get_height()
    height2= second.get_height()
    total = height1 + height2
    ax.text(first.get_x()+first.get_width()/2,height1+2,'{0:.0%}'.format(height1/total),ha="center")
    ax.text(second.get_x()+second.get_width()/2,height2+2,'{0:.0%}'.format(height2/total),ha="center")
dataset.loc[:,"age_band"] = pd.cut(dataset.age,bins=[25,35,45,60,80])
f,ax = plt.subplots(figsize=(10,8))
sns.countplot("age_band",hue="target",data=dataset)
bars = ax.patches
half = int(len(ax.patches)/2)
ax.legend(["absence","presence"])

for first,second in zip(bars[:half],bars[half:]):
    height1 =  first.get_height()
    height2 = second.get_height()
    total_height= height1+height2
    ax.text(first.get_x()+first.get_width()/2, height1+1,'{0:.0%}'.format(height1/total_height), ha ='center')
    ax.text(second.get_x()+second.get_width()/2, height2+1,'{0:.0%}'.format(height2/total_height), ha ='center')
f,ax= plt.subplots()
sns.countplot("age_band",hue="sex",data=dataset)
ax.legend(["female","male"])

f,ax = plt.subplots(figsize=(10,7))
sns.boxplot("target","chol",data=dataset)
ax.set_xticklabels(["absence","presence"])
y= dataset["target"]
dataset.drop(["target","age_band"],axis=1,inplace=True)
X_train,X_test,y_train,y_test = train_test_split(dataset,y,test_size=0.3,random_state=42)
param_space_skopt =[
    space.Integer(3,10,name="max_depth"),
    space.Integer(50,1000,name="n_estimators"),
    space.Categorical(["gini","entropy"],name="criterion"),
    space.Real(0.1,1,name="max_features"),
    space.Integer(2,10,name="min_samples_leaf")
]

model = RandomForestClassifier()

@use_named_args(param_space_skopt)
def objective_skopt(**params_skopt):
    model.set_params(**params_skopt)
    skf = StratifiedKFold(n_splits=5,random_state=42)
    scores = -np.mean(cross_val_score(model,X_train,y_train,cv=skf,scoring="accuracy"))
    return scores
result = gp_minimize(objective_skopt,dimensions= param_space_skopt, n_calls=25, n_random_starts=10,verbose=10,random_state=42)
-result.fun
from skopt.plots import plot_convergence
plot_convergence(result)
model_skopt =RandomForestClassifier(n_estimators= result.x[1],criterion=result.x[2],max_depth=result.x[0],min_samples_leaf=result.x[4],max_features=result.x[3],random_state=42)
model_skopt.fit(X_train,y_train)
y_pred_skopt = model_skopt.predict(X_test)
skopt_score = accuracy_score(y_test,y_pred_skopt)
skopt_score
param_space_hopt = {
    "max_depth":scope.int(hp.quniform("max_depth",3,10,1)),
              "n_estimators":scope.int(hp.quniform("n_estimators",50,1000,1)),
               "criterion":hp.choice("criterion",["gini","entropy"]),
               "max_features":hp.uniform("max_features",0.1,1),
               "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",2,10,1))
              }

def objective_hopt(params_hopt):
    model_hopt = RandomForestClassifier(**params_hopt)
    skf = StratifiedKFold(n_splits=5,random_state=42)
    scores = -np.mean(cross_val_score(model_hopt,X_train,y_train,cv=skf,scoring="accuracy"))
    return scores

trial_hopt = Trials()
hyopt = fmin(fn=objective_hopt,space = param_space_hopt, algo=tpe.suggest,max_evals=25,trials=trial_hopt) 
hyopt
main_plot_history(trial_hopt)
model_hopt =RandomForestClassifier(n_estimators= int(hyopt["n_estimators"]),criterion="gini",max_depth=int(hyopt["max_depth"]),min_samples_leaf=int(hyopt["min_samples_leaf"]),max_features=hyopt["max_features"],random_state=42)
model_hopt.fit(X_train,y_train)
y_pred_hyopt = model_hopt.predict(X_test)
hyopt_score = accuracy_score(y_test,y_pred_hyopt)
hyopt_score
def optimization_optuna(trial_optuna):
    
    n_estimators = trial_optuna.suggest_int("n_estimators",50,1000)
    max_depth = trial_optuna.suggest_int("max_depth",3,10)
    criterion = trial_optuna.suggest_categorical("criterion",["entropy","gini"])
    min_samples_split = trial_optuna.suggest_int("min_samples_leaf",2,10)
    max_features = trial_optuna.suggest_uniform("max_features",0.1,1)
    

    model_optuna = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,criterion=criterion,
                                         min_samples_split=min_samples_split,max_features=max_features)
    skf = StratifiedKFold(n_splits=5)
    score = cross_val_score(model_optuna,X_train,y_train,cv=skf,scoring="accuracy")
    return np.mean(score)
study = optuna.create_study(direction="maximize")
result = study.optimize(optimization_optuna,n_trials=25)
study.best_params
model_optuna =RandomForestClassifier(n_estimators= study.best_params["n_estimators"],criterion=study.best_params["criterion"],max_depth=study.best_params["max_depth"],min_samples_leaf=study.best_params["min_samples_leaf"],max_features=study.best_params["max_features"],random_state=42)
model_optuna.fit(X_train,y_train)
y_pred_optuna = model_optuna.predict(X_test)
optuna_score = accuracy_score(y_test,y_pred_optuna)
optuna_score
optuna.visualization.plot_optimization_history(study)