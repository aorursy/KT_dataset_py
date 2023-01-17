import numpy as np 
import pandas as pd 
from datetime import datetime
from matplotlib import pyplot

import os
print(os.listdir("../input"))
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


train = pd.read_csv("../input/train.csv")
sub  = pd.read_csv("../input/test.csv")
train.head()
print("train:", train.shape, "test:", sub.shape, sep="\n")
y = train.y
train = train.iloc[:,1:20]
sub = sub.iloc[:,1:20]
des = pd.concat((train, sub))
des.head()
from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(train.iloc[:,14:20])
des.iloc[:,14:20] = std_scale.transform(des.iloc[:,14:20])
des.loc[des.Dias_Ultima_Camp >= 0, "Dias_Ultima_Cat"] = "0 a 5"
des.loc[des.Dias_Ultima_Camp >= 5, "Dias_Ultima_Cat"] = "5 a 10"
des.loc[des.Dias_Ultima_Camp >= 10, "Dias_Ultima_Cat"] = "10 a 20"
des.loc[des.Dias_Ultima_Camp >= 25, "Dias_Ultima_Cat"] = "20 a 50"
des.loc[des.Dias_Ultima_Camp >= 50, "Dias_Ultima_Cat"] = "50 a 100"
des.loc[des.Dias_Ultima_Camp >= 100, "Dias_Ultima_Cat"] = "> 100"
des.loc[des.Dias_Ultima_Camp >= 999, "Dias_Ultima_Cat"] = "Nunca"

des = des.drop(["Dias_Ultima_Camp"], axis = 1)
des.loc[des.Educacion == "illiterate", "Educacion_Num"] = 1
des.loc[des.Educacion == "basic.4y", "Educacion_Num"] = 2
des.loc[des.Educacion == "basic.6y", "Educacion_Num"] = 5
des.loc[des.Educacion == "basic.9y", "Educacion_Num"] = 8
des.loc[des.Educacion == "high.school", "Educacion_Num"] = 15
des.loc[des.Educacion == "unknown", "Educacion_Num"] = 19
des.loc[des.Educacion == "university.degree", "Educacion_Num"] = 30
des.loc[des.Educacion == "professional.course", "Educacion_Num"] = 40

des = des.drop(["Educacion"], axis = 1)
des.loc[des.Estado_Civil == "single", "Estado_Civil_Num"] = 1
des.loc[des.Estado_Civil == "married", "Estado_Civil_Num"] = 5
des.loc[des.Estado_Civil == "divorced", "Estado_Civil_Num"] = 2
des.loc[des.Estado_Civil == "unknown", "Estado_Civil_Num"] = 3.5

des = des.drop(["Estado_Civil"], axis = 1)
des.loc[des.Tipo_Trabajo == "student", "Tipo_Trabajo_Num"] = 1
des.loc[des.Tipo_Trabajo == "unemployed", "Tipo_Trabajo_Num"] = 2
des.loc[des.Tipo_Trabajo == "housemaid", "Tipo_Trabajo_Num"] = 4
des.loc[des.Tipo_Trabajo == "self-employed", "Tipo_Trabajo_Num"] = 6
des.loc[des.Tipo_Trabajo == "blue-collar", "Tipo_Trabajo_Num"] = 9
des.loc[des.Tipo_Trabajo == "technician", "Tipo_Trabajo_Num"] = 10
des.loc[des.Tipo_Trabajo == "retired", "Tipo_Trabajo_Num"] = 12
des.loc[des.Tipo_Trabajo == "services", "Tipo_Trabajo_Num"] = 15
des.loc[des.Tipo_Trabajo == "entrepreneur", "Tipo_Trabajo_Num"] = 20
des.loc[des.Tipo_Trabajo == "admin.", "Tipo_Trabajo_Num"] = 27
des.loc[des.Tipo_Trabajo == "management", "Tipo_Trabajo_Num"] = 30
des.loc[des.Tipo_Trabajo == "unknown", "Tipo_Trabajo_Num"] = 16

des = des.drop(["Tipo_Trabajo"], axis = 1)
des.loc[des.Consumo == "no", "Consumo_Num"] = -1
des.loc[des.Consumo == "yes", "Consumo_Num"] = 1
des.loc[des.Consumo == "unknown", "Consumo_Num"] = -0.7

des = des.drop(["Consumo"], axis = 1)
des.loc[des.Vivienda == "no", "Vivienda_Num"] = -1
des.loc[des.Vivienda == "yes", "Vivienda_Num"] = 1
des.loc[des.Vivienda == "unknown", "Vivienda_Num"] = 0.06

des = des.drop(["Vivienda"], axis = 1)
des["Score"] = des.Tipo_Trabajo_Num*des.Educacion_Num*des.Estado_Civil_Num
des["Score1"] = des.Tipo_Trabajo_Num*des.Educacion_Num
des["Score2"] = des.Tipo_Trabajo_Num*des.Estado_Civil_Num
des["Score3"] = des.Educacion_Num*des.Estado_Civil_Num
des["minScore"] = des[["Tipo_Trabajo_Num","Estado_Civil_Num","Educacion_Num"]].min(axis=1)
des["maxScore"] = des[["Tipo_Trabajo_Num","Estado_Civil_Num","Educacion_Num"]].max(axis=1)
des["meanScore"] = des[["Tipo_Trabajo_Num","Estado_Civil_Num","Educacion_Num"]].mean(axis=1)
des.head()
des = pd.get_dummies(des)

X = des.iloc[0:32967,:]
X_sub = des.iloc[32967:41188,:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4)
X_train.head()
params = {
    "learning_rate"    : [0.04, 0.05, 0.06, 0.07, 0.1],
    "max_depth"        : [4, 5, 6, 7, 8, 9, 10],
    "min_child_weight" : [6, 7, 8, 9, 10, 11, 12],
    "gamma"            : [0.38, 0.39, 0.4, 0.41, 0.42],
    "colsample_bytree" : [0.5, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75]
}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
import xgboost
folds = 4

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

classifier = xgboost.XGBClassifier(tree_method='gpu_hist', nthread=6, n_estimators = 300)

random_search = RandomizedSearchCV(classifier, param_distributions = params, n_iter = 200, scoring = "roc_auc", n_jobs = -1, cv = skf.split(X_train,y_train), verbose = 3)
#start_time = timer(None)
random_search.fit(X_train, y_train)
#timer(start_time)
random_search.best_estimator_
# Bayesian Optimization
from bayes_opt import BayesianOptimization

#def xgbc_cv(max_delta_step,subsample,colsample_bylevel,reg_lambda,reg_alpha,scale_pos_weight):

def xgbc_cv(learning_rate,min_child_weight,max_depth,colsample_bytree,gamma,n_estimators):
    
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    estimator_function = xgboost.XGBClassifier(learning_rate= learning_rate,
                                           min_child_weight= int(min_child_weight),
                                           max_depth=int(max_depth),
                                           #max_delta_step= int(max_delta_step),
                                           #subsample= subsample,
                                           colsample_bytree= colsample_bytree,
                                           #colsample_bylevel= colsample_bylevel,
                                           #reg_lambda= reg_lambda,
                                           #reg_alpha = reg_alpha,
                                           gamma= gamma,
                                           n_estimators= int(n_estimators),
                                           #scale_pos_weight= scale_pos_weight,
                                           nthread = -1,
                                           objective='binary:logistic',
                                           seed = seed)
    
    scores = cross_val_score(estimator_function, X_train, y_train, scoring = "roc_auc", n_jobs = -1, cv = 5)
    
    # return the mean validation score to be maximized 
    return scores.mean()
# alpha is a parameter for the gaussian process
# Note that this is itself a hyperparemter that can be optimized.
gp_params = {"alpha": 1e-10}

seed = 112 # Random seed

hyperparameter_space = {   
    'learning_rate': (0.01, 0.1),
    'min_child_weight': (5, 15),
    'max_depth': (5, 20),
    #'max_delta_step': (0, 20),
    #'subsample': (0,1.0),
    'colsample_bytree': (0.01, 1.0),
    #'colsample_bylevel': (0.01, 1.0),
    #'reg_lambda': (0, 2.0),
    #'reg_alpha': (0, 1.0),
    'gamma': (0.1, 5),
    'n_estimators': (100, 250),
    #'scale_pos_weight': (0, 2.0)
}

xgbcBO = BayesianOptimization(f = xgbc_cv, 
                             pbounds =  hyperparameter_space,
                             random_state = seed,
                             verbose = 10)

xgbcBO.maximize(init_points=10,n_iter=20,acq='ucb', kappa= 3, **gp_params)
#classifier1 = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.7, gamma=0.4,
#              learning_rate=0.04, max_delta_step=0, max_depth=4,
#              min_child_weight=12, missing=None, n_estimators=300, n_jobs=1,
#              nthread=6, objective='binary:logistic', random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#              silent=None, subsample=1, tree_method='gpu_hist', verbosity=1)
classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.42,
              learning_rate=0.04, max_delta_step=0, max_depth=4,
              min_child_weight=10, missing=None, n_estimators=300, n_jobs=1,
              nthread=6, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, tree_method='gpu_hist', verbosity=1)
seed = 112

classifier = xgboost.XGBClassifier(learning_rate= 0.04556,
                                       min_child_weight= int(6.292),
                                       max_depth=int(5.165),
                                       max_delta_step= 0,
                                       subsample= 1,
                                       colsample_bytree= 0.3664,
                                       colsample_bylevel= 1,
                                       reg_lambda= 1,
                                       reg_alpha = 0,
                                       gamma= 4.776,
                                       n_estimators= 300,
                                       scale_pos_weight= 1,
                                       nthread = -1,
                                       objective='binary:logistic',
                                   
                                       seed = seed)
submission = pd.read_csv('../input/sampleSubmission.csv')
ids = submission['ID'].values
eval_set = [(X_train, y_train), (X_test, y_test)]
classifier.fit(X_train, y_train, eval_metric="auc", eval_set=eval_set, verbose=10)
results = classifier.evals_result()
epochs = len(results['validation_0']["auc"])
x_axis = range(0, epochs)
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']["auc"], label='Train')
ax.plot(x_axis, results['validation_1']["auc"], label='Test')
ax.legend()
pyplot.ylabel('ROC AUC')
pyplot.title('ROC AUC')
pyplot.show()
from xgboost import plot_importance

pyplot.rcParams["figure.figsize"] = (20,10)

plot_importance(classifier)
pyplot.show()
y_sub = classifier.predict_proba(X_sub)
output = pd.DataFrame({"ID": ids, "y": y_sub[:,1]})
output.to_csv("submission.csv", index=False)