import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from xgboost import plot_importance
import seaborn as sns
import os
# importing libraries

from sklearn.preprocessing import StandardScaler as ss
#dimentional reductionality

from sklearn.decomposition import PCA
#libraries for data splitting

from sklearn.model_selection import train_test_split
#libraries for model pipelining

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
#libraries for model parameter tuning

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from bayes_opt import BayesianOptimization
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
import eli5
from eli5.sklearn import PermutationImportance
#miscaleneous Libraries

import time
import random
import gc
from scipy.stats import uniform
import os
#Importing libraries for performance measures

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve
#loading dataset
pd.set_option("display.max_columns", 100)
os.chdir("../input")
data=pd.read_csv("winequalityN.csv")
#getting to know the data
data.head()
data.info()
data.shape
data.isnull().sum()
data.dropna(axis=0, inplace=True)
data.isnull().sum()
data.shape
data.head(3)
data.describe()
data.corr()
plt.figure(figsize=(12, 12))

sns.heatmap(data.corr(),annot=True,vmin=-1,cmap='YlGnBu')
sns.countplot(x = data.quality, data=data, hue='type', palette="rocket")
fig = plt.figure(figsize=(24,10))

features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "total sulfur dioxide", "chlorides", "fixed acidity", "citric acid","sulphates"]



for i in range(8):

    ax1 = fig.add_subplot(2,4,i+1)

    sns.barplot(x='quality', y=features[i],data=data, hue='type', palette='rocket')
#split data as predictors and target
X= data.iloc[ : , 1:14]
X.head(4)
y=data.iloc[:, 0]
y.head(4)
y=y.map({'white':1, 'red':0})
y.dtype
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True)
Xgb_pipelist = [('ss', ss() ),

            ('pca', PCA()),

            ('xg',  XGBClassifier(silent = False,

                                  n_jobs=2)        # Specify other parameters here

            )

            ]
Xgb_pipeline=Pipeline(Xgb_pipelist)
#parameter tuning



#grid search



parameters = {'xg__learning_rate':  [0.4, 0.05],

              'xg__n_estimators':   [100,  150],

              'xg__max_depth':      [3,5],

              'pca__n_components' : [5,7]

              }       
#    Create Grid Search object first with all necessary



grid_search = GridSearchCV(Xgb_pipeline,

                   parameters,         

                   n_jobs = 2,         

                   cv =2 ,             

                   verbose =2,      

                   scoring = ['accuracy', 'roc_auc'],  

                   refit = 'roc_auc'   

                   )
#fitting the data

start = time.time()

grid_search.fit(X_train, y_train)   

end = time.time()

(end - start)/60 
f"Best score: {grid_search.best_score_} "
f"Best parameter set {grid_search.best_params_}"
plt.bar(grid_search.best_params_.keys(), grid_search.best_params_.values(), color='b')

plt.xticks(rotation=10)
y_pred = grid_search.predict(X_test)

y_pred
accuracy = accuracy_score(y_test, y_pred)
f"Accuracy: {accuracy * 100.0}"
Xgb_pipelist = [('sts', ss() ),

            ('pca', PCA()),

            ('xg',  XGBClassifier(silent = False,

                                  n_jobs=2)        # Specify other parameters here

            )

            ]
Xgb_pipeline=Pipeline(Xgb_pipelist)
#random Search 



parameter_random = {'xg__learning_rate':  uniform(0, 1),

              'xg__n_estimators':   range(50,100),

              'xg__max_depth':      range(3,5),

              'pca__n_components' : range(5,7)}
random_search = RandomizedSearchCV(Xgb_pipeline,

                        param_distributions=parameter_random,

                        scoring= ['roc_auc', 'accuracy'],

                        n_iter=15,          

                        verbose = 3,

                        refit = 'roc_auc',

                        n_jobs = 2,          

                        cv = 2               

                        )
start = time.time()

random_search.fit(X_train, y_train)

end = time.time()

(end - start)/60
f"Best score: {random_search.best_score_} "
f"Best parameter set: {random_search.best_params_} "
plt.bar(random_search.best_params_.keys(), random_search.best_params_.values(), color='y')

plt.xticks(rotation=10)
y_pred = random_search.predict(X_test)

y_pred
accuracy = accuracy_score(y_test, y_pred)

f"Accuracy: {accuracy * 100.0}"
parameter_set = {

           'learning_rate':  (0, 1),                 

           'n_estimators':   (50,100),               

           'max_depth':      (3,5),                 

           'n_components' :  (5,7)          

            }
def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    #  Make pipeline. Pass parameters directly here

    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?

                              PCA(n_components=int(round(n_components))),

                              XGBClassifier(

                                           silent = False,

                                           n_jobs=2,

                                           learning_rate=learning_rate,

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )



    # Now fit the pipeline and evaluate

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_train,

                                y = y_train,

                                cv = 2,

                                n_jobs = 2,

                                scoring = 'f1'

                                ).mean()             # take the average of all results





    #  Finally return maximum/average value of result

    return cv_result
bayesian_opt = BayesianOptimization(

                             xg_eval,     

                             parameter_set  

                             )
start = time.time()
bayesian_opt.maximize(init_points=5,

               n_iter=15,        

               )
f"Best parameter set: {bayesian_opt.max} "
bayesian_opt.max.values()
for features in bayesian_opt.max.values(): 

    print(features)
features
plt.bar(features.keys(), features.values(), color='r')

plt.xticks(rotation=10)
#Fitting parameters into our model and Feature Importance
#Model with parameters of grid search

model_gs = XGBClassifier(

                    learning_rate = grid_search.best_params_['xg__learning_rate'],

                    max_depth = grid_search.best_params_['xg__max_depth'],

                    n_estimators=grid_search.best_params_['xg__n_estimators']

                    )



#Model with parameters of random search

model_rs = XGBClassifier(

                    learning_rate = random_search.best_params_['xg__learning_rate'],

                    max_depth = random_search.best_params_['xg__max_depth'],

                    n_estimators=random_search.best_params_['xg__n_estimators']

                    )



#Model with parameters of bayesian optimization

model_bo = XGBClassifier(

                    learning_rate = int(features['learning_rate']),

                    max_depth = int(features['max_depth']),

                    n_estimators=int(features['n_estimators'])

                    )
start = time.time()

model_gs.fit(X_train, y_train)

model_rs.fit(X_train, y_train)

model_bo.fit(X_train, y_train)

y_pred_gs = model_gs.predict(X_test)

y_pred_rs = model_rs.predict(X_test)

y_pred_bo = model_bo.predict(X_test)
accuracy_gs = accuracy_score(y_test, y_pred_gs)

accuracy_rs = accuracy_score(y_test, y_pred_rs)

accuracy_bo = accuracy_score(y_test, y_pred_gs)
print("Grid search Accuracy: "+str(accuracy_gs))

print("Grid search Accuracy: "+str(accuracy_rs))

print("Bayesian Optimization Accuracy: "+str(accuracy_bo))
model_gs.feature_importances_

model_rs.feature_importances_

model_bo.feature_importances_

plot_importance(model_gs)

plot_importance(model_rs)

plot_importance(model_bo)

plt.show()