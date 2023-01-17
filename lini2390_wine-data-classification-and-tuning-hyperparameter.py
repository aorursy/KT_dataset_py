#data analysis

import numpy as np

import pandas as pd
#plotting

import matplotlib.pyplot as plt

from xgboost import plot_importance

import seaborn as sns
#data pre-processing

from sklearn.preprocessing import StandardScaler as ss
#for dimensionality reduction

from sklearn.decomposition import PCA
#data splitting and model parameter search

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
#for performance measures

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve
#for pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline
#modeling 

from xgboost.sklearn import XGBClassifier
#model pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

#for Bayes optimization

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization

#Importing Miscelaneous libraries

import time

import os

import gc

import random

from scipy.stats import uniform
#Find feature importance of ANY BLACK BOX estimator

import eli5

from eli5.sklearn import PermutationImportance
#to display 100 rows

pd.set_option('display.max_columns', 100)
#working directory selection

os.chdir("../input") 

data = pd.read_csv("winequalityN.csv")
total_lines = 6497

num_lines = 0.99 * total_lines

num_lines 
p = num_lines/total_lines
p
data = pd.read_csv(

         tr_f,

         header=0,   

         skiprows=lambda i: (i>0) and (random.random() > p)

         )
data.shape
data.columns
data.dtypes
data.describe()
data.head()
data.info()
data.isnull().any()
data.isnull().sum()
data.dropna(axis=0, inplace=True)

data.shape
data.head(2)
data.isnull().any()
data['quality'].unique()
data.corr(method ='pearson') 
plt.figure(figsize=(10, 10))

sns.heatmap(data.corr(method ='pearson') ,annot=True,vmin=-1)
data.corr(method ='kendall')
plt.figure(figsize=(10, 10))

sns.heatmap(data.corr(method ='kendall') ,annot=True,vmin=-1)
#correlation between features

axes = pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (20,10), diagonal = 'kde' ,s=80)

corr = data.corr().values



plt.xticks(fontsize =10,rotation =0)

plt.yticks(fontsize =10)

for ax in axes.ravel():

    ax.set_xlabel(ax.get_xlabel(),fontsize = 15, rotation = 60)

    ax.set_ylabel(ax.get_ylabel(),fontsize = 15, rotation = 60)

# put the correlation between each pair of variables on each graph

for i, j in zip(*np.triu_indices_from(axes, k=1)):

    axes[i, j].annotate("%.3f" %corr[i, j], (0.8, 0.8), xycoords="axes fraction", ha="center", va="center")

#data exploration using pair plots

g = sns.pairplot(data, diag_kind="kde", markers="+",hue='type',

               plot_kws=dict(s=50, edgecolor="b", linewidth=1),

                 diag_kws=dict(shade=True))
#dividing datas into predictors

X = data.iloc[ :, 1:13]

X.head(2)
#target

y = data.iloc[ : , 0]

y.head(2)
#Transform label data to '1' and '0'

y = y.map({'white':1, 'red' : 0})

y.dtype
colnames = X.columns.tolist()
#Split dataset into train and validation parts

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.35,

                                                    shuffle = True

                                                    )

X_train.shape
X_test.shape
y_train.shape
y_test.shape
steps_xg = [('sts', ss() ),

            ('pca', PCA()),

            ('xg',  XGBClassifier(silent = False,

                                  n_jobs=2)        

            )

            ]

pipe_xg = Pipeline(steps_xg)
parameters = {'xg__learning_rate':  [0.03, 0.05], # learning rate decides what percentage

                                                  #  of error is to be fitted by

                                                  #   by next boosted tree.

                                                  # See this answer in stackoverflow:

                                                  # https://stats.stackexchange.com/questions/354484/why-does-xgboost-have-a-learning-rate

                                                  # Coefficients of boosted trees decide,

                                                  #  in the overall model or scheme, how much importance

                                                  #   each boosted tree shall have. Values of these

                                                  #    Coefficients are calculated by modeling

                                                  #     algorithm and unlike learning rate are

                                                  #      not hyperparameters. These Coefficients

                                                  #       get adjusted by l1 and l2 parameters

              'xg__n_estimators':   [100,  150],  # Number of boosted trees to fit

                                                  # l1 and l2 specifications will change

                                                  # the values of coeff of boosted trees

                                                  # but not their numbers



              'xg__max_depth':      [3,5],

              'pca__n_components' : [7,12]

              }                               # Total: 2 * 2 * 2 * 2

clf = GridSearchCV(pipe_xg,            # pipeline object

                   parameters,         # possible parameters

                   n_jobs = 2,         # USe parallel cpu threads

                   cv =2 ,             # No of folds

                   verbose =2,         # Higher the value, more the verbosity

                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance

                   refit = 'roc_auc'   # Refitting final model on what parameters?

                                       # Those which maximise auc

                   )
start = time.time()

clf.fit(X_train, y_train)

end = time.time()

(end - start)/60               
f"Best score: {clf.best_score_} "

f"Best parameter set {clf.best_params_}"

y_pred = clf.predict(X_test)

y_pred
accuracy = accuracy_score(y_test, y_pred)

f"Accuracy: {accuracy * 100.0}"



perm = PermutationImportance(

                            clf,

                            random_state=1

                            )

start = time.time()

perm.fit(X_test, y_test)

end = time.time()

(end - start)/60
eli5.show_weights(

                  perm,

                  feature_names = colnames      # X_test.columns.tolist()

                  )

fw = eli5.explain_weights_df(

                  perm,

                  feature_names = colnames      # X_test.columns.tolist()

                  )



# Print importance

fw
# Tune parameters using randomized search

# Hyperparameters to tune and their ranges

parameters = {'xg__learning_rate':  uniform(0, 1),

              'xg__n_estimators':   range(50,100),

              'xg__max_depth':      range(3,5),

              'pca__n_components' : range(5,7)}
#     Create the object first

rs = RandomizedSearchCV(pipe_xg,

                        param_distributions=parameters,

                        scoring= ['roc_auc', 'accuracy'],

                        n_iter=15,          # Max combination of

                                            # parameter to try. Default = 10

                        verbose = 3,

                        refit = 'roc_auc',

                        n_jobs = 2,          # Use parallel cpu threads

                        cv = 2               # No of folds.

                                             # So n_iter * cv combinations

                        )
start = time.time()

rs.fit(X_train, y_train)

end = time.time()

(end - start)/60
f"Best score: {rs.best_score_} "
f"Best score: {rs.best_score_} "
y_pred_rs = rs.predict(X_test)
accuracy_rs = accuracy_score(y_test, y_pred_rs)

f"Accuracy: {accuracy_rs * 100.0}"
#baya

para_set = {

           'learning_rate':  (0, 1),                 

           'n_estimators':   (50,100),               

           'max_depth':      (3,5),                 

           'n_components' :  (5,7)          

            }
#    Create a function that when passed some parameters

#    evaluates results using cross-validation

#    This function is used by BayesianOptimization() object



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
#      Instantiate BayesianOptimization() object

#      This object  can be considered as performing an internal-loop

#      i)  Given parameters, xg_eval() evaluates performance

#      ii) Based on the performance, set of parameters are selected

#          from para_set and fed back to xg_eval()

#      (i) and (ii) are repeated for given number of iterations

#

xgBO = BayesianOptimization(

                             xg_eval,     # Function to evaluate performance.

                             para_set     # Parameter set from where parameters will be selected

                             )
gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian
start = time.time()

xgBO.maximize(init_points=5,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=25,        # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

              **gp_params

               )

end = time.time()

(end-start)/60
xgBO.res

xgBO.max
# Model with parameters of grid search

model_gs = XGBClassifier(

                    learning_rate = clf.best_params_['xg__learning_rate'],

                    max_depth = clf.best_params_['xg__max_depth'],

                    n_estimators=clf.best_params_['xg__n_estimators']

                    )



#  Model with parameters of random search

model_rs = XGBClassifier(

                    learning_rate = rs.best_params_['xg__learning_rate'],

                    max_depth = rs.best_params_['xg__max_depth'],

                    n_estimators=rs.best_params_['xg__n_estimators']

                    )



#  Model with parameters of Bayesian Optimization

model_bo = XGBClassifier(

                    learning_rate = xgBO.max['params']['learning_rate'],

                    max_depth = int(xgBO.max['params']['max_depth']),

                    n_estimators= int(xgBO.max['params']['n_estimators'])

                    )
# Modeling with all the parameters

start = time.time()

model_gs.fit(X_train, y_train)

model_rs.fit(X_train, y_train)

model_bo.fit(X_train, y_train)

end = time.time()

(end - start)/60
# Predictions with all the models

y_pred_gs = model_gs.predict(X_test)

y_pred_rs = model_rs.predict(X_test)

y_pred_bo = model_bo.predict(X_test)
accuracy_gs = accuracy_score(y_test, y_pred_gs)

print("Grid Search",accuracy_gs)

accuracy_rs = accuracy_score(y_test, y_pred_rs)

print("Random Search",accuracy_rs)

accuracy_bo = accuracy_score(y_test, y_pred_bo)

print("Bayesian Optimization",accuracy_bo)
model_gs.feature_importances_

model_rs.feature_importances_

model_bo.feature_importances_

plot_importance(model_gs)

plot_importance(model_rs)

plot_importance(model_bo)
conm_gs = confusion_matrix(y_test,y_pred_gs)

conm_gs

conm_rs = confusion_matrix(y_test,y_pred_rs)

conm_rs

conm_bo = confusion_matrix(y_test,y_pred_bo)

conm_bo