# Importing Libraries

# For data manipulation

import numpy as np

import pandas as pd



# Data Preprocessing

from sklearn.preprocessing import StandardScaler as ss



# Dimensionality Reduction

from sklearn.decomposition import PCA



# Data Splitting and model parameter search

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



# For modelling

from xgboost.sklearn import XGBClassifier



# For model pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline



# For model evaluation

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix



# For plotting

import matplotlib.pyplot as plt

from xgboost import plot_importance

import seaborn as sns



# For Bayes Optimization

from sklearn.model_selection import cross_val_score



from bayes_opt import BayesianOptimization



# For finding feature importance

import eli5

from eli5.sklearn import PermutationImportance



# Miscellaneous

import time

import os

import gc

import random

from scipy.stats import uniform
# Set option to dislay many rows

pd.set_option('display.max_columns', 100)
# set file directory

os.chdir('../input/')
# Data has 6497 rows 

tr_f = "winequalityN.csv"



# Total number of lines and lines to read:

total_lines = 6497

num_lines = 6487



# Read randomly 'p' fraction of files

p = num_lines/total_lines  # fraction of lines to read (99% approximately)

p
# Pick up random rows from hard-disk

data = pd.read_csv(

         tr_f,

         header=0,   

         skiprows=lambda i: (i>0) and (random.random() > p)

         )
# Explore data

data.shape
data.info()
data.type.value_counts()
# Check for null values

data.isnull().sum()
# Deleting the rows with null values

data.dropna(axis=0, inplace=True)
data.shape
data.head(3)
data.describe()
data.corr()
sns.countplot(x = data.quality, data=data, hue='type', palette="rocket")
sns.set(style="ticks")

def hide_current_axis(*args, **kwds):

    plt.gca().set_visible(False)



p = sns.pairplot(data, vars = ['fixed acidity','free sulfur dioxide', 'total sulfur dioxide', 'volatile acidity', 'residual sugar','chlorides','density','citric acid'], diag_kind = 'kde', 

             hue='type',

             height = 4,

             palette="rocket")

p.map_upper(hide_current_axis)
plt.figure(figsize=(14,14))

sns.heatmap(data.iloc[:,0:13].corr(), cbar = True,  square = True, annot=True, cmap= 'BuGn_r')



#Free sulfur dioxide and total sulfar dioxide, and Density and alcohol are the most correlated features
fig = plt.figure(figsize=(22,10))

features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "total sulfur dioxide", "chlorides", "fixed acidity", "density","sulphates"]



for i in range(8):

    ax1 = fig.add_subplot(2,4,i+1)

    sns.boxplot(x="type", y=features[i],data=data, palette="rocket");

    

# Fixed Acidity: acid that contributes to the conservation of wine.

# Volatile Acidity: Amount of acetic acid in wine at high levels can lead to an unpleasant taste of vinegar.

# Citric Acid: found in small amounts, can add “freshness” and flavor to wines.

# Residual sugar: amount of sugar remaining after the end of the fermentation.

# Chlorides: amount of salt in wine.

# Free Sulfur Dioxide: it prevents the increase of microbes and the oxidation of the wine.

# Total Sulfur Dioxide: it shows the aroma and taste of the wine.

# Density: density of water, depends on the percentage of alcohol and amount of sugar.

# pH: describes how acid or basic a wine is on a scale of 0 to 14.

# Sulfates: additive that acts as antimocrobian and antioxidant.

# Alcohol: percentage of alcohol present in the wine.
fig = plt.figure(figsize=(24,10))

features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "total sulfur dioxide", "chlorides", "fixed acidity", "citric acid","sulphates"]



for i in range(8):

    ax1 = fig.add_subplot(2,4,i+1)

    sns.barplot(x='quality', y=features[i],data=data, hue='type', palette='rocket')

    
# Divide data into predictors and target

X = data.iloc[ :, 1:13]

X.head(2)
# 1st index or 1st column is target

y = data.iloc[ : , 0]

y.head()
#  Transform type data to '1' and '0'

y = y.map({'white':1, 'red' : 0})

y.dtype           # int64
# Store column names somewhere for use in feature importance

colnames = X.columns.tolist()



colnames
# Split dataset into train and validation parts

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.30,

                                                    shuffle = True

                                                    )

X_train.shape  
#### Create pipeline ####

#### Pipe using XGBoost



steps_xg = [('sts', ss() ),

            ('pca', PCA()),

            ('xg',  XGBClassifier(silent = False,

                                  n_jobs=2)        # Specify other parameters here

            )

            ]

# Instantiate Pipeline object

pipe_xg = Pipeline(steps_xg)
##################### Grid Search #################



#   Specify xgboost parameter-range

#   Dictionary of parameters (16 combinations)

#     Syntax: {

#              'transformerName_parameterName' : [ <listOfValues> ]

#              }





parameters = {'xg__learning_rate':  [0.3, 0.05],

              'xg__n_estimators':   [50,  100],

              'xg__max_depth':      [3,5],

              'pca__n_components' : [5,7]

              }                               # Total: 2 * 2 * 2 * 2
#    Grid Search (16 * 2) iterations

#    Create Grid Search object first with all necessary

#    specifications. Note that data, X, as yet is not specified

clf = GridSearchCV(pipe_xg,            # pipeline object

                   parameters,         # possible parameters

                   n_jobs = 2,         # USe parallel cpu threads

                   cv =5 ,             # No of folds

                   verbose =2,         # Higher the value, more the verbosity

                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance

                   refit = 'roc_auc'   # Refitting final model on what parameters?

                                       # Those which maximise auc

                   )
# Start fitting data to pipeline

start = time.time()

clf.fit(X_train, y_train)

end = time.time()

(end - start)/60
f"Best score: {clf.best_score_} "
f"Best parameter set {clf.best_params_}"
y_pred_gs = clf.predict(X_test)

# Accuracy

accuracy_gs = accuracy_score(y_test, y_pred_gs)

f"Accuracy: {accuracy_gs * 100.0}"
#  Find feature importance of any BLACK Box model



# Instantiate the importance object

perm = PermutationImportance(

                            clf,

                            random_state=1

                            )



# fit data & learn

start = time.time()

perm.fit(X_test, y_test)

end = time.time()

(end - start)/60
#  Conclude: Get feature weights



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
#####################  Randomized Search #################



# Tune parameters using randomized search

# Hyperparameters to tune and their ranges

parameters = {'xg__learning_rate':  uniform(0, 1),

              'xg__n_estimators':   range(50,100),

              'xg__max_depth':      range(3,5),

              'pca__n_components' : range(5,7)}
#     Tune parameters using random search

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
# Run random search for 25 iterations. 21 minutes

start = time.time()

rs.fit(X_train, y_train)

end = time.time()

(end - start)/60
# Evaluate

f"Best score: {rs.best_score_} "

f"Best parameter set: {rs.best_params_} "
# Make predictions

y_pred_rs = rs.predict(X_test)
# Accuracy

accuracy_rs = accuracy_score(y_test, y_pred_rs)

f"Accuracy: {accuracy_rs * 100.0}"
###############  Tuning using Bayes Optimization ############

# Which parameters to consider and what is each one's range

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
#     Gaussian process parameters

#     Modulate intelligence of Bayesian Optimization process

#     This parameters controls how much noise the GP can handle,

#     so increase it whenever you think that extra flexibility is needed.

gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian

                                 # process.
#  Fit/train (so-to-say) the BayesianOptimization() object

#     Start optimization. 25minutes

#     Our objective is to maximize performance (results)

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
#  Get values of parameters that maximise the objective

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
# 9.4 Accuracy from all the models

accuracy_gs = accuracy_score(y_test, y_pred_gs)

accuracy_rs = accuracy_score(y_test, y_pred_rs)

accuracy_bo = accuracy_score(y_test, y_pred_bo)

print("Grid Search",accuracy_gs)

print("Random Search",accuracy_rs)

print("Bayesian Optimization",accuracy_bo)
#  Get feature importances from all the models

model_gs.feature_importances_

model_rs.feature_importances_

model_bo.feature_importances_

plot_importance(model_gs)

plot_importance(model_rs)

plot_importance(model_bo)

# Confusion matrix for all the models



cm_gs = confusion_matrix(y_test,y_pred_gs)

cm_rs = confusion_matrix(y_test,y_pred_rs)

cm_bo = confusion_matrix(y_test,y_pred_bo)



cms = [cm_gs, cm_rs, cm_bo]

classifiers = ["Grid Search","Random Search","Bayesian Optimization"]



def plot_confusion_matrix(cms):

   

    fig = plt.figure(figsize=(20,12))

    plt.subplots_adjust( hspace=0.5, wspace=0.4)

    for i in range(3):

        j = i+1

        ax = fig.add_subplot(1,3,j)

        plt.imshow(cms[i], interpolation='nearest', cmap=plt.cm.Pastel1)



        classNames = ['Red','White']



        plt.ylabel('Actual', size='large')



        plt.xlabel('Predicted', size='large')



        tick_marks = np.arange(len(classNames))

        plt.xticks(tick_marks, classNames, size='x-large')



        plt.yticks(tick_marks, classNames, size='x-large')



        s = [['TN','FP'], ['FN', 'TP']]

    

    

        plt.text(-0.23,0.05, str(s[0][0])+" = "+str(cms[i][0][0]), size='x-large')

        plt.text(0.8,0.05, str(s[0][1])+" = "+str(cms[i][0][1]), size='x-large')

        plt.text(-0.23,1.05, str(s[1][0])+" = "+str(cms[i][1][0]), size='x-large')

        plt.text(0.8,1.05, str(s[1][1])+" = "+str(cms[i][1][1]), size='x-large')

        plt.title(classifiers[i], fontsize=15)



plot_confusion_matrix(cms)
# Get probability of occurrence of each class

y_pred_prob_gs = model_gs.predict_proba(X_test)

y_pred_prob_rs = model_rs.predict_proba(X_test)

y_pred_prob_bo = model_bo.predict_proba(X_test)



# Draw ROC curve

fpr_gs, tpr_gs, thresholds = roc_curve(y_test,

                                 y_pred_prob_gs[: , 0],

                                 pos_label= 0

                                 )



fpr_rs, tpr_rs, thresholds = roc_curve(y_test,

                                 y_pred_prob_rs[: , 0],

                                 pos_label= 0

                                 )



fpr_bo, tpr_bo, thresholds = roc_curve(y_test,

                                 y_pred_prob_bo[: , 0],

                                 pos_label= 0

                                 )

# AUC

auc_gs = auc(fpr_gs,tpr_gs)

auc_rs = auc(fpr_rs,tpr_rs)

auc_bo = auc(fpr_bo,tpr_bo)
performance = pd.DataFrame({ "Classifiers":["Grid Search","Random Search",'Bayesian Optimization'],

                             "Accuracy": [accuracy_score(y_test,y_pred_gs),accuracy_score(y_test,y_pred_rs),accuracy_score(y_test,y_pred_bo)],

                             "Precision": [precision_score(y_test,y_pred_gs),precision_score(y_test,y_pred_rs),precision_score(y_test,y_pred_bo)],

                             "AUC":[auc_gs,auc_rs,auc_bo],

                             "Recall":[recall_score(y_test,y_pred_gs),recall_score(y_test,y_pred_rs),recall_score(y_test,y_pred_bo)],

                             "f1_score":[f1_score(y_test,y_pred_gs),f1_score(y_test,y_pred_rs),f1_score(y_test,y_pred_bo)]})
performance
fig = plt.figure(figsize=(12,10))   # Create window frame

ax = fig.add_subplot(111)   # Create axes



#8.1 Connect diagonals

ax.plot([0, 1], [0, 1], ls="--")  # Dashed diagonal line



#8.2 Labels 

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')



#8.3 Set graph limits

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])



#8.4 Plot each graph now

ax.plot(fpr_gs, tpr_gs, label = "gs")

ax.plot(fpr_rs, tpr_rs, label = "rs")

ax.plot(fpr_bo, tpr_bo, label = "bo")





#8.5 Set legend and show plot

ax.legend(loc="lower right")

plt.show()

    