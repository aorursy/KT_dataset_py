#Clear ipython memory

%reset -f



#Data manipulation and plotting modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from xgboost import plot_importance

import seaborn as sns



#Call Standarad Scaler

from sklearn.preprocessing import StandardScaler as ss



#Cal PCA for Dimensionality reduction

from sklearn.decomposition import PCA



#Data splitting and model parameter search

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from bayes_opt import BayesianOptimization



#Call Modeling module, we will be using XBG for modeling.

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import precision_recall_fscore_support



#Model pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline





#Model evaluation metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix



# This is needed for Bayes optimization takes an estimator, performs cross-validation

# and gives out average score

from sklearn.model_selection import cross_val_score



#Find feature importance of ANY BLACK BOX estimator

import eli5

from eli5.sklearn import PermutationImportance





#Misc

import time

import os

import gc

import random

from scipy.stats import uniform



#Set option to dislay many rows

pd.set_option('display.max_columns', 100)

# Set working directory and set the filename

print(os.listdir("../input"))

os.listdir()

tr_f = "../input/winequalityN.csv"
# Total number of rows. Here I am reading 99% rows, just wanted to show we can also read part of data.

total_lines = 6498

num_lines = 0.99 * total_lines    # 99% of data

num_lines

p = num_lines/total_lines

df = pd.read_csv(tr_f,header=0, # First row is header-row

         skiprows=lambda i: (i>0) and (random.random() > p))
#Below is the code to check and remove the any NaN values present in the rows.

print (df.isnull().sum())

data=df.dropna()

data.isnull().sum()
data.shape
data.columns.values 
data.dtypes.value_counts() 
data.head(3)
data.describe()
data.type.value_counts()
# Box plot to show quality and fixed acidity

sns.boxplot(x='quality',y='fixed acidity', data=data)
# Count plot to show how many white and red wine types are there.

sns.countplot(x='type',data=data)
#Bar plot to show the quality and fixed acidity

sns.barplot(x = 'quality', y = 'fixed acidity', data = data)
# Heat map to show the correlation between the feature columns.

sns.heatmap(data.corr(),annot=True)
#Scatter plot to show the qulity and fixed acidity

sns.scatterplot("quality","fixed acidity",data=data)
#boxplot to show quality and volatile acidty with type as hue parameter.

sns.boxplot(x="quality", y="volatile acidity",hue='type',data=data)
# Separate the feature and target columns

X = data.iloc[ :, 1:13]

print (X.head(2))

y = data.iloc[ : , 0]

y.head(2)
#Map the target data to '1' and '0'

y = y.map({'white':1, 'red' : 0})

y.dtype
y.head()
# Split dataset into train and validation parts

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.35,

                                                    shuffle = True

                                                    )
print (X_train.shape)        

print (X_test.shape)         

print (y_train.shape)        

print (y_test.shape)
# Pipelining

steps_xg = [('sts', ss() ),

            ('pca', PCA()),

            ('xg',  XGBClassifier(silent = False,

                                  n_jobs=2)        

            )

            ]

pipe_xg = Pipeline(steps_xg)
parameters = {'xg__learning_rate':  [0, 1],

              'xg__n_estimators':   [50,  100],  

              'xg__max_depth':      [3,5],

              'pca__n_components' : [5,7] }
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
parameters = {'xg__learning_rate':  uniform(0, 1),

              'xg__n_estimators':   range(50,100),

              'xg__max_depth':      range(3,5),

              'pca__n_components' : range(5,7)}
rs = RandomizedSearchCV(pipe_xg,

                        param_distributions=parameters,

                        scoring= ['roc_auc', 'accuracy'],

                        n_iter=15,          

                                            

                        verbose = 3,

                        refit = 'roc_auc',

                        n_jobs = 2,          

                        cv = 2              

                                             

                        )
#Run random search for 25 iterations. 

start = time.time()

rs.fit(X_train, y_train)

end = time.time()

(end - start)/60
f"Best score: {rs.best_score_} "
f"Best parameter set: {rs.best_params_} "
y_pred = rs.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

f"Accuracy: {accuracy * 100.0}"
para_set = {

           'learning_rate':  (0, 1),                 

           'n_estimators':   (50,100),               

           'max_depth':      (3,5),               

           'n_components' :  (5,7)                

            }

def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    pipe_xg1 = make_pipeline (ss(),                        

                              PCA(n_components=int(round(n_components))),

                              XGBClassifier(

                                           silent = False,

                                           n_jobs=2,

                                           learning_rate=learning_rate,

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )



    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_train,

                                y = y_train,

                                cv = 2,

                                n_jobs = 2,

                                scoring = 'f1'

                                ).mean()             # take the average of all results





    return cv_result
xgBO = BayesianOptimization(

                             xg_eval,     # Function to evaluate performance.

                             para_set     # Parameter set from where parameters will be selected

                             )
start = time.time()

xgBO.maximize(init_points=5,    

               n_iter=25,        

               )

end = time.time()

(end-start)/60
xgBO.res
xgBO.max
# Grid Search

xg_gs = XGBClassifier(learning_rate = clf.best_params_['xg__learning_rate'],

                    max_depth = clf.best_params_['xg__max_depth'],

                    n_estimators=clf.best_params_['xg__n_estimators'])

#Randomized search

xg_rs = XGBClassifier(learning_rate = rs.best_params_['xg__learning_rate'],

                    max_depth = rs.best_params_['xg__max_depth'],

                    n_estimators=rs.best_params_['xg__n_estimators'])

#Bayes Optimization

xg_bo = XGBClassifier(learning_rate = xgBO.max['params']['learning_rate'],

                    max_depth = int(xgBO.max['params']['max_depth']),

                    n_estimators= int(xgBO.max['params']['n_estimators']))
#Fit the data using X_Train and y_train

xg_gs1 = xg_gs.fit(X_train,y_train)

xg_rs1 = xg_rs.fit(X_train,y_train)

xg_bo1 = xg_bo.fit(X_train,y_train)
y_pred_xg_gs = xg_gs1.predict(X_test)

y_pred_xg_rs = xg_rs1.predict(X_test)

y_pred_xg_bo = xg_bo1.predict(X_test)
y_pred_xg_gs_prob = xg_gs1.predict_proba(X_test)

y_pred_xg_rs_prob = xg_rs1.predict_proba(X_test)

y_pred_xg_bo_prob = xg_bo1.predict_proba(X_test)
print (accuracy_score(y_test,y_pred_xg_gs))

print (accuracy_score(y_test,y_pred_xg_rs))

print (accuracy_score(y_test,y_pred_xg_bo))
confusion_matrix(y_test,y_pred_xg_gs)

confusion_matrix(y_test,y_pred_xg_rs)

confusion_matrix(y_test,y_pred_xg_bo)
fpr_xg_gs, tpr_xg_gs, thresholds = roc_curve(y_test,

                                 y_pred_xg_gs_prob[: , 1],

                                 pos_label= 1

                                 )

fpr_xg_rs, tpr_xg_rs, thresholds = roc_curve(y_test,

                                 y_pred_xg_rs_prob[: , 1],

                                 pos_label= 1

                                 )

fpr_xg_bo, tpr_xg_bo, thresholds = roc_curve(y_test,

                                 y_pred_xg_bo_prob[: , 1],

                                 pos_label= 1

                                 )
p_xg_gs,r_xg_gs,f_xg_gs,_ = precision_recall_fscore_support(y_test,y_pred_xg_gs)

p_xg_rs,r_xg_rs,f_xg_rs,_ = precision_recall_fscore_support(y_test,y_pred_xg_rs)

p_xg_bo,r_xg_bo,f_xg_bo,_ = precision_recall_fscore_support(y_test,y_pred_xg_bo)
print (auc(fpr_xg_gs,tpr_xg_gs))

print (auc(fpr_xg_rs,tpr_xg_rs))

print (auc(fpr_xg_bo,tpr_xg_bo))
fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

# 9.2 Also connect diagonals

ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line

# 9.3 Labels etc

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for multiple tuning methods')

# 9.4 Set graph limits

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])



# 9.5 Plot each graph now

ax.plot(fpr_xg_gs, tpr_xg_gs, label = "xg_gs")

ax.plot(fpr_xg_rs, tpr_xg_rs, label = "xg_rs")

ax.plot(fpr_xg_bo, tpr_xg_bo, label = "xg_bo")

# 9.6 Set legend and show plot

ax.legend(loc="lower right")

plt.show()