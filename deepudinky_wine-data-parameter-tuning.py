import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import time

import os

import gc

import random



from sklearn.preprocessing import StandardScaler as ss

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score



from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance



from bayes_opt import BayesianOptimization



import eli5

from eli5.sklearn import PermutationImportance



from scipy.stats import uniform
# Load the dataset

#df = pd.read_csv('C:\\Users\\nt65000\\Downloads\\winequalityN.csv')

os.chdir("../input") 

df = pd.read_csv("winequalityN.csv")
# Show all the records

pd.set_option('display.max_columns', 100)
df.head()
df.tail()
df.info()
df.shape
df.describe
df['quality'].unique()
# Corelation coefficient - To measure the strength of the relationship between two variables

corr=df.corr()
corr
# Checking for null values

df.isnull().sum()
# Deleting the rows with null values

df.dropna(axis=0, inplace=True)
# Checking data after dropping null value rows

df.shape
plt.figure(figsize=(14,6))

sns.heatmap(corr,annot=True)
# Find percentage of wine types

plt.figure(figsize=(15,7))

 

# Data to plot

labels = 'white', 'green'

sizes = [4870,1593]

colors = ['green', 'yellow']

explode = (0.1, 0 )  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('The percentage of type of wine',fontsize=20)

plt.legend(('white', 'green'),fontsize=15)

plt.axis('equal')

plt.show()
data_red = df[df.type == "red"]

data_red.plot(kind = "scatter", x = "residual sugar", y = "alcohol", alpha = .5, color = "r")

plt.title("Alcohol - Residual Sugar Scatter Plot")

plt.show()
g = sns.pairplot(df,palette="hls",diag_kind="kde",hue='type')
# Split the data into predictors and target

X = df.iloc[ :, 1:13]

y = df.iloc[ : , 0]
#  Transform type data to '1' and '0'

y = y.map({'white':1, 'red' : 0})

y.dtype          
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

                                  n_jobs=2)        

            )

            ]

# Instantiate Pipeline object

pipe_xg = Pipeline(steps_xg)
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
f"Best score: {rs.best_score_} "
# Make predictions

y_pred_rs = rs.predict(X_test)
# Accuracy

accuracy_rs = accuracy_score(y_test, y_pred_rs)

f"Accuracy: {accuracy_rs * 100.0}"
###############  Tuning using Bayes Optimization ############

para_set = {

           'learning_rate':  (0, 1),                 

           'n_estimators':   (50,100),               

           'max_depth':      (3,5),                 

           'n_components' :  (5,7)          

            }
#    Create a function that when passed some parameters

#    evaluates results using cross-validation

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
# Instantiate BayesianOptimization() object

xgBO = BayesianOptimization(

                             xg_eval,     

                             para_set     

                             )
# Gaussian process parameters

gp_params = {"alpha": 1e-5}      
#  Fit/train the BayesianOptimization() object

start = time.time()

xgBO.maximize(init_points=5,    

               n_iter=25,        

              **gp_params

               )

end = time.time()

(end-start)/60
# Get the values of parameters that maximise the objective

xgBO.res

xgBO.max
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

model_rs.fit(X_train, y_train)

model_bo.fit(X_train, y_train)

end = time.time()

(end - start)/60
# Predictions with all the models

y_pred_rs = model_rs.predict(X_test)

y_pred_bo = model_bo.predict(X_test)
# Accuracy from all the models

accuracy_rs = accuracy_score(y_test, y_pred_rs)

accuracy_bo = accuracy_score(y_test, y_pred_bo)

print("Random Search",accuracy_rs)

print("Bayesian Optimization",accuracy_bo)
# Feature importances from all the models

model_rs.feature_importances_

model_bo.feature_importances_

plot_importance(model_rs)

plot_importance(model_bo)
# Get probability of occurrence of each class

y_pred_prob_rs = model_rs.predict_proba(X_test)

y_pred_prob_bo = model_bo.predict_proba(X_test)



# Draw ROC curve

fpr_rs, tpr_rs, thresholds = roc_curve(y_test,

                                 y_pred_prob_rs[: , 0],

                                 pos_label= 0

                                 )



fpr_bo, tpr_bo, thresholds = roc_curve(y_test,

                                 y_pred_prob_bo[: , 0],

                                 pos_label= 0

                                 )

# AUC

auc_rs = auc(fpr_rs,tpr_rs)

auc_bo = auc(fpr_bo,tpr_bo)
performance = pd.DataFrame({ "Classifiers":["Random Search",'Bayesian Optimization'],

                             "Accuracy": [accuracy_score(y_test,y_pred_rs),accuracy_score(y_test,y_pred_bo)],

                             "Precision": [precision_score(y_test,y_pred_rs),precision_score(y_test,y_pred_bo)],

                             "AUC":[auc_rs,auc_bo],

                             "Recall":[recall_score(y_test,y_pred_rs),recall_score(y_test,y_pred_bo)],

                             "f1_score":[f1_score(y_test,y_pred_rs),f1_score(y_test,y_pred_bo)]})

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

ax.plot(fpr_rs, tpr_rs, label = "rs")

ax.plot(fpr_bo, tpr_bo, label = "bo")





#8.5 Set legend and show plot

ax.legend(loc="lower right")

plt.show()
plt.bar(rs.best_params_.keys(), rs.best_params_.values(), color='y')

plt.xticks(rotation=25)
for features in xgBO.max.values(): 

    print(features)
plt.bar(features.keys(), features.values(), color='y')

plt.xticks(rotation=25)