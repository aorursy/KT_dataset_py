# Data manipulation and plotting modules

import numpy as np

import pandas as pd
# Data pre-processing

# z = (x-mean)/stdev

from sklearn.preprocessing import StandardScaler as ss
# Dimensionality reduction

from sklearn.decomposition import PCA
# Data splitting and model parameter search

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
# Modeling modules

from xgboost.sklearn import XGBClassifier
# Model pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline
# Model evaluation metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix
# Modules for graph plotting

import matplotlib.pyplot as plt

from xgboost import plot_importance

import seaborn as sns
# Needed for Bayes optimization

# Takes an estimator, performs cross-validation and gives out average score

from sklearn.model_selection import cross_val_score
# Baynesian Optimization Module

# Refer: https://github.com/fmfn/BayesianOptimization

from bayes_opt import BayesianOptimization
# Importing some supporting modules

import time

import random

from scipy.stats import uniform
# Data has 6498 rows including header and 13 columns.

# Assuming it is very big data size and we cannot process it all

# and so we have to make a model on data sample we will import only 90% of the data.

data = pd.read_csv(

         "../input/winequalityN.csv",

         header = 0,   # First row is header-row

         # 'and' operator returns True if both values are True

         #  random.random() returns values between (0,1)

         #  No of rows skipped will be around 60% of total

         skiprows = lambda i: (i>0) and (random.random() > 0.9)

         )
# Explore Data

f"Data Shape : { data.shape }"
print("\033[1mColumns in this dataset\033[0m\n\n", "\n".join(data.columns.values))
print("\033[1mData Types\033[0m\n\n",data.dtypes.value_counts())
print("\033[1mData Types\033[0m\n\n",data.dtypes)
# Converting data type of quality column to float.

data["quality"] = data["quality"].astype(float)
# Data Contents

data.head(10)
data.describe()
# Check any null values / NAN

data.isnull().any()
data.isnull().sum()
# Fill NaN by the most frequntly occuring number, we can choose mean also but for this example will go for mode.

df = data.isnull().any().reset_index()

df
na_columns = df.loc[df.iloc[:, 1] == True, "index"].tolist()

print("\033[1mColumns list:\033[0m \n\n{0}".format("\n".join(na_columns)))
# Getting mode for the columns in the list na_columns

values = dict(map(lambda i: (i, float(data[i].mode())), na_columns))

maxlen = max([len(i) for i in values.keys()])

print("\033[1mMode Values:\033[0m \n")

for key, value in values.items():

    print(key, " " * (maxlen - len(key) + 5), "{:.2}".format(value))
# Replacing Null / NaN values in data set.

data = data.fillna(value = values)
# Validating if any null values left

data.isnull().any()
# Pairplot for some of the data feautres

sns.pairplot(data.iloc[:, 4:10])
# Plotting correlation table

fig = plt.figure(figsize=(20,15))

sns.heatmap(data.corr(), annot=True, annot_kws={"size": 14})
# Divide data into predictors and target X is predictors and Y is target.

X = data.drop("type", axis=1)

X.head()
# Y is type column

y = data["type"]

y.value_counts()
# Mapping white lable by 1 and red by 0

y = y.map({'white': 1, 'red': 0})

y.value_counts()
# Split dataset into train and validation parts

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = True)
# Check training and test data set shape

print("X_train :", X_train.shape)

print("X_test  :", X_test.shape)

print("y_train :", y_train.shape)

print("y_test  :", y_test.shape)
################# CC. Create pipeline #################

#### Pipe using XGBoost



# 5 Pipeline steps

# steps: List of (name, transform) tuples

#       (implementing fit/transform) that are

#       chained, in the order in which they

#       are chained, with the last object an

#       estimator.

#      Format: [(name, transformer), (name, transformer)..(name, estimator)]

steps_xg = [('sts', ss()), ('pca', PCA()), ('xg',  XGBClassifier(silent = False, n_jobs=1))]



# Instantiate Pipeline object

pipe_xg = Pipeline(steps_xg)
##################### Grid Search #################



# 6.  Specify xgboost parameter-range

# 6.1 Dictionary of parameters (16 combinations)

#     Syntax: {

#              'transformerName_parameterName' : [ <listOfValues> ]

#              }

#

parameters = {'xg__learning_rate':  [0.1, 0.3, 0.5, 0.8], # learning rate decides what percentage

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

              'xg__n_estimators':   [50, 65, 85, 100],  # Number of boosted trees to fit

                                                  # l1 and l2 specifications will change

                                                  # the values of coeff of boosted trees

                                                  # but not their numbers

              'xg__max_depth':      [2, 3, 5, 7],

              'pca__n_components' : [0.3, 0.5, 0.7, 0.9]}
#    Grid Search iterations

#    Create Grid Search object first with all necessary

#    specifications. Note that data, X, as yet is not specified

#    Pipeline avoids data leakage during GridSearch

#    See this: https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976

clf = GridSearchCV(pipe_xg,            # pipeline object

                   parameters,         # possible parameters

                   n_jobs = 1,         # USe parallel cpu threads

                   cv =5 ,             # No of folds

                   verbose = 1,         # Higher the value, more the verbosity

                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance

                   refit = 'roc_auc'   # Refitting final model on what parameters? those which maximise auc

                   )
# Start fitting data to pipeline

clf.fit(X_train, y_train)
# Best Score and best parameters

print("Best score :", clf.best_score_ * 100)

print("Best parameter set :", clf.best_params_)
# Making predictions

y_pred = clf.predict(X_test)
# Check Accuracy

accuracy = accuracy_score(y_test, y_pred)

f"Accuracy: {accuracy * 100.0}"
# Confusion matrix

cm = confusion_matrix( y_test,y_pred)

sns.heatmap(cm, annot=True, fmt='g', annot_kws={"size": 14})
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

print("True Positives : ", tp)

print("True Negatives : ", tn)

print("False Positives : ", fp)

print("False Negatives : ", fn)
# probbaility of occurrence of each class

y_pred_prob = clf.predict_proba(X_test)

print("y_pred_prob shape : ", y_pred_prob.shape)
# Probability values in y_pred_prob are ordered

# column-wise, as:

clf.classes_
# Draw ROC curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[: , 0], pos_label = 0)
# Plot the ROC curve

fig = plt.figure(figsize=(10,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

ax.plot(fpr, tpr)           # Plot on the axes

# Also connect diagonals

ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line

# Labels etc

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for Higgs Boson particle')



ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])

plt.show()
# AUC

auc(fpr,tpr)
##################### EE. Randomized Search #################



# Tune parameters using randomized search

# Hyperparameters to tune and their ranges

parameters = {'xg__learning_rate':  uniform(0, 1),

              'xg__n_estimators':   range(50,100),

              'xg__max_depth':      range(2,7),

              'pca__n_components' : range(5, 10)}
# Tune parameters using random search

# Create the object first

rs = RandomizedSearchCV(pipe_xg,

                        param_distributions=parameters,

                        scoring= ['roc_auc', 'accuracy'],

                        n_iter=15,          # Max combination of parameter to try. Default = 10

                        verbose = 3,

                        refit = 'roc_auc',

                        n_jobs = 4,          # Use parallel cpu threads

                        cv = 7               # No of folds, so n_iter * cv combinations

                        )
# Run random search for 25 iterations.

rs.fit(X_train, y_train)
# Evaluation

print("Best score: ", rs.best_score_)

print("Best parameter set: ", rs.best_params_)
# Make predictions

y_pred = rs.predict(X_test)
# Accuracy check

accuracy = accuracy_score(y_test, y_pred)

f"Accuracy: {accuracy * 100.0}"
# Confusion matrix

cm = confusion_matrix( y_test,y_pred)

sns.heatmap(cm, annot=True, fmt='g', annot_kws={"size": 14})
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

print("True Positives : ", tp)

print("True Negatives : ", tn)

print("False Positives : ", fp)

print("False Negatives : ", fn)
"""

Step 1: Define BayesianOptimization function.

    It broadly acts as follows"

    s1. Gets a dictionary of parameters that specifies

        possible range of values for each one of

        the parameters. [Our set: para_set ]

    s2. Picks one value for each one of the parameters

        (from the specified ranges as in (s1)) evaluate,

        a loss-function that is given to it, say,

        accuracy after cross-validation.

        [Our function: xg_eval() ]

    s3. Depending upon the value of accuracy returned

        by the evaluator and also past values of accuracy

        returned, this function, creates gaussian

        processes and picks up another set of parameters

        from the given dictionary of parameters

    s4. The parameter set is then fed back to (s2) above

        for evaluation

    s5. (s2) t0 (s4) are repeated for given number of

        iterations and then final set of parameters

        that optimizes objective is returned



"""



# Which parameters to consider and what is each one's range

para_set = {

           'learning_rate':  (0, 1),                 # any value between 0 and 1

           'n_estimators':   (50,100),               # any number between 50 to 100

           'max_depth':      (2,7),                 # any depth between 3 to 10

           'n_components' :  (5,10)                 # any number between 20 to 30

            }
# Create a function that when passed some parameters

#    evaluates results using cross-validation

#    This function is used by BayesianOptimization() object



def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    

    # 12.1 Make pipeline. Pass parameters directly here

    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?

                              PCA(n_components = int(round(n_components))),

                              XGBClassifier(

                                           silent = False,

                                           n_jobs = 4,

                                           learning_rate = learning_rate,

                                           max_depth = int(round(max_depth)),

                                           n_estimators = int(round(n_estimators))

                                           )

                             )



    # 12.2 Now fit the pipeline and evaluate

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X = X_train,

                                y = y_train,

                                cv = 2,

                                n_jobs = 4,

                                scoring = 'f1'

                                ).mean()             # take the average of all results

    # 12.3 Finally return maximum/average value of result

    return cv_result

# This is the main workhorse

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
# Gaussian process parameters

#     Modulate intelligence of Bayesian Optimization process

#     This parameters controls how much noise the GP can handle,

#     so increase it whenever you think that extra flexibility is needed.

#     gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian process.

#

# Fit/train (so-to-say) the BayesianOptimization() object

#     Start optimization. 25minutes

#     Our objective is to maximize performance (results)

xgBO.maximize(init_points=5,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=9,        # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

#              **gp_params

               )
# Get values of parameters that maximise the objective

max_xgBO = xgBO.max



print("\033[1mBest Score:\033[0m ", max_xgBO["target"])







maxlen = max([len(i) for i in max_xgBO["params"].keys()])



print("\n\n\033[1mMaximum performing combination:\033[0m \n")

for key, value in max_xgBO["params"].items():

    print(key, " " * (maxlen - len(key) + 5), value)


