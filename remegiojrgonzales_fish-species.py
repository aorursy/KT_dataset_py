# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np

import pandas as pd

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/fishmarket/fishes.csv')
sns.pairplot(df, hue="Species") 
sns.countplot(data=df, x="Species").set_title("Species Outcome")
sns.relplot(data=df, x="Weight", y="Height", hue="Species", palette="bright", height=6)
sns.relplot(data=df, x="Length1", y="Width", hue="Species", palette="bright", height=6)
# Check if there are any null values

df.isnull().values.any()
# Remove null values

df = df.dropna()
# Check if there are any null values

df.isnull().values.any()
#Drop not needed columns / features

df.drop('Id', axis=1, inplace=True)

df.head()
# Import required libraries for performance metrics

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split
def get_performance_measures(actual, prediction):

    matrix = confusion_matrix(actual, prediction)

    FP = matrix.sum(axis=0) - np.diag(matrix)  

    FN = matrix.sum(axis=1) - np.diag(matrix)

    TP = np.diag(matrix)

    TN = matrix.sum() - (FP + FN + TP)



    return(TP, FP, TN, FN)
#Custom Scorers



# # Sensitivity, hit rate, recall, or true positive rate

# TPR = TP/(TP+FN)

# # Specificity or true negative rate

# TNR = TN/(TN+FP) 

# # Precision or positive predictive value

# PPV = TP/(TP+FP)

# # Negative predictive value

# NPV = TN/(TN+FN)

# # Fall out or false positive rate

# FPR = FP/(FP+TN)

# # False negative rate

# FNR = FN/(TP+FN)

# # False discovery rate

# FDR = FP/(TP+FP)



# # Overall accuracy

# ACC = (TP+TN)/(TP+FP+FN+TN)



# Also remember:

# specificity = true negative rate

# sensitivity = true positive rate



def sensitivity_score(y_true, y_pred, mode="multiclass"):

    if mode == "multiclass":

        TP, FP, TN, FN = get_performance_measures(y_true, y_pred)

        TPR = (TP/(TP+FN)).mean()

    elif mode == "binary":

        TP, FP, TN, FN = get_performance_measures(y_true, y_pred)

        TPR = (TP/(TP+FN))[1] # Since the [0] part is the index

    else:

        raise Exception("Mode not recognized!")

    

    return TPR



def specificity_score(y_true, y_pred, mode="multiclass"):

    if mode == "multiclass":

        TP, FP, TN, FN = get_performance_measures(y_true, y_pred)

        TNR = (TN/(TN+FP)).mean()

    elif mode == "binary":

        TP, FP, TN, FN = get_performance_measures(y_true, y_pred)

        TNR = (TN/(TN+FP))[1]

    else:

        raise Exception("Mode not recognized!")

    

    return TNR
# Define dictionary with performance metrics

# To know what everaging to use: https://stats.stackexchange.com/questions/156923/should-i-make-decisions-based-on-micro-averaged-or-macro-averaged-evaluation-mea#:~:text=So%2C%20micro%2Daveraged%20measures%20add,is%20more%20like%20an%20average.





scoring = {

            'accuracy':make_scorer(accuracy_score), 

            'precision':make_scorer(precision_score, average='weighted'),

            'f1_score':make_scorer(f1_score, average='weighted'),

            'recall':make_scorer(recall_score, average='weighted'), 

            'sensitvity':make_scorer(sensitivity_score, mode="multiclass"), 

            'specificity':make_scorer(specificity_score, mode="multiclass"), 

           }
# Import required libraries for machine learning classifiers

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.naive_bayes import GaussianNB #Naive Bayes

from sklearn.linear_model import LogisticRegression #Logistic Regression

from sklearn.svm import LinearSVC # Support Vector Machine

from sklearn.neighbors import KNeighborsClassifier #K-nearest Neighbors

from sklearn.cluster import KMeans #K-means



# Instantiate the machine learning classifiers

decisionTreeClassifier_model = DecisionTreeClassifier()

gaussianNB_model = GaussianNB()

logisticRegression_model = LogisticRegression(max_iter=10000)

linearSVC_model = LinearSVC(dual=False)

kNeighbors_model = KNeighborsClassifier()
# features = data frame set that contain your features that will be used as input to see if prediction is equal to actual result

# target = data frame set (1 column usually) that will contain your target or actual results.

# folds = this is added so we can easily change the number of folds we want to do with our data set.

# folding is a technique to minimise overfitting and therefore make our model more accurate.

def models_evaluation(features, target, folds):    

    # Perform cross-validation to each machine learning classifier

    decisionTreeClassifier_result = cross_validate(decisionTreeClassifier_model, features, target, cv=folds, scoring=scoring)

    gaussianNB_result = cross_validate(gaussianNB_model, features, target, cv=folds, scoring=scoring)

    logisticRegression_result = cross_validate(logisticRegression_model, features, target, cv=folds, scoring=scoring)

    linearSVC_result = cross_validate(linearSVC_model, features, target, cv=folds, scoring=scoring)

    kNeighbors_result = cross_validate(kNeighbors_model, features, target, cv=folds, scoring=scoring)

    # kMeans_result = cross_validate(kMeans_model, features, target, cv=folds, scoring=scoring)



    # Create a data frame with the models perfoamnce metrics scores

    models_scores_table = pd.DataFrame({

      'Decision Tree':[

                        decisionTreeClassifier_result['test_accuracy'].mean(),

                        decisionTreeClassifier_result['test_precision'].mean(),

                        decisionTreeClassifier_result['test_recall'].mean(),

                        decisionTreeClassifier_result['test_sensitvity'].mean(),

                        decisionTreeClassifier_result['test_specificity'].mean(),

                        decisionTreeClassifier_result['test_f1_score'].mean()

                       ],



      'Gaussian Naive Bayes':[

                                gaussianNB_result['test_accuracy'].mean(),

                                gaussianNB_result['test_precision'].mean(),

                                gaussianNB_result['test_recall'].mean(),

                                gaussianNB_result['test_sensitvity'].mean(),

                                gaussianNB_result['test_specificity'].mean(),

                                gaussianNB_result['test_f1_score'].mean()

                              ],



      'Logistic Regression':[

                                logisticRegression_result['test_accuracy'].mean(),

                                logisticRegression_result['test_precision'].mean(),

                                logisticRegression_result['test_recall'].mean(),

                                logisticRegression_result['test_sensitvity'].mean(),

                                logisticRegression_result['test_specificity'].mean(),

                                logisticRegression_result['test_f1_score'].mean()

                            ],



      'Support Vector Classifier':[

                                    linearSVC_result['test_accuracy'].mean(),

                                    linearSVC_result['test_precision'].mean(),

                                    linearSVC_result['test_recall'].mean(),

                                    linearSVC_result['test_sensitvity'].mean(),

                                    linearSVC_result['test_specificity'].mean(),

                                    linearSVC_result['test_f1_score'].mean()

                                   ],



       'K-nearest Neighbors':[

                        kNeighbors_result['test_accuracy'].mean(),

                        kNeighbors_result['test_precision'].mean(),

                        kNeighbors_result['test_recall'].mean(),

                        kNeighbors_result['test_sensitvity'].mean(),

                        kNeighbors_result['test_specificity'].mean(),

                        kNeighbors_result['test_f1_score'].mean()

                       ],



      },



      index=['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1 Score', ])

    

    # Return models performance metrics scores data frame

    return(models_scores_table)
# Let's try to look at our data frame again one last time

df.head()
# Specify features columns

# Actually what we are doing here is that we are just dropping the Species column since that is our class

# and the remaining columns will then be our features (eg. inputs to come up to a class)

# axis 0 basically means to drop all of that column

features = df.drop(columns="Species", axis=0)



# Now let's see what features looks like

features



# Don't mind the left hand side, those are just index mainly used for viewing
evaluationResult = models_evaluation(features, target, 5)

view = evaluationResult

view = view.rename_axis('Test Type').reset_index() #Add the index names to the column. This will be used for our presentation



# https://pandas.pydata.org/docs/reference/api/pandas.melt.html

# Re-Organizing our dataframe to fit our view need

view = view.melt(var_name='Classifier', value_name='Value', id_vars='Test Type')

# result

sns.catplot(data=view, x="Test Type", y="Value", hue="Classifier", kind='bar', palette="bright", alpha=0.8, legend=True, height=5, margin_titles=True, aspect=2)
# In here we just add a new column to our raw data frame, that gets the result for the highest

# scoring classifier in every score test.

evaluationResult['Best Score'] = evaluationResult.idxmax(axis=1)

evaluationResult