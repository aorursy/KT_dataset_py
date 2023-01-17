# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sbrn

# Check the files we have in the our folder
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read CSV from kaggle
dataFrame = pd.read_csv('/kaggle/input/datasets_714830_1245295_wine quality.csv') 

# Display first 5 rows of the data set
dataFrame.head() 

# Wine Classes Distribution
sbrn.countplot(data=dataFrame, y="quality").set_title("Wine Classes Distribution")
# Check for any null values on the data set
dataFrame.isnull().values.any()
# Remove null values on the data set
dataFrame = dataFrame.dropna()
# Check again for any null values on the data set
dataFrame.isnull().values.any()
# Importing libraries from https://scikit-learn.org/ for performance scoring and evaluation of the models

# For n-fold cross validation later
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix 

# For the calculation and graphical visualization of scores later
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def get_performance_scores(actual, predicted):
    # Compute confusion matrix to evaluate the accuracy of a classification.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    matrix = confusion_matrix(actual, predicted)
    
    # Get the false positive score by subtracting a diagonal array from sum of matrix's first axis/column
    false_positive = matrix.sum(axis=0) - np.diag(matrix)  
    
    # Get the false negative score by subtracting a diagonal array from sum of matrix's second axis/column
    false_negative = matrix.sum(axis=1) - np.diag(matrix)
    
    # Get the true positive score by constructing a diagonal array from data of matrix variable
    true_positive = np.diag(matrix)
    
    # Get the true negative score by subtracting the sum of FP, FN, TP to the sum of the confusion matrix result
    true_negative = matrix.sum() - (false_positive + false_negative + true_positive)

    return(true_positive, false_positive, true_negative, false_negative)

# Sensitivity = true positive rate
def sensitivity_rate(actual, predicted):
    true_positive, false_positive, true_negative, false_negative = get_performance_scores(actual, predicted)
    sensitivity = (true_positive / (true_positive + false_negative))[1] # 2nd index
    return sensitivity

# Specificity = true negative rate
def specificity_rate(actual, predicted): 
    true_positive, false_positive, true_negative, false_negative = get_performance_scores(actual, predicted)
    specificity = (true_negative / (true_negative + false_positive))[1] # 2nd index
    return specificity

# Defining the scorers with performance metrics we imported from sklearn and also from what we defined above
scoring = {
            'accuracy':make_scorer(accuracy_score), 
            'precision':make_scorer(precision_score, average='weighted'),
            'f1_score':make_scorer(f1_score, average='weighted'),
            'recall':make_scorer(recall_score, average='weighted'), 
            'sensitvity':make_scorer(sensitivity_rate), 
            'specificity':make_scorer(specificity_rate), 
           }
# Import required libraries for machine learning classifiers
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.svm import LinearSVC # Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier #K-nearest Neighbors


# Define our machine learning classifiers variables to store data later
decisionTreeClassifierModel = DecisionTreeClassifier()
logisticRegressionModel = LogisticRegression(max_iter=10000)
linearSVCModel = LinearSVC(dual=False)
kNeighborsModel = KNeighborsClassifier()
# features = attributes columns of our data set
# target = classes column of our data set
# folds = this is added so we can easily change the number of folds we want to do with our data set.

def decisionTreeClassifierModel_evaluation(features, target, folds):   
    decisionTreeClassifier_result = cross_validate(decisionTreeClassifierModel, features, target, cv=folds, scoring=scoring)
    decisionTreeClassifier_table = pd.DataFrame({
      'Decision Tree':[
                        decisionTreeClassifier_result['test_accuracy'].mean(),
                        decisionTreeClassifier_result['test_precision'].mean(),
                        decisionTreeClassifier_result['test_recall'].mean(),
                        decisionTreeClassifier_result['test_sensitvity'].mean(),
                        decisionTreeClassifier_result['test_specificity'].mean(),
                        decisionTreeClassifier_result['test_f1_score'].mean()
                       ],
      },
    index=['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1 Score', ])
    return(decisionTreeClassifier_table)

def logisticRegressionModel_evaluation(features, target, folds):   
    logisticRegression_result     = cross_validate(logisticRegressionModel, features, target, cv=folds, scoring=scoring)
    logisticRegression_table = pd.DataFrame({
        'Logistic Regression':[
                                logisticRegression_result['test_accuracy'].mean(),
                                logisticRegression_result['test_precision'].mean(),
                                logisticRegression_result['test_recall'].mean(),
                                logisticRegression_result['test_sensitvity'].mean(),
                                logisticRegression_result['test_specificity'].mean(),
                                logisticRegression_result['test_f1_score'].mean()
                            ],
      },
      index=['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1 Score', ])
    return(logisticRegression_table)
    
def linearSVCModel_evaluation(features, target, folds):   
    linearSVC_result              = cross_validate(linearSVCModel, features, target, cv=folds, scoring=scoring)
    linearSVC_table = pd.DataFrame({
      'Support Vector Classifier':[
                                    linearSVC_result['test_accuracy'].mean(),
                                    linearSVC_result['test_precision'].mean(),
                                    linearSVC_result['test_recall'].mean(),
                                    linearSVC_result['test_sensitvity'].mean(),
                                    linearSVC_result['test_specificity'].mean(),
                                    linearSVC_result['test_f1_score'].mean()
                                   ],
      },
      index=['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1 Score', ])
    return(linearSVC_table)
    
def kNeighborsModel_evaluation(features, target, folds):   
    kNeighbors_result             = cross_validate(kNeighborsModel, features, target, cv=folds, scoring=scoring)
    kNeighbors_table = pd.DataFrame({
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
    return(kNeighbors_table)
# Let's try to look at our data frame again one last time
dataFrame.head()
# Specify features columns
# Actually what we are doing here is that we are just dropping the Species column since that is our class
# and the remaining columns will then be our features (eg. inputs to come up to a class)
# axis 0 basically means to drop all of that column
features = dataFrame.drop(columns="quality", axis=0)
features
# Don't mind the left hand side, those are just index mainly used for viewing
# Specify target column
# Now we try to get the frame of only our target. Which is the "Species" column
target = dataFrame["quality"]

# Do note that csv files are also zero-index, that means a row starts from zero.
target
DT_evaluation = decisionTreeClassifierModel_evaluation(features, target, 5)
LR_evaluation = logisticRegressionModel_evaluation(features, target, 5)
LSVC_evaluation = linearSVCModel_evaluation(features, target, 5)
KN_evaluation = kNeighborsModel_evaluation(features, target, 5)

# merge data frames from various models
comparison_graph = pd.concat([LR_evaluation, DT_evaluation, KN_evaluation, LSVC_evaluation ], axis=1)

# rename the axis for our graph
view = comparison_graph
view = view.rename_axis('Performance Scores').reset_index() #Add the index names to the column. This will be used for our presentation

# https://pandas.pydata.org/docs/reference/api/pandas.melt.html
view = view.melt(var_name = 'Classifier', value_name = 'Value', id_vars = 'Performance Scores')

# plot the results
sbrn.catplot(data=view, x="Performance Scores", y="Value", hue="Classifier", kind='bar', palette="deep", alpha=1, legend=True, height=10, margin_titles=True, aspect=2)

# Check highest score for every result
comparison_graph['Highest Score'] = comparison_graph.idxmax(axis=1)
comparison_graph

