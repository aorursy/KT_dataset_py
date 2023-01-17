# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings('always')  



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')



print(f'Data Frame Shape (rows, columns): {df.shape}') 

df.head()

            
sns.pairplot(df, hue="quality")
# Visualize outcome of classes

sns.countplot(data=df, x="quality").set_title("Wine quality")
#sns.relplot(data=df, x="", y="", hue="quality", palette="bright", height=6)
# This is the library to import to be able to use random under sampler balancing technique.

from imblearn.under_sampling import RandomUnderSampler

 

# If you want to know when to balance a data set, just read here:

# https://stats.stackexchange.com/questions/227088/when-should-i-balance-classes-in-a-training-data-set

# But basically if you expect your classes to be equally rare, then you have to balance them.

 

# There are various way to balance a dataset, one basic way is to under sample.

# You can read more about these balancing techniques here: https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html

 

# Setup our Under Sampler

rUnderSampler = RandomUnderSampler(random_state=10) # Actually you can use any number here. This is just a random seed.



# df.drop(columns="Species", axis=0) 

# target = df["Species"]

# Perform random under sampling. Then pass in the features (the one without the classes nga df) and the target (katong puro classes lang based on my tutorial)

dfBalancedFeatures, dfBalanceTarget = rUnderSampler.fit_resample(df.drop(columns="quality", axis=0), df["quality"])

 

# Visualize new classes distributions

sns.countplot(dfBalanceTarget ).set_title('Balanced Data Set')

 

# Now continue to just use the new balanced data frames on your notebook.
print(f'Data Frame Shape (rows, columns): {df.shape}') 

df.isnull().values.any()
df = df.dropna()

df.isnull().values.any()
from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split

def perf_measure(actual, prediction):

    cm = confusion_matrix (actual, prediction)

    FP = cm.sum(axis=0) - np.diag(cm)  

    FN = cm.sum(axis=1) - np.diag(cm)

    TP = np.diag(cm)

    TN = cm.sum() - (FP + FN + TP)



    return(TP, FP, TN, FN)
def sensitivity_score(y_true, y_pred, mode="multiclass"):

    if mode == "multiclass":

        TP, FP, TN, FN = perf_measure(y_true, y_pred)

        TPR = (TP/(TP+FN)).mean()

    elif mode == "binary":

        TP, FP, TN, FN = perf_measure(y_true, y_pred)

        TPR = (TP/(TP+FN))[1] 

    else:

        raise Exception("Mode not recognized!")

    

    return TPR



def specificity_score(y_true, y_pred, mode="multiclass"):

    if mode == "multiclass":

        TP, FP, TN, FN = perf_measure(y_true, y_pred)

        TNR = (TN/(TN+FP)).mean()

    elif mode == "binary":

        TP, FP, TN, FN = perf_measure(y_true, y_pred)

        TNR = (TN/(TN+FP))[1]

    else:

        raise Exception("Mode not recognized!")

    

    return TNR
scoring = {

            'accuracy':make_scorer(accuracy_score), 

            'precision':make_scorer(precision_score, average='weighted',zero_division='warn'),

            'f1_score':make_scorer(f1_score, average='weighted'),

            'recall':make_scorer(recall_score, average='weighted'), 

            'sensitivity':make_scorer(sensitivity_score, mode="multiclass"), 

            'specificity':make_scorer(specificity_score, mode="multiclass"), 

           }
from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.naive_bayes import GaussianNB #Naive Bayes

from sklearn.linear_model import LogisticRegression #Logistic Regression

from sklearn.svm import LinearSVC # Support Vector Machine

# from sklearn.neighbors import KNeighborsClassifier #K-nearest Neighbors

# from sklearn.cluster import KMeans #K-means



# Instantiate the machine learning classifiers

DTClassifier_model = DecisionTreeClassifier()

gaussianNB_model = GaussianNB()

LR_model = LogisticRegression(max_iter=10000)

linearSVC_model = LinearSVC(dual=False)

#kNeighbors_model = KNeighborsClassifier()
def models_evaluation(features, target, folds):    

    # Perform cross-validation to each machine learning classifier

    DTClassifier_result = cross_validate(DTClassifier_model, features, target, cv=folds, scoring=scoring)

    gaussianNB_result = cross_validate(gaussianNB_model, features, target, cv=folds, scoring=scoring)

    LR_result = cross_validate(LR_model, features, target, cv=folds, scoring=scoring)

    linearSVC_result = cross_validate(linearSVC_model, features, target, cv=folds, scoring=scoring)

    

    

    models_scores_table = pd.DataFrame({

      'Decision Tree':[

                        DTClassifier_result['test_accuracy'].mean(),

                        DTClassifier_result['test_precision'].mean(),

                        DTClassifier_result['test_recall'].mean(),

                        DTClassifier_result['test_sensitivity'].mean(),

                        DTClassifier_result['test_specificity'].mean(),

                        DTClassifier_result['test_f1_score'].mean()

                       ],

      'Gaussian Naive Bayes':[

                        gaussianNB_result['test_accuracy'].mean(),

                        gaussianNB_result['test_precision'].mean(),

                        gaussianNB_result['test_recall'].mean(),

                        gaussianNB_result['test_sensitivity'].mean(),

                        gaussianNB_result['test_specificity'].mean(),

                        gaussianNB_result['test_f1_score'].mean()

                              ],

      'Logistic Regression':[

                        LR_result['test_accuracy'].mean(),

                        LR_result['test_precision'].mean(),

                        LR_result['test_recall'].mean(),

                        LR_result['test_sensitivity'].mean(),

                        LR_result['test_specificity'].mean(),

                        LR_result['test_f1_score'].mean()

                            ],

      'Support Vector Classifier':[

                       linearSVC_result['test_accuracy'].mean(),

                       linearSVC_result['test_precision'].mean(),

                       linearSVC_result['test_recall'].mean(),

                       linearSVC_result['test_sensitivity'].mean(),

                       linearSVC_result['test_specificity'].mean(),

                       linearSVC_result['test_f1_score'].mean()

                                   ],

         },



      index=['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1 Score', ])

        

        

    return(models_scores_table)
df.head()
features = df.drop(columns="quality", axis=0)

features
target = df["quality"]

target

evaluationResult = models_evaluation(features, target, 5)

view = evaluationResult

view = view.rename_axis('Test Type').reset_index() #Add the index names to the column. This will be used for our presentation



# https://pandas.pydata.org/docs/reference/api/pandas.melt.html

# Re-Organizing our dataframe to fit our view need

view = view.melt(var_name='Classifier', value_name='Value', id_vars='Test Type')

# result

sns.catplot(data=view, x="Test Type", y="Value", hue="Classifier", kind='bar', palette="bright", alpha=0.8, legend=True, height=5, margin_titles=True, aspect=2)
evaluationResult['Best Score'] = evaluationResult.idxmax(axis=1)

evaluationResult