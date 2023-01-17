# Where to save the figures

PROJECT_ROOT_DIR = "."

datafile_path = "../input/caravan-insurance-challenge.csv"

output_path = "images"



# Cost / benefit:

# Cost to call = -10

# Benefit = 100

# b(Y,p) = b(TP) = 100 - 10 = 90

# c(N,p) = c(FN) = 0

# c(Y,n) = c(FP) = -10

# b(N,n) = b(TN) = 0

TP_amount = 90

FN_amount = 0

FP_amount = -10

TN_amount = 0
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12
# Save figures

def save_fig(fig_id, tight_layout=True):

    path = os.path.join(PROJECT_ROOT_DIR, output_path, fig_id + ".png")

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format='png', dpi=300)
data_all = pd.read_csv(datafile_path)

data_all.head()
data_all.info()

# data_all.list() # List all columns in the dataset
data_all["ORIGIN"].value_counts()
data_all.describe()
%matplotlib inline

import matplotlib.pyplot as plt

#data_all.hist(bins=20, figsize=(50,40))

#save_fig("attributes_histogram_plots")

# plt.show()
data_train = data_all.loc[data_all['ORIGIN'] == 'train']

data_test = data_all.loc[data_all['ORIGIN'] != 'train']

print(len(data_train), "train +", len(data_test), "test")
# Remove the "ORIGIN" attribute, not needed anymore

data_train = data_train.drop("ORIGIN", axis=1)

data_test = data_test.drop("ORIGIN", axis=1)
# Create a copy of the training set we can modify

data = data_train.copy()
# What is the proportion of CARAVAN

data['CARAVAN'].value_counts()

# TODO Print proportion instead
# Scatter plot

corr_matrix = data.corr()
corr_matrix["CARAVAN"].sort_values(ascending=False)
'''

from pandas.plotting import scatter_matrix



attributes = ["CARAVAN", "PPERSAUT", "APERSAUT", "APLEZIER", "PWAPART", "MKOOPKLA", "PBRAND", "PPLEZIER", "MINKGEM", "MOPLLAAG"]

scatter_matrix(data[attributes], figsize=(20, 20))

save_fig("scatter_matrix_plot")

'''
from pandas.plotting import parallel_coordinates



#plt.figure()

#parallel_coordinates(data, 'CARAVAN')
%matplotlib inline

import matplotlib.pyplot as plt

#plt.scatter(data["PPERSAUT"], data["APERSAUT"], c = data["CARAVAN"], alpha=0.8)

#plt.legend()

#plt.grid(True)

# save_fig("XXX")

#plt.show()
#data.plot.scatter(x='PPERSAUT', y='APERSAUT', c='CARAVAN', label='TestThomas', figsize=(8,5), s=50, alpha=1)
target_name = ["NO_INS", "HAS_INS"]

target_columns = ["CARAVAN"]

# y_train = data_train[target_name].copy().values # to convert the dataframe to a NumPy array

y_train = data_train[target_columns].copy()

data = data_train.drop(target_columns, axis=1) # drop labels for training set
y_test = data_test[target_columns].copy()

data_test = data_test.drop(target_columns, axis=1)
from sklearn.base import BaseEstimator, TransformerMixin



# Create a class to select numerical or categorical columns 

# since Scikit-Learn doesn't handle DataFrames yet

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
# Dummification of "MOSTYPE" and "MOSHOOFD" nominal attributes

# TODO Ignore them for the first round !!!

cat_attributes = ["MOSTYPE", "MOSHOOFD"]
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelBinarizer



# Pipeline for categorical attributes

cat_pipeline = Pipeline([

        ('selector', DataFrameSelector(cat_attributes)),

        ('label_binarizer', LabelBinarizer()),

    ])
# Numerical attributes

data_num = data.drop(cat_attributes, axis=1)

num_attributes = list(data_num)

#num_attributes
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler



# Pipeline for numerical attributes

num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attributes)),

        ('imputer', Imputer(strategy="median")),

        ('std_scaler', MinMaxScaler()), # Not necessary for Decision Tree, add this as a parameter to keep that option

#        ('std_scaler', StandardScaler()),

    ])
from sklearn.pipeline import Pipeline



# Union of features created by previous pipelines

from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),

    ])



# TODO Categorical attributes need to be managed properly

full_pipeline = num_pipeline
X_train = full_pipeline.fit_transform(data)
# Be careful I only selected num_attributes here

features_name = num_attributes
# Check the shape of the prepared data

X_train.shape
# Mandatory to transform y_train into a shape accepted by cross_val_score and cross_val_predict methods, with only 1 dimension

c, r = y_train.values.shape

print(c, r)

y_train_old = y_train

y_train = y_train.values.reshape(c,)
X_test = full_pipeline.transform(data_test)

X_test.shape
# Mandatory to transform y_train into a shape accepted by cross_val_score and cross_val_predict methods, with only 1 dimension

c, r = y_test.values.shape

print(c, r)

y_test = y_test.values.reshape(c,)
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel



print("Initial shape: ", X_train.shape)



clf = ExtraTreesClassifier()

clf = clf.fit(X_train, y_train)



model = SelectFromModel(clf, prefit=True)

X_new = model.transform(X_train)

print("New shape: ", X_new.shape)



sorted(zip(clf.feature_importances_, features_name), reverse=True)
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.tree import DecisionTreeClassifier



# Create the RFE object and compute a cross-validated score.

#clf_dt = DecisionTreeClassifier()

svc = SVC(kernel="linear")

dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_leaf_nodes=None, min_impurity_decrease=0, min_impurity_split=None, min_samples_leaf=0.1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')



# The "accuracy" scoring is proportional to the number of correct

# classifications

rfecv = RFECV(estimator=dt, step=1, cv=StratifiedKFold(3),

              scoring='roc_auc', verbose = 0)

rfecv.fit(X_train, y_train)



# print(rfecv.ranking_)



print("Optimal number of features : %d" % rfecv.n_features_)



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
# data_prepared_df.columns[rfecv.support_]
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



KBest = SelectKBest(chi2, k=20)

X_new = KBest.fit_transform(X_train, y_train)

sorted(zip(KBest.scores_, features_name), reverse=True)
# Transform the Numpy array into a Pandas DataFrame

data_prepared_df = pd.DataFrame(X_train, columns = num_attributes, index = list(data_num.index.values))

data_prepared_df.head()

#data_prepared_df.columns.values
# Display scores

def display_scores(scores):

    print('Scores:                        ', scores)

    print('Mean:                          '+"{:.2f}".format(scores.mean()))

    print('Standard deviation:            '+"{:.2f}".format(scores.std()))
# Display confusion matrix

def display_conf_matrix(y_train, y_pred):

    # Confusion matrix: row -> actual, column -> predicted

    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(y_train, y_pred))
# Precision, recall, AUC, F1-score

def display_perf_metrics(y_train, y_pred, y_proba):

    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, cohen_kappa_score

    print('Accuracy           '+"{:.2f}".format(accuracy_score(y_train, y_pred)*100)+'%')

    print('Precision          '+"{:.2f}".format(precision_score(y_train, y_pred)*100)+'%')

    print('Recall             '+"{:.2f}".format(recall_score(y_train, y_pred)*100)+'%')

    print('F1                 '+"{:.2f}".format(f1_score(y_train, y_pred)))

    print('AUC                '+"{:.2f}".format(roc_auc_score(y_train, y_proba)))

    print('Cohen Kappa        '+"{:.2f}".format(cohen_kappa_score(y_train, y_pred)))
# Plot precision / recall

def plot_precision_vs_recall(precisions, recalls):

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])
# Plot ROC curve

def plot_roc_curve(fpr, tpr, label=None):

    from sklearn.metrics import auc

    plt.plot(fpr, tpr, linewidth=2, label=(label + ' (area = %0.2f)' % auc(fpr, tpr)))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.legend(loc="lower right")

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)
# Plot precision and recall vs threashold

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    plt.xlabel("Threshold")

    plt.legend(loc="upper left")

    plt.ylim([0, 1])
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - An object to be used as a cross-validation generator.

          - An iterable yielding train/test splits.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : integer, optional

        Number of jobs to run in parallel (default 1).

    """

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
# Profit calculation

def calculate_profit(cm, FN_amount, TP_amount, TN_amount, FP_amount):

    # cm: confusion matrix

    # FN_amount: benefit when false negative

    # TP_amount: benefit when true positive

    # TN_amount: benefit when true negative

    # FP_amount: benefit when false positive

    return (cm[0][0]*TN_amount + cm[1][0]*FN_amount + cm[0][1]*FP_amount +

                   cm[1][1]*TP_amount)
# Create ranking and profit data frame

def create_ranking_profit_df(name, y_actual, y_proba):

    # Write a CSV file with results

    df_scores = pd.DataFrame({

            'score': y_proba[:,1],

            'actual': y_actual,

        })

    

    df_scores = df_scores.sort_values("score", ascending = False)

    df_scores["profit"] = (TP_amount - FP_amount) * df_scores["actual"] + FP_amount

    df_scores["cum_profit"] = df_scores["profit"].cumsum(axis = 0)

    # print(df_scores["cum_profit"].values)

    # print(ranking_dt.head())

    

    # Save a copy of the dataframe

    path = os.path.join(PROJECT_ROOT_DIR, output_path, "df_ranking_profit_" + name + ".csv")

    df_scores.to_csv(path)

    

    return df_scores
# Plot profit curve

def plot_profit_curve(name, df_scores):

    plt.plot(df_scores["cum_profit"].values, linewidth=1, label=name + " Profit curve")

    plt.plot([0, 0], [0, 1], 'k--')

    plt.legend(loc="lower left")

    plt.xlabel('Number of test instances', fontsize=16)

    plt.ylabel('Profit', fontsize=16)

    plt.grid(True)
# Create the model

from sklearn.tree import DecisionTreeClassifier



clf_dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)

clf_dt.fit(X_train, y_train)
from sklearn import tree

import graphviz

dot_data = tree.export_graphviz(clf_dt, out_file=None, 

                         feature_names=features_name,  

                         class_names=target_name,  

                         filled=True, rounded=True,  

                         special_characters=True) 

graph = graphviz.Source(dot_data)

#graph.format = 'png'

graph.render(os.path.join(PROJECT_ROOT_DIR, output_path, "decision_tree"))

graph
print(X_train.shape)

print(y_train.shape)
# Use 10-fold cross validation to have a first view on model accuracy

# data_predicted = model_dtree.predict(data_scaled)

from sklearn.model_selection import cross_val_score

score_dt = cross_val_score(clf_dt, X_train, y_train, cv=10, scoring="roc_auc", n_jobs=1)
# Prediction

from sklearn.model_selection import cross_val_predict

y_pred_dt = cross_val_predict(clf_dt, X_train, y_train, cv=10, n_jobs=-1)

# predict_proba is the method to call on the DecisionTree classifier (http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict_proba)

y_proba_dt = cross_val_predict(clf_dt, X_train, y_train, cv=10, method='predict_proba', n_jobs=-1)
print(y_pred_dt)

print(y_proba_dt)

print(score_dt)
# Write a CSV file with results

df_y_proba_dt = pd.DataFrame({

        'score': y_proba_dt[:,1],

        'actual': y_train,

        'pred': y_pred_dt

    })

#print(df_y_proba_dt.sort_values('score', axis=0, ascending=False))

df_y_proba_dt.to_csv('images/y_proba_dt.csv')
# Precision, recalls and thresholds

from sklearn.metrics import precision_recall_curve

precisions_dt, recalls_dt, thresholds_dt = precision_recall_curve(y_train, y_proba_dt[:,1])

print(precisions_dt)

print(recalls_dt)

print(thresholds_dt)
# Display scores

display_scores(score_dt)

display_conf_matrix(y_train, y_pred_dt)

display_perf_metrics(y_train, y_pred_dt, y_proba_dt[:,1])
# Classification report

from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred_dt, target_names=target_name))
# Plot precision / recall

#plt.figure(figsize=(8, 6))

plot_precision_vs_recall(precisions_dt, recalls_dt)

save_fig("precision_vs_recall_plot")

plt.show()
# ROC curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_proba_dt[:,1])



#plt.figure(figsize=(8, 6))

plot_roc_curve(fpr, tpr, "Decision tree")

save_fig("roc_curve_plot")

plt.show()
# Create classifiers

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier



clf_log = LogisticRegression(n_jobs=-1, random_state = 42)

clf_rf = RandomForestClassifier(n_jobs=-1, random_state = 42)

clf_svm = SVC(kernel = 'rbf', probability = True, random_state = 42)



clf_voting = VotingClassifier(

    estimators=[('lr', clf_log), ('rf', clf_rf), ('svc', clf_svm)],

    voting='soft')



# Create a dataframe to manage the different classifiers

clf_list_values = [ ('Decision Tree', clf_dt),

                    ('Logistic Regression', clf_log),

                    ('Random Forrest', clf_rf),

                    ('SVM', clf_svm),

                    ('Voting LR, SVM, RF', clf_voting)

                ]

clf_list = pd.DataFrame.from_records(clf_list_values, columns=['name', 'clf'])
from sklearn.metrics import roc_curve, auc

from cycler import cycler



print('       Models Performance       ')

print('--------------------------------')



# plt.figure(figsize=(8, 6))

plt.rc('lines', linewidth=1)

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k']) +

                           cycler('linestyle', ['-', '--', ':', '-.','-', '--', ':', '-.'])))



for index, row in clf_list.iterrows():

    name = row['name']

    clf = row['clf']

    clf.fit(X_train, y_train)

    y_pred = cross_val_predict(clf, X_train, y_train, cv=10, n_jobs=-1)

    y_proba = cross_val_predict(clf, X_train, y_train, cv=10, method='predict_proba', n_jobs=-1)

    print("--- ", clf.__class__.__name__, " ---")

    display_perf_metrics(y_train, y_pred, y_proba[:,1]) # TODO Fix issue with SVM on precision and F1 score...

    display_conf_matrix(y_train, y_pred)

    fpr, tpr, thresholds = roc_curve(y_train, y_proba[:,1])

    plt.plot(fpr, tpr, linewidth=1, label=name+" (area = %0.2f)" % auc(fpr, tpr))



#plot_roc_curve(fpr, tpr, "Decision tree")

plt.plot([0, 1], [0, 1], 'k--')

plt.axis([0, 1, 0, 1])

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.legend(loc="lower right", fontsize=16)

save_fig("roc_curve_comparison_plot")

plt.show()
from sklearn.metrics import confusion_matrix

clf_rf

y_pred = cross_val_predict(clf_rf, X_train, y_train, cv=10, n_jobs=-1)

cm_rf_tmp = confusion_matrix(y_train, y_pred)

cm_rf_tmp
from sklearn.model_selection import GridSearchCV

from scipy.stats import randint



param_grid = [

        {'max_depth': [5, 8, 10, 15, 20], 'max_features': [10, 20, 50, 80], 'min_samples_leaf': [0.0001, 0.001, 0.01, 0.1, 0.2]},

        #{'max_depth': [5, 50, 100], 'max_features': [10, 50], 'min_samples_leaf': [0.0001, 0.01, 0.1, 0.2]},

    ]



grid_search_dt = GridSearchCV(clf_dt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

grid_search_dt.fit(X_train, y_train)



print("AUC= ", grid_search_dt.best_score_, grid_search_dt.best_params_)

grid_search_dt.best_estimator_
cvres_grid = grid_search_dt.cv_results_

for mean_score, params in zip(cvres_grid["mean_test_score"], cvres_grid["params"]):

    print("AUC= ", mean_score, params)
feature_importances_grid = grid_search_dt.best_estimator_.feature_importances_

sorted(zip(feature_importances_grid, features_name), reverse=True)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'max_depth': randint(low=4, high=20),

        'max_features': randint(low=10, high=80),

#        'max_leaf_nodes': randint(low=10, high=1000),

        'min_samples_leaf': randint(low=1, high=100)

    }



rnd_search = RandomizedSearchCV(clf_dt, param_distributions=param_distribs,

                                n_iter=200, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)

#print(clf_dt.get_params().keys())

rnd_search.fit(X_train, y_train)
print("AUC= ", rnd_search.best_score_, rnd_search.best_params_)

rnd_search.best_estimator_
cvres_rnd = rnd_search.cv_results_

for mean_score, params in zip(cvres_rnd["mean_test_score"], cvres_rnd["params"]):

    print("AUC= ", mean_score, params)
feature_importances_rnd = rnd_search.best_estimator_.feature_importances_

sorted(zip(feature_importances_rnd, features_name), reverse=True)
clf_best_dt = grid_search_dt.best_estimator_

clf_best_dt
y_pred_dt = cross_val_predict(clf_best_dt, X_train, y_train, cv=10, n_jobs=-1)

y_proba_dt = cross_val_predict(clf_best_dt, X_train, y_train, cv=10, method='predict_proba', n_jobs=-1)

ranking_profit_dt = create_ranking_profit_df("Decision Tree", y_train, y_proba_dt)
ranking_profit_dt.head()
# Performance metrics

from sklearn.metrics import confusion_matrix

cm_dt = confusion_matrix(y_train, y_pred_dt)

display_conf_matrix(y_train, y_pred_dt)

display_perf_metrics(y_train, y_pred_dt, y_proba_dt[:,1])
# Precision, recalls and thresholds

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba_dt[:,1])

#print(precisions)

#print(recalls)

#print(thresholds)
# Plot precision / recall

plot_precision_vs_recall(precisions, recalls)

save_fig("precision_vs_recall_plot_optimized_model_dt")

plt.show()
# Plot precision and recall vs threshold

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
# ROC curve

from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_train, y_proba_dt[:,1])



#plt.figure(figsize=(8, 6))

plot_roc_curve(fpr, tpr, "Decision tree")

save_fig("roc_curve_plot_optimized_model_dt")

plt.show()
from sklearn.model_selection import GridSearchCV

from scipy.stats import randint



param_grid = [

        {'solver' : ['newton-cg', 'lbfgs', 'sag'], 'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },

        {'solver' : ['liblinear'], 'penalty': ['l1','l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

    ]



grid_search_log = GridSearchCV(clf_log, param_grid, cv=5, scoring='precision', n_jobs=-1)

grid_search_log.fit(X_train, y_train)



print("Precision= ", grid_search_log.best_score_, grid_search_log.best_params_)

grid_search_log.best_estimator_
# I can try to use my own scorer based on the max profit.

# from sklearn.metrics import make_scorer

# benefit_scorer = make_scorer(calculate_profit, greater_is_better=True)
cvres_grid_log = grid_search_log.cv_results_

for mean_score, params in zip(cvres_grid_log["mean_test_score"], cvres_grid_log["params"]):

    print("Precision= ", mean_score, params)
clf_best_log = grid_search_log.best_estimator_

clf_best_log
y_pred_log = cross_val_predict(clf_best_log, X_train, y_train, cv=10, n_jobs=-1)

y_proba_log = cross_val_predict(clf_best_log, X_train, y_train, cv=10, method='predict_proba', n_jobs=-1)

ranking_profit_log = create_ranking_profit_df("Logistic Regression", y_train, y_proba_log)
ranking_profit_log.head()
# Performance metrics

cm_log = confusion_matrix(y_train, y_pred_log)

display_conf_matrix(y_train, y_pred_log)

display_perf_metrics(y_train, y_pred_log, y_proba_log[:,1])
# Precision, recalls and thresholds

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba_log[:,1])

#print(precisions)

#print(recalls)

#print(thresholds)
# Plot precision / recall

plot_precision_vs_recall(precisions, recalls)

save_fig("precision_vs_recall_plot_optimized_model_dt")

plt.show()
# Plot precision and recall vs threshold

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
# ROC curve

from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_train, y_proba_log[:,1])



#plt.figure(figsize=(8, 6))

plot_roc_curve(fpr, tpr, "Decision tree")

save_fig("roc_curve_plot_optimized_model_dt")

plt.show()
# Select best features from previous decision tree

model = SelectFromModel(clf_best_dt, prefit=True)

X_new = model.transform(X_train)

print("Nb of features selected: ", X_new.shape)
from sklearn.model_selection import GridSearchCV

from scipy.stats import randint



param_grid = [

        {'max_depth': [5, 8, 10, 15, 20], 'max_features': [5, 8, 13], 'min_samples_leaf': [0.0001, 0.001, 0.01, 0.1, 0.2]},

    ]



grid_search_rf = GridSearchCV(clf_rf, param_grid, cv=5, scoring='f1', n_jobs=-1)

grid_search_rf.fit(X_new, y_train)



print("F1= ", grid_search_rf.best_score_, grid_search_rf.best_params_)

clf_rf2 = grid_search_rf.best_estimator_

clf_rf2
cvres_grid_rf = grid_search_rf.cv_results_

for mean_score, params in zip(cvres_grid_rf["mean_test_score"], cvres_grid_rf["params"]):

    print("F1= ", mean_score, params)
clf_best_rf = grid_search_rf.best_estimator_

clf_best_rf
y_pred_rf = cross_val_predict(clf_best_rf, X_new, y_train, cv=10, n_jobs=-1)

y_proba_rf = cross_val_predict(clf_best_rf, X_new, y_train, cv=10, method='predict_proba', n_jobs=-1)

ranking_profit_rf = create_ranking_profit_df("Random Forest", y_train, y_proba_rf)
# Performance metrics

cm_rf = confusion_matrix(y_train, y_pred_rf)

display_conf_matrix(y_train, y_pred_rf)

display_perf_metrics(y_train, y_pred_rf, y_proba_rf[:,1])
# Precision, recalls and thresholds

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba_rf[:,1])

#print(precisions)

#print(recalls)

#print(thresholds)
# Plot precision / recall

plot_precision_vs_recall(precisions, recalls)

save_fig("precision_vs_recall_plot_optimized_model_rf")

plt.show()
# Plot precision and recall vs threshold

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

save_fig("precision_recall_vs_threshold_plot_optimized_model_rf")

plt.show()
# ROC curve

from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_train, y_proba_rf[:,1])



#plt.figure(figsize=(8, 6))

plot_roc_curve(fpr, tpr, "Decision tree")

save_fig("roc_curve_plot_optimized_model_rf")

plt.show()
from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



#train_sizes, train_scores, valid_scores = learning_curve(

#    clf_best_rf, X_train, y_train, train_sizes=[50, 80, 110], cv=5)

#print(train_sizes)

#print(train_scores)

#print(valid_scores)



title = "Learning Curves (Decision Tree)"

#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(clf_best_dt, title, X_train, y_train, (0.8, 1.01), cv=10, n_jobs=7, train_sizes=[ 0.1, 0.33, 0.55, 0.78, 1.])



plt.show()
from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



title = "Learning Curves (Logistic Regression)"

#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(clf_best_log, title, X_train, y_train, (0.8, 1.01), cv=10, n_jobs=1, train_sizes=[ 0.1, 0.33, 0.55, 0.78, 1.])



plt.show()
from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



title = "Learning Curves (Random Forest)"

# Cross validation with 100 iterations to get smoother mean test and train

# score curves, each time with 20% data randomly selected as a validation set.

#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

#plot_learning_curve(clf_best_rf, title, X_train, y_train, ylim=(0.8, 1.01), cv=cv, n_jobs=7)

plot_learning_curve(clf_best_rf, title, X_train, y_train, (0.8, 1.01), cv=10, n_jobs=7, train_sizes=[ 0.1, 0.33, 0.55, 0.78, 1.])



plt.show()
print(cm_dt)

print(cm_log)

print(cm_rf)
print('Compare Profit from Models')

print('-------------------------------------')



Profit_DT = calculate_profit(cm_dt, FN_amount, TP_amount, TN_amount, FP_amount)

print('Decision Tree Profit:        ' + str(Profit_DT))



Profit_LOG = calculate_profit(cm_log, FN_amount, TP_amount, TN_amount, FP_amount)

print('Logistic Regression Profit:  ' + str(Profit_LOG))



Profit_RF = calculate_profit(cm_rf, FN_amount, TP_amount, TN_amount, FP_amount)

print('Random Forest Profit:        ' + str(Profit_RF))
# Plot profit curve

plt.figure(figsize=(8, 6))

plot_profit_curve("Decision tree", ranking_profit_dt)

plot_profit_curve("Logistic regression", ranking_profit_log)

plot_profit_curve("Random forest", ranking_profit_rf)

save_fig("profit_curve")

plt.show()
y_pred_final = cross_val_predict(clf_best_dt, X_test, y_test, cv=10, n_jobs=-1)

y_proba_final = cross_val_predict(clf_best_dt, X_test, y_test, cv=10, method='predict_proba', n_jobs=-1)

ranking_profit_final = create_ranking_profit_df("Decision Tree", y_test, y_proba_final)
ranking_profit_final
plot_profit_curve("Decision tree on test set", ranking_profit_final)