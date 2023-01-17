# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import yellowbrick
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import RadViz


# Text Processing
import re
import itertools
import spacy
import string
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm
from collections import Counter

# Machine Learning packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import sklearn.cluster as cluster

# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

import pickle as pkl
from scipy import sparse
from numpy import asarray
from numpy import savetxt

# Fix imbalance
from imblearn.under_sampling import InstanceHardnessThreshold

# Model training and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

#Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import classification_report
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
result_svd_vec_types  = pd.read_csv("../input/nlp-to-predict-myers-briggs-personality-type/data/output_csv/result_svd_vec_types.csv")
result_svd_vec_types.drop(["Unnamed: 0"], axis=1, inplace=True)
result_svd_vec_types.head()
result_svd_vec_types.shape
X = result_svd_vec_types.drop(["type","enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj",
                               "intp", "isfj", "isfp", "istj", "istp"], axis=1).values
y = result_svd_vec_types["type"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    specificities = tn / (tn+fp)
    specificity = (specificities.sum())/ 16

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity'  : [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'logit': LogisticRegression(),
          'knn': KNeighborsClassifier(),
          'decisiontree': DecisionTreeClassifier(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
raise SystemExit("Here it comes a very consuming memory process that takes about 45 minutes")
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_svd.csv")
models_df
xgboost = GradientBoostingClassifier().fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10,20))
viz = FeatureImportances(xgboost)
viz.fit(X, y)
viz.show()
#viz.show(outpath="images/output_images/feature_importance_types.png")
sns.set_context("talk")
plt.show()
result_svd_vec_types  = pd.read_csv("../input/nlp-to-predict-myers-briggs-personality-type/data/output_csv/result_svd_vec_types.csv")
result_svd_vec_types.drop(["Unnamed: 0"], axis=1, inplace=True)
result_svd_vec_types.head()
result_svd_vec_types.shape
def sampling_k_elements(group, k=39):
    if len(group) < k:
        return group
    return group.sample(k)

balanced_svd = result_svd_vec_types.groupby("type").apply(sampling_k_elements).reset_index(drop=True)
X = balanced_svd.drop(["type","enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj",
                               "intp", "isfj", "isfp", "istj", "istp"], axis=1).values
y = balanced_svd["type"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    specificities = tn / (tn+fp)
    specificity = (specificities.sum())/ 16

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity': [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'logit': LogisticRegression(),
          'knn': KNeighborsClassifier(),
          'decisiontree': DecisionTreeClassifier(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_svd_resampled.csv")
models_df
result_umap_types  = pd.read_csv("../input/nlp-to-predict-myers-briggs-personality-type/data/output_csv/result_umap_types.csv")
result_umap_types.drop(["Unnamed: 0"], axis=1, inplace=True)
result_umap_types.head()
result_umap_types.shape
X = result_umap_types.drop(["type","enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj",
                               "intp", "isfj", "isfp", "istj", "istp"], axis=1).values
y = result_umap_types["type"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    specificities = tn / (tn+fp)
    specificity = (specificities.sum())/ 16

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity': [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'logit': LogisticRegression(),
          'knn': KNeighborsClassifier(),
          'decisiontree': DecisionTreeClassifier(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_umap.csv")
models_df
result_umap_types  = pd.read_csv("../input/nlp-to-predict-myers-briggs-personality-type/data/output_csv/result_umap_types.csv")
result_umap_types.drop(["Unnamed: 0"], axis=1, inplace=True)
result_umap_types.head()
result_umap_types.shape
def sampling_k_elements(group, k=39):
    if len(group) < k:
        return group
    return group.sample(k)

balanced_umap = result_umap_types.groupby("type").apply(sampling_k_elements).reset_index(drop=True)
X = balanced_umap.drop(["type","enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj",
                               "intp", "isfj", "isfp", "istj", "istp"], axis=1).values
y = balanced_umap["type"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    specificities = tn / (tn+fp)
    specificity = (specificities.sum())/ 16

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity': [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'logit': LogisticRegression(),
          'knn': KNeighborsClassifier(),
          'decisiontree': DecisionTreeClassifier(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_umap_resampled.csv")
models_df
result_umap_svd_types  = pd.read_csv("../input/nlp-to-predict-myers-briggs-personality-type/data/output_csv/result_umap_svd_types.csv")
result_umap_svd_types.drop(["Unnamed: 0"], axis=1, inplace=True)
result_umap_svd_types.head()
result_umap_svd_types.shape
X = result_umap_svd_types.drop(["type","enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj",
                               "intp", "isfj", "isfp", "istj", "istp"], axis=1).values
y = result_umap_svd_types["type"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    specificities = tn / (tn+fp)
    specificity = (specificities.sum())/ 16

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity': [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'logit': LogisticRegression(),
          'knn': KNeighborsClassifier(),
          'decisiontree': DecisionTreeClassifier(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_umap_svd.csv")
models_df
result_umap_svd_types  = pd.read_csv("../input/nlp-to-predict-myers-briggs-personality-type/data/output_csv/result_umap_svd_types.csv")
result_umap_svd_types.drop(["Unnamed: 0"], axis=1, inplace=True)
result_umap_svd_types.head()
result_umap_svd_types.shape
def sampling_k_elements(group, k=39):
    if len(group) < k:
        return group
    return group.sample(k)

balanced_umap_svd = result_umap_svd_types.groupby("type").apply(sampling_k_elements).reset_index(drop=True)
X = balanced_umap_svd.drop(["type","enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj",
                               "intp", "isfj", "isfp", "istj", "istp"], axis=1).values
y = balanced_umap_svd["type"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    specificities = tn / (tn+fp)
    specificity = (specificities.sum())/ 16

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity': [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'logit': LogisticRegression(),
          'knn': KNeighborsClassifier(),
          'decisiontree': DecisionTreeClassifier(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_umap_svd_resampled.csv")
models_df
result_svd_vec_dimensions  = pd.read_csv("../input/nlp-to-predict-myers-briggs-personality-type/data/output_csv/result_svd_vec_dimensions.csv")
result_svd_vec_dimensions.drop(["Unnamed: 0"], axis=1, inplace=True)
result_svd_vec_dimensions.head()
result_svd_vec_dimensions.shape
X = result_svd_vec_dimensions.drop(["type","i-e", "n-s", "t-f", "j-p"], axis=1).values
y = result_svd_vec_dimensions["i-e"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
    specificity = tn / (tn+fp)

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity'  : [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_i-e.csv")
models_df
xgboost = GradientBoostingClassifier().fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10,20))
viz = FeatureImportances(xgboost)
viz.fit(X, y)
viz.show()
#viz.show(outpath="images/output_images/feature_importance_i-e.png")
sns.set_context("talk")
plt.show()
X = result_svd_vec_dimensions.drop(["type","i-e", "n-s", "t-f", "j-p"], axis=1).values
y = result_svd_vec_dimensions["n-s"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
    specificity = tn / (tn+fp)

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity'  : [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_n-s.csv")
models_df
xgboost = GradientBoostingClassifier().fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10,20))
viz = FeatureImportances(xgboost)
viz.fit(X, y)
viz.show()
#viz.show(outpath="images/output_images/feature_importance_n-s.png")
sns.set_context("talk")
plt.show()
X = result_svd_vec_dimensions.drop(["type","i-e", "n-s", "t-f", "j-p"], axis=1).values
y = result_svd_vec_dimensions["t-f"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
    specificity = tn / (tn+fp)

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity'  : [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_t-f.csv")
models_df
xgboost = GradientBoostingClassifier().fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10,20))
viz = FeatureImportances(xgboost)
viz.fit(X, y)
viz.show()
#viz.show(outpath="images/output_images/feature_importance_t-f.png")
sns.set_context("talk")
plt.show()
X = result_svd_vec_dimensions.drop(["type","i-e", "n-s", "t-f", "j-p"], axis=1).values
y = result_svd_vec_dimensions["j-p"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print ((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape))
def baseline_report(model, X_train, X_test, y_train, y_test, name):
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model.fit(X_train, y_train)
    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='recall_weighted', n_jobs=-1))
    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=strat_k_fold, scoring='f1_weighted', n_jobs=-1))
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()
    specificity = tn / (tn+fp)

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'specificity'  : [specificity]
                            })   
    return df_model
models = {'gnb': GaussianNB(),
          'randomforest': RandomForestClassifier(),
          'xgboost': GradientBoostingClassifier(),
          'MLPC': MLPClassifier()
         }
# Evaluation of models
models_df = pd.concat([baseline_report(model, X_train, X_test, y_train, y_test, name) for (name, model) in models.items()])
#models_df.to_csv("data/output_csv/models_j-p.csv")
models_df
xgboost = GradientBoostingClassifier().fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10,20))
viz = FeatureImportances(xgboost)
viz.fit(X, y)
viz.show()
#viz.show(outpath="images/output_images/feature_importance_j-p.png")
sns.set_context("talk")
plt.show()
dimensions = 0.840666 * 0.886559 * 0.846915 * 0.783083
types = 0.624998

print("F1 Scores:")
print("Types =", types,"vs","Dimensions =", dimensions)