import pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC

from sklearn.ensemble import VotingClassifier

from sklearn.feature_selection import RFECV
import warnings

warnings.filterwarnings('ignore')
initial_data = pd.read_csv('../input/data.csv')
initial_data.head()
initial_data.drop(initial_data.columns[0], axis=1, inplace=True)
for column in initial_data.columns:

    if "Unnamed" in column:

        initial_data.drop(column, axis = 1, inplace=True)
initial_data.head()
initial_data['diagnosis']=initial_data['diagnosis'].map({'M':1,'B':0})
initial_data.head()
X = initial_data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean','fractal_dimension_mean',

                 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',

                 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst','fractal_dimension_worst']]



y = initial_data['diagnosis']
col_labels = ['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean','fractal_dimension_mean',

              'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',

              'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst','fractal_dimension_worst'

             ]

initial_data.columns = col_labels
for c in col_labels:

    no_missing = initial_data[c].isnull().sum()

    if no_missing > 0:

        print(c)

        print(no_missing)

    else:

        print(c)

        print("No missing values")

        print(' ')
import seaborn as sns

import matplotlib.pyplot as plt



sns.countplot(initial_data['diagnosis'],label="Sum")



plt.show()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.34)
LR = LogisticRegression()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LR_fit_time = scores['fit_time'].mean()

LR_score_time = scores['score_time'].mean()

LR_accuracy = scores['test_accuracy'].mean()

LR_precision = scores['test_precision_macro'].mean()

LR_recall = scores['test_recall_macro'].mean()

LR_f1 = scores['test_f1_weighted'].mean()

LR_roc = scores['test_roc_auc'].mean()
decision_tree = DecisionTreeClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

dtree_fit_time = scores['fit_time'].mean()

dtree_score_time = scores['score_time'].mean()

dtree_accuracy = scores['test_accuracy'].mean()

dtree_precision = scores['test_precision_macro'].mean()

dtree_recall = scores['test_recall_macro'].mean()

dtree_f1 = scores['test_f1_weighted'].mean()

dtree_roc = scores['test_roc_auc'].mean()
SVM = SVC(probability = True)



scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

SVM_fit_time = scores['fit_time'].mean()

SVM_score_time = scores['score_time'].mean()

SVM_accuracy = scores['test_accuracy'].mean()

SVM_precision = scores['test_precision_macro'].mean()

SVM_recall = scores['test_recall_macro'].mean()

SVM_f1 = scores['test_f1_weighted'].mean()

SVM_roc = scores['test_roc_auc'].mean()
LDA = LinearDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LDA_fit_time = scores['fit_time'].mean()

LDA_score_time = scores['score_time'].mean()

LDA_accuracy = scores['test_accuracy'].mean()

LDA_precision = scores['test_precision_macro'].mean()

LDA_recall = scores['test_recall_macro'].mean()

LDA_f1 = scores['test_f1_weighted'].mean()

LDA_roc = scores['test_roc_auc'].mean()
QDA = QuadraticDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

QDA_fit_time = scores['fit_time'].mean()

QDA_score_time = scores['score_time'].mean()

QDA_accuracy = scores['test_accuracy'].mean()

QDA_precision = scores['test_precision_macro'].mean()

QDA_recall = scores['test_recall_macro'].mean()

QDA_f1 = scores['test_f1_weighted'].mean()

QDA_roc = scores['test_roc_auc'].mean()
random_forest = RandomForestClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

forest_fit_time = scores['fit_time'].mean()

forest_score_time = scores['score_time'].mean()

forest_accuracy = scores['test_accuracy'].mean()

forest_precision = scores['test_precision_macro'].mean()

forest_recall = scores['test_recall_macro'].mean()

forest_f1 = scores['test_f1_weighted'].mean()

forest_roc = scores['test_roc_auc'].mean()
KNN = KNeighborsClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

KNN_fit_time = scores['fit_time'].mean()

KNN_score_time = scores['score_time'].mean()

KNN_accuracy = scores['test_accuracy'].mean()

KNN_precision = scores['test_precision_macro'].mean()

KNN_recall = scores['test_recall_macro'].mean()

KNN_f1 = scores['test_f1_weighted'].mean()

KNN_roc = scores['test_roc_auc'].mean()
bayes = GaussianNB()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

bayes_fit_time = scores['fit_time'].mean()

bayes_score_time = scores['score_time'].mean()

bayes_accuracy = scores['test_accuracy'].mean()

bayes_precision = scores['test_precision_macro'].mean()

bayes_recall = scores['test_recall_macro'].mean()

bayes_f1 = scores['test_f1_weighted'].mean()

bayes_roc = scores['test_roc_auc'].mean()
models_initial = pd.DataFrame({

    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],

    'Fitting time': [LR_fit_time, dtree_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],

    'Scoring time': [LR_score_time, dtree_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],

    'Accuracy'    : [LR_accuracy, dtree_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],

    'Precision'   : [LR_precision, dtree_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],

    'Recall'      : [LR_recall, dtree_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],

    'F1_score'    : [LR_f1, dtree_f1, SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],

    'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],

    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])



models_initial.sort_values(by='Accuracy', ascending=False)
correlation = initial_data.corr()



mask = np.zeros_like(correlation, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(20, 20))



cmap = sns.diverging_palette(180, 20, as_cmap=True)

sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.show()
X_corr = initial_data[['smoothness_mean', 'radius_se', 'texture_se', 'smoothness_se', 'symmetry_se', 

                       'fractal_dimension_se', 'texture_worst', 'symmetry_worst','fractal_dimension_worst']]

y_corr = initial_data['diagnosis']
correlation = X_corr.corr()



mask = np.zeros_like(correlation, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(5, 5))



cmap = sns.diverging_palette(180, 20, as_cmap=True)

sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.show()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.34)
LR = LogisticRegression()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LR_fit_time = scores['fit_time'].mean()

LR_score_time = scores['score_time'].mean()

LR_accuracy = scores['test_accuracy'].mean()

LR_precision = scores['test_precision_macro'].mean()

LR_recall = scores['test_recall_macro'].mean()

LR_f1 = scores['test_f1_weighted'].mean()

LR_roc = scores['test_roc_auc'].mean()
decision_tree = DecisionTreeClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

dtree_fit_time = scores['fit_time'].mean()

dtree_score_time = scores['score_time'].mean()

dtree_accuracy = scores['test_accuracy'].mean()

dtree_precision = scores['test_precision_macro'].mean()

dtree_recall = scores['test_recall_macro'].mean()

dtree_f1 = scores['test_f1_weighted'].mean()

dtree_roc = scores['test_roc_auc'].mean()
SVM = SVC(probability = True)



scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

SVM_fit_time = scores['fit_time'].mean()

SVM_score_time = scores['score_time'].mean()

SVM_accuracy = scores['test_accuracy'].mean()

SVM_precision = scores['test_precision_macro'].mean()

SVM_recall = scores['test_recall_macro'].mean()

SVM_f1 = scores['test_f1_weighted'].mean()

SVM_roc = scores['test_roc_auc'].mean()
LDA = LinearDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LDA_fit_time = scores['fit_time'].mean()

LDA_score_time = scores['score_time'].mean()

LDA_accuracy = scores['test_accuracy'].mean()

LDA_precision = scores['test_precision_macro'].mean()

LDA_recall = scores['test_recall_macro'].mean()

LDA_f1 = scores['test_f1_weighted'].mean()

LDA_roc = scores['test_roc_auc'].mean()
QDA = QuadraticDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

QDA_fit_time = scores['fit_time'].mean()

QDA_score_time = scores['score_time'].mean()

QDA_accuracy = scores['test_accuracy'].mean()

QDA_precision = scores['test_precision_macro'].mean()

QDA_recall = scores['test_recall_macro'].mean()

QDA_f1 = scores['test_f1_weighted'].mean()

QDA_roc = scores['test_roc_auc'].mean()
random_forest = RandomForestClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

forest_fit_time = scores['fit_time'].mean()

forest_score_time = scores['score_time'].mean()

forest_accuracy = scores['test_accuracy'].mean()

forest_precision = scores['test_precision_macro'].mean()

forest_recall = scores['test_recall_macro'].mean()

forest_f1 = scores['test_f1_weighted'].mean()

forest_roc = scores['test_roc_auc'].mean()
KNN = KNeighborsClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

KNN_fit_time = scores['fit_time'].mean()

KNN_score_time = scores['score_time'].mean()

KNN_accuracy = scores['test_accuracy'].mean()

KNN_precision = scores['test_precision_macro'].mean()

KNN_recall = scores['test_recall_macro'].mean()

KNN_f1 = scores['test_f1_weighted'].mean()

KNN_roc = scores['test_roc_auc'].mean()
bayes = GaussianNB()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

bayes_fit_time = scores['fit_time'].mean()

bayes_score_time = scores['score_time'].mean()

bayes_accuracy = scores['test_accuracy'].mean()

bayes_precision = scores['test_precision_macro'].mean()

bayes_recall = scores['test_recall_macro'].mean()

bayes_f1 = scores['test_f1_weighted'].mean()

bayes_roc = scores['test_roc_auc'].mean()
models_correlation = pd.DataFrame({

    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],

    'Fitting time': [LR_fit_time, dtree_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],

    'Scoring time': [LR_score_time, dtree_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],

    'Accuracy'    : [LR_accuracy, dtree_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],

    'Precision'   : [LR_precision, dtree_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],

    'Recall'      : [LR_recall, dtree_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],

    'F1_score'    : [LR_f1, dtree_f1, SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],

    'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],

    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])



models_correlation.sort_values(by='Accuracy', ascending=False)
models = [LogisticRegression(),

         DecisionTreeClassifier(),

         SVC(probability = True),

         LinearDiscriminantAnalysis(),

         QuadraticDiscriminantAnalysis(),

         RandomForestClassifier(),

         KNeighborsClassifier(),

         GaussianNB()]



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.34)
for model in models:

    scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=20)

    #print(model, scores['fit_time'].mean(), scores['score_time'].mean(), scores['test_accuracy'].mean(),

          #scores['test_precision_macro'].mean(), scores['test_recall_macro'].mean(), 

          #scores['test_f1_weighted'].mean(), scores['test_roc_auc'].mean())
models_ens = list(zip(['LR', 'DT', 'SVM', 'LDA', 'QDA', 'RF', 'KNN', 'NB'], models))



model_ens = VotingClassifier(estimators = models_ens, voting = 'hard')

model_ens.fit(X_train, y_train)

pred = model_ens.predict(X_test)

#prob = model_ens.predict_proba(X_test)[:,1]



acc_hard = accuracy_score(y_test, pred)

prec_hard = precision_score(y_test, pred)

recall_hard = recall_score(y_test, pred)

f1_hard = f1_score(y_test, pred)

roc_auc_hard = 'not applicable'
model_ens = VotingClassifier(estimators = models_ens, voting = 'soft')

model_ens.fit(X_train, y_train)

pred = model_ens.predict(X_test)

prob = model_ens.predict_proba(X_test)[:,1]



acc_soft = accuracy_score(y_test, pred)

prec_soft = precision_score(y_test, pred)

recall_soft = recall_score(y_test, pred)

f1_soft = f1_score(y_test, pred)

roc_auc_soft = roc_auc_score(y_test, prob)
models_ensembling = pd.DataFrame({

    'Model'       : ['Ensebling_hard', 'Ensembling_soft'],

    'Accuracy'    : [acc_hard, acc_soft],

    'Precision'   : [prec_hard, prec_soft],

    'Recall'      : [recall_hard, recall_soft],

    'F1_score'    : [f1_hard, f1_soft],

    'AUC_ROC'     : [roc_auc_hard, roc_auc_soft],

    }, columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])



models_ensembling.sort_values(by='Accuracy', ascending=False)
X.shape
lsvc = LinearSVC(C=0.05, penalty="l1", dual=False).fit(X, y)

model = SelectFromModel(lsvc, prefit=True)

X_svc = model.transform(X)

X_svc.shape #reduction from 30 to 10 features
X_train, X_test, y_train, y_test = train_test_split(X_svc,y,test_size=0.34)
LR = LogisticRegression()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LR_fit_time = scores['fit_time'].mean()

LR_score_time = scores['score_time'].mean()

LR_accuracy = scores['test_accuracy'].mean()

LR_precision = scores['test_precision_macro'].mean()

LR_recall = scores['test_recall_macro'].mean()

LR_f1 = scores['test_f1_weighted'].mean()

LR_roc = scores['test_roc_auc'].mean()
decision_tree = DecisionTreeClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

dtree_fit_time = scores['fit_time'].mean()

dtree_score_time = scores['score_time'].mean()

dtree_accuracy = scores['test_accuracy'].mean()

dtree_precision = scores['test_precision_macro'].mean()

dtree_recall = scores['test_recall_macro'].mean()

dtree_f1 = scores['test_f1_weighted'].mean()

dtree_roc = scores['test_roc_auc'].mean()
SVM = SVC(probability = True)



scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

SVM_fit_time = scores['fit_time'].mean()

SVM_score_time = scores['score_time'].mean()

SVM_accuracy = scores['test_accuracy'].mean()

SVM_precision = scores['test_precision_macro'].mean()

SVM_recall = scores['test_recall_macro'].mean()

SVM_f1 = scores['test_f1_weighted'].mean()

SVM_roc = scores['test_roc_auc'].mean()
LDA = LinearDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LDA_fit_time = scores['fit_time'].mean()

LDA_score_time = scores['score_time'].mean()

LDA_accuracy = scores['test_accuracy'].mean()

LDA_precision = scores['test_precision_macro'].mean()

LDA_recall = scores['test_recall_macro'].mean()

LDA_f1 = scores['test_f1_weighted'].mean()

LDA_roc = scores['test_roc_auc'].mean()
QDA = QuadraticDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

QDA_fit_time = scores['fit_time'].mean()

QDA_score_time = scores['score_time'].mean()

QDA_accuracy = scores['test_accuracy'].mean()

QDA_precision = scores['test_precision_macro'].mean()

QDA_recall = scores['test_recall_macro'].mean()

QDA_f1 = scores['test_f1_weighted'].mean()

QDA_roc = scores['test_roc_auc'].mean()
random_forest = RandomForestClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

forest_fit_time = scores['fit_time'].mean()

forest_score_time = scores['score_time'].mean()

forest_accuracy = scores['test_accuracy'].mean()

forest_precision = scores['test_precision_macro'].mean()

forest_recall = scores['test_recall_macro'].mean()

forest_f1 = scores['test_f1_weighted'].mean()

forest_roc = scores['test_roc_auc'].mean()
KNN = KNeighborsClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

KNN_fit_time = scores['fit_time'].mean()

KNN_score_time = scores['score_time'].mean()

KNN_accuracy = scores['test_accuracy'].mean()

KNN_precision = scores['test_precision_macro'].mean()

KNN_recall = scores['test_recall_macro'].mean()

KNN_f1 = scores['test_f1_weighted'].mean()

KNN_roc = scores['test_roc_auc'].mean()
bayes = GaussianNB()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

bayes_fit_time = scores['fit_time'].mean()

bayes_score_time = scores['score_time'].mean()

bayes_accuracy = scores['test_accuracy'].mean()

bayes_precision = scores['test_precision_macro'].mean()

bayes_recall = scores['test_recall_macro'].mean()

bayes_f1 = scores['test_f1_weighted'].mean()

bayes_roc = scores['test_roc_auc'].mean()
models_sfm = pd.DataFrame({

    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],

    'Fitting time': [LR_fit_time, dtree_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],

    'Scoring time': [LR_score_time, dtree_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],

    'Accuracy'    : [LR_accuracy, dtree_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],

    'Precision'   : [LR_precision, dtree_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],

    'Recall'      : [LR_recall, dtree_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],

    'F1_score'    : [LR_f1, dtree_f1, SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],

    'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],

    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])



models_sfm.sort_values(by='Accuracy', ascending=False)
lsvc = LinearSVC(C=0.05, penalty="l1", dual=False)

model = RFECV(estimator=lsvc, step=1, cv=20).fit(X,y)

X_rfecv = model.transform(X)

X_rfecv.shape #reduction from 30
X_train, X_test, y_train, y_test = train_test_split(X_rfecv,y,test_size=0.34)
LR = LogisticRegression()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LR_fit_time = scores['fit_time'].mean()

LR_score_time = scores['score_time'].mean()

LR_accuracy = scores['test_accuracy'].mean()

LR_precision = scores['test_precision_macro'].mean()

LR_recall = scores['test_recall_macro'].mean()

LR_f1 = scores['test_f1_weighted'].mean()

LR_roc = scores['test_roc_auc'].mean()
decision_tree = DecisionTreeClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

dtree_fit_time = scores['fit_time'].mean()

dtree_score_time = scores['score_time'].mean()

dtree_accuracy = scores['test_accuracy'].mean()

dtree_precision = scores['test_precision_macro'].mean()

dtree_recall = scores['test_recall_macro'].mean()

dtree_f1 = scores['test_f1_weighted'].mean()

dtree_roc = scores['test_roc_auc'].mean()
SVM = SVC(probability = True)



scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

SVM_fit_time = scores['fit_time'].mean()

SVM_score_time = scores['score_time'].mean()

SVM_accuracy = scores['test_accuracy'].mean()

SVM_precision = scores['test_precision_macro'].mean()

SVM_recall = scores['test_recall_macro'].mean()

SVM_f1 = scores['test_f1_weighted'].mean()

SVM_roc = scores['test_roc_auc'].mean()
LDA = LinearDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LDA_fit_time = scores['fit_time'].mean()

LDA_score_time = scores['score_time'].mean()

LDA_accuracy = scores['test_accuracy'].mean()

LDA_precision = scores['test_precision_macro'].mean()

LDA_recall = scores['test_recall_macro'].mean()

LDA_f1 = scores['test_f1_weighted'].mean()

LDA_roc = scores['test_roc_auc'].mean()
QDA = QuadraticDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

QDA_fit_time = scores['fit_time'].mean()

QDA_score_time = scores['score_time'].mean()

QDA_accuracy = scores['test_accuracy'].mean()

QDA_precision = scores['test_precision_macro'].mean()

QDA_recall = scores['test_recall_macro'].mean()

QDA_f1 = scores['test_f1_weighted'].mean()

QDA_roc = scores['test_roc_auc'].mean()
random_forest = RandomForestClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

forest_fit_time = scores['fit_time'].mean()

forest_score_time = scores['score_time'].mean()

forest_accuracy = scores['test_accuracy'].mean()

forest_precision = scores['test_precision_macro'].mean()

forest_recall = scores['test_recall_macro'].mean()

forest_f1 = scores['test_f1_weighted'].mean()

forest_roc = scores['test_roc_auc'].mean()
KNN = KNeighborsClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

KNN_fit_time = scores['fit_time'].mean()

KNN_score_time = scores['score_time'].mean()

KNN_accuracy = scores['test_accuracy'].mean()

KNN_precision = scores['test_precision_macro'].mean()

KNN_recall = scores['test_recall_macro'].mean()

KNN_f1 = scores['test_f1_weighted'].mean()

KNN_roc = scores['test_roc_auc'].mean()
bayes = GaussianNB()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

bayes_fit_time = scores['fit_time'].mean()

bayes_score_time = scores['score_time'].mean()

bayes_accuracy = scores['test_accuracy'].mean()

bayes_precision = scores['test_precision_macro'].mean()

bayes_recall = scores['test_recall_macro'].mean()

bayes_f1 = scores['test_f1_weighted'].mean()

bayes_roc = scores['test_roc_auc'].mean()
models_rfecv = pd.DataFrame({

    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],

    'Fitting time': [LR_fit_time, dtree_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],

    'Scoring time': [LR_score_time, dtree_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],

    'Accuracy'    : [LR_accuracy, dtree_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],

    'Precision'   : [LR_precision, dtree_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],

    'Recall'      : [LR_recall, dtree_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],

    'F1_score'    : [LR_f1, dtree_f1, SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],

    'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],

    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])



models_rfecv.sort_values(by='Accuracy', ascending=False)
lsvc = LinearSVC(C=0.05, penalty="l1", dual=False).fit(X, y)

etc = ExtraTreesClassifier()

etc.fit(X, y)



model = SelectFromModel(etc, prefit=True)

X_etc = model.transform(X)

X_etc.shape 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.34)
LR = LogisticRegression()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LR_fit_time = scores['fit_time'].mean()

LR_score_time = scores['score_time'].mean()

LR_accuracy = scores['test_accuracy'].mean()

LR_precision = scores['test_precision_macro'].mean()

LR_recall = scores['test_recall_macro'].mean()

LR_f1 = scores['test_f1_weighted'].mean()

LR_roc = scores['test_roc_auc'].mean()
decision_tree = DecisionTreeClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

dtree_fit_time = scores['fit_time'].mean()

dtree_score_time = scores['score_time'].mean()

dtree_accuracy = scores['test_accuracy'].mean()

dtree_precision = scores['test_precision_macro'].mean()

dtree_recall = scores['test_recall_macro'].mean()

dtree_f1 = scores['test_f1_weighted'].mean()

dtree_roc = scores['test_roc_auc'].mean()
SVM = SVC(probability = True)



scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

SVM_fit_time = scores['fit_time'].mean()

SVM_score_time = scores['score_time'].mean()

SVM_accuracy = scores['test_accuracy'].mean()

SVM_precision = scores['test_precision_macro'].mean()

SVM_recall = scores['test_recall_macro'].mean()

SVM_f1 = scores['test_f1_weighted'].mean()

SVM_roc = scores['test_roc_auc'].mean()
LDA = LinearDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

LDA_fit_time = scores['fit_time'].mean()

LDA_score_time = scores['score_time'].mean()

LDA_accuracy = scores['test_accuracy'].mean()

LDA_precision = scores['test_precision_macro'].mean()

LDA_recall = scores['test_recall_macro'].mean()

LDA_f1 = scores['test_f1_weighted'].mean()

LDA_roc = scores['test_roc_auc'].mean()
QDA = QuadraticDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

QDA_fit_time = scores['fit_time'].mean()

QDA_score_time = scores['score_time'].mean()

QDA_accuracy = scores['test_accuracy'].mean()

QDA_precision = scores['test_precision_macro'].mean()

QDA_recall = scores['test_recall_macro'].mean()

QDA_f1 = scores['test_f1_weighted'].mean()

QDA_roc = scores['test_roc_auc'].mean()
random_forest = RandomForestClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

forest_fit_time = scores['fit_time'].mean()

forest_score_time = scores['score_time'].mean()

forest_accuracy = scores['test_accuracy'].mean()

forest_precision = scores['test_precision_macro'].mean()

forest_recall = scores['test_recall_macro'].mean()

forest_f1 = scores['test_f1_weighted'].mean()

forest_roc = scores['test_roc_auc'].mean()
KNN = KNeighborsClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

KNN_fit_time = scores['fit_time'].mean()

KNN_score_time = scores['score_time'].mean()

KNN_accuracy = scores['test_accuracy'].mean()

KNN_precision = scores['test_precision_macro'].mean()

KNN_recall = scores['test_recall_macro'].mean()

KNN_f1 = scores['test_f1_weighted'].mean()

KNN_roc = scores['test_roc_auc'].mean()
bayes = GaussianNB()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=20)



sorted(scores.keys())

bayes_fit_time = scores['fit_time'].mean()

bayes_score_time = scores['score_time'].mean()

bayes_accuracy = scores['test_accuracy'].mean()

bayes_precision = scores['test_precision_macro'].mean()

bayes_recall = scores['test_recall_macro'].mean()

bayes_f1 = scores['test_f1_weighted'].mean()

bayes_roc = scores['test_roc_auc'].mean()
models_tree = pd.DataFrame({

    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],

    'Fitting time': [LR_fit_time, dtree_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],

    'Scoring time': [LR_score_time, dtree_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],

    'Accuracy'    : [LR_accuracy, dtree_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],

    'Precision'   : [LR_precision, dtree_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],

    'Recall'      : [LR_recall, dtree_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],

    'F1_score'    : [LR_f1, dtree_f1, SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],

    'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],

    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])



models_tree.sort_values(by='Accuracy', ascending=False)
model_general = pd.concat([models_initial['Model'], models_initial['Accuracy'], 

                           models_correlation['Model'],models_correlation['Accuracy'],

                          models_sfm['Model'], models_sfm['Accuracy'],

                          models_rfecv['Model'], models_rfecv['Accuracy'],

                          models_tree['Model'], models_tree['Accuracy'],

                          models_ensembling['Model'], models_ensembling['Accuracy']]

                          , axis=1)



model_general.columns = ['W/out reduction', 'Accuracy', 'Correlation', 'Accuracy_corr',

                        'Linear+SFM', 'Accuracy_sfm', 'Linear+RFECV', 'Accuracy_RFECV', 'Extra trees',

                         'Accuracy_trees', 'Voting', 'Accuracy_voting']



model_general.sort_values(by='Accuracy', ascending=False)