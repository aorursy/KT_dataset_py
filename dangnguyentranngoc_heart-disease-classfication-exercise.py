# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Magic dunction (I have no idea :'))

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Set matplotlib style

plt.style.use('fivethirtyeight')
heart_disease_data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

heart_disease_data
heart_disease_data.sample(5)
heart_disease_data.describe()
heart_disease_data.info()
heart_disease_data.isnull().sum()
heart_disease_data.isna().sum()
#Find out how many of each class we have

heart_disease_data.target.value_counts()
heart_disease_data.target.value_counts().plot(kind='bar', color=['salmon', 'lightblue']);
heart_disease_data.sex.value_counts()
pd.crosstab(heart_disease_data.sex, heart_disease_data.target)
# Plot a chart

pd.crosstab(heart_disease_data.sex, heart_disease_data.target).plot(kind='bar',

                                                                   figsize=(10, 6),

                                                                   color=['salmon', 'lightblue'])



plt.title('Heart disease frequency for sex')

plt.xlabel('0 = No Diesease, 1 = Disease')

plt.ylabel('Amount')

plt.legend(['Female', 'Male'])

plt.xticks(rotation=0);
# Create a new figure

plt.figure(figsize=(10, 6))



# Plot positive class

plt.scatter(heart_disease_data.age[heart_disease_data.target == 1],

           heart_disease_data.thalach[heart_disease_data.target == 1],

           c='salmon')



# Plot negative class

plt.scatter(heart_disease_data.age[heart_disease_data.target == 0],

           heart_disease_data.thalach[heart_disease_data.target == 0],

           c='lightblue')



plt.axhline(heart_disease_data.thalach.mean(), ls='--', c='darkblue')



plt.title('Heart Disease in function of Age and Max Heart Rate')

plt.xlabel('Age')

plt.ylabel('Max heart rate')

plt.legend(['Max heart rate mean', 'Disease', 'No disease']);
pd.crosstab(heart_disease_data.cp, heart_disease_data.target)
# Plot some chart...

pd.crosstab(heart_disease_data.cp, heart_disease_data.target).plot(kind='bar',

                                                                  figsize=(10, 6),

                                                                  color=['salmon', 'lightblue'])



plt.title('Heart Disease Frequency Per Chest Pain Type')

plt.xlabel('Chest Pain type')

plt.ylabel('Amount')

plt.legend(['No Disease', 'Disease'])

plt.xticks(rotation=0);
# Make a correlation matrix

heart_disease_data.corr()
# Plot a heatmap for correlation matrix

corr_matrix = heart_disease_data.corr()



fig, ax = plt.subplots(figsize=(15, 10))



ax = sns.heatmap(corr_matrix,

                annot=True,

                linewidths=0.5,

                fmt='.2f',

                cmap='YlGnBu');
# Models from Scikit-Learn

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Model Evaluations

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve
heart_disease_data.head()
# Split data

X = heart_disease_data.drop('target', axis=1)

y = heart_disease_data.target
X
y
# Split data into train and test sets

np.random.seed(42)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train
y_train, len(y_train)
type(y_test), type(X_train)
# Create a dictionary of models

models = {'Logistic Regression': LogisticRegression(),

          'KNN': KNeighborsClassifier(),

          'Random Forest': RandomForestClassifier()}



def fit_and_score(models, X_train, X_test, y_train, y_test):

    """Function fits and ecaluates given ML models.

    

    Args:

        models (Dictionary): different Scikit-Learn ML models.

        X_train (DataFrame): training data.

        X_test (DataFrame): testing data.

        y_train (Series): train labels

        y_test (Series): test labels

    

    return:

        A dictionary of model scores

    """

    np.random.seed(42)

    model_scores = {}

    

    for name, model in models.items():

        model.fit(X_train, y_train)

        model_scores[name] = model.score(X_test, y_test)

    

    return model_scores
model_scores = fit_and_score(models=models,

                             X_train=X_train,

                             X_test=X_test,

                             y_train=y_train,

                             y_test=y_test)



model_scores
model_compare = pd.DataFrame(model_scores, index=['acuracy'])

model_compare.T.plot.bar();
# Create a dict of parameter options for Logistic Regression

log_reg_grid = {'C': np.logspace(-4, 4, 20),

                'solver': ['newton-cg', 'lbfgs', 'liblinear'],

                'penalty': ['l1','l2', 'none']}



# Create a dict of parameter options for Random Forest classifier

rf_grid = {'n_estimators': np.arange(10, 1000, 50),

           'max_depth': [None, 3, 5, 10],

           'min_samples_split': np.arange(2,20, 2),

           'min_samples_leaf': np.arange(1, 20, 1),

           'max_features': ['auto', 'sqrt', 'log2']}
np.random.seed(42)



rs_rf = RandomizedSearchCV(RandomForestClassifier(),

                           param_distributions=rf_grid,

                           cv=5,

                           n_iter=20,

                           verbose=True,

                           n_jobs=-1)



rs_rf.fit(X_train, y_train)
rs_rf.best_params_
rs_rf.score(X_test, y_test)
np.random.seed(42)



rs_log_reg = RandomizedSearchCV(LogisticRegression(),

                                param_distributions=log_reg_grid,

                                cv=5,

                                n_iter=20,

                                n_jobs=-1,

                                verbose=True)



rs_log_reg.fit(X_train, y_train)
rs_log_reg.best_params_
rs_log_reg.score(X_test, y_test)
# Try LogisticRegression with GridSearchCV

gs_log_reg = GridSearchCV(LogisticRegression(),

                          param_grid=log_reg_grid,

                          cv=5,

                          n_jobs=-1,

                          verbose=True)



gs_log_reg.fit(X_train, y_train)
gs_log_reg.best_params_
gs_log_reg.score(X_test, y_test)
# Make prediction with tuned LogisticRegression model

y_preds = gs_log_reg.predict(X_test)

y_preds
y_test
# Plot ROC curve and calculate AUC metric

plot_roc_curve(gs_log_reg, X_test, y_test);
# Confusion matrix

print(confusion_matrix(y_test, y_preds))
# Plot confustion matrix

def plot_conf_mt(y_test, y_preds):

    """Function plot confusion matrix usiing Seaborn heatmap

    

    Ags:

        y_test (series): test labels

        y_preds (array): predicted labels

    """

    fig, ax = plt.subplots(figsize=(3, 3))

    ax = sns.heatmap(confusion_matrix(y_test, y_preds),

                     annot=True,

                     cbar=False)

    

    plt.xlabel('True label')

    plt.ylabel('Predicted label')

    

plot_conf_mt(y_test, y_preds)
print(classification_report(y_test, y_preds))
# Create a new LogisticRegression with best parameter

clf = LogisticRegression(C= 0.23357214690901212,

                         penalty='l2',

                         solver='liblinear')
# Cross-validate accuracy

cv_acc = cross_val_score(clf,

                         X,

                         y,

                         cv=5,

                         scoring='accuracy')



cv_acc
cv_acc = np.mean(cv_acc)

cv_acc
# Cross-validated precision

cv_precision = cross_val_score(clf,

                               X,

                               y,

                               cv=5,scoring='precision')



cv_precision = np.mean(cv_precision)

cv_precision
# Cross-validate recall

cv_recall = cross_val_score(clf,

                               X,

                               y,

                               cv=5,scoring='recall')



cv_recall = np.mean(cv_recall)

cv_recall
# Cross-validate f1-score

cv_f1 = cross_val_score(clf,

                        X,

                        y,

                        cv=5,scoring='f1')



cv_f1 = np.mean(cv_f1)

cv_f1
# Visualize cross-validated metric

cv_metric = pd.DataFrame({'Accuracy':cv_acc,

                          'Precision': cv_precision,

                          'Recall': cv_recall,

                          'F1':cv_f1},

                         index=[0])



cv_metric.T.plot.bar(title='Cross-validated metric', legend=False);
clf.fit(X_train, y_train)
clf.coef_
feature_dict = dict(zip(heart_disease_data.columns, list(clf.coef_[0])))

feature_dict
# Visualize feature importance

feature_df = pd.DataFrame(feature_dict, index=[0])

feature_df.T.plot.bar(title='Feature important', legend=False);
# Import XGBoost Classifier

from xgboost import XGBClassifier
xg_model = XGBClassifier(silent=False)

xg_model.fit(X_train, y_train)
xg_model.score(X_test, y_test)
np.random.seed(42)



# Create a hyperparameter grid

xg_grid = {'scale_pos_weight': [1],

           'silent': [False],

           'learning_rate': [0.001, 0.1, 0.3],

           'colsample_bytree': [0.4, 0.6, 0.8],

           'subsample': [0.6, 0.8],

           'objective': ['binary:logistic'],

           'n_estimators': np.arange(10, 1000, 50),

           'reg_alpha': [0.3],

           'max_depth': [4, 6, 10, 15, 20],

           'gamma': [0, 2, 4, 6, 8, 10]}



# Setup

rs_xg = RandomizedSearchCV(xg_model,

                         param_distributions=xg_grid,

                         cv=5,

                         n_iter=1000,

                         verbose=True)



rs_xg.fit(X_train, y_train)
rs_xg.best_params_
rs_xg.score(X_test, y_test)


tbh_xg = XGBClassifier(silent=False,

                       scale_pos_weight=1,

                       learning_rate=0.01,

                       colsample_bytree=0.8,

                       subsample = 0.8,

                       objective='binary:logistic',

                       n_estimators=5000,

                       reg_alpha = 0.005,

                       max_depth=4,

                       gamma=0,

                       min_child_weight=6)



tbh_xg.fit(X_train, y_train)
tbh_xg.score(X_test, y_test)
y_preds = tbh_xg.predict(X_test)
plot_roc_curve(tbh_xg, X_test, y_test);
plot_conf_mt(y_test, y_preds)
print(classification_report(y_test, y_preds))