# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Read data from CSV

df_raw = pd.read_csv('../input/weatherAUS.csv')
df_raw.head(5)
df_raw.count().sort_values()
df = df_raw.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am', 'RISK_MM', 'Date', 'Location'],axis=1)

df.shape
df.dropna(how='any', inplace=True)
df.shape
fig, ax = plt.subplots(figsize=[15,15])

sns.heatmap(df.corr(), ax=ax, cmap='Blues', annot=True);

ax.set_title("Pearson correlation coefficients", size=20);


params=['MinTemp','MaxTemp','Rainfall','WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

pd.plotting.scatter_matrix(df[params], alpha=0.2, figsize=(20, 20))

plt.show()
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No':0})

df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No':0})

df = pd.get_dummies(data=df, columns=['WindGustDir', 'WindDir3pm', 'WindDir9am']) 
X = df.loc[:, df.columns != 'RainTomorrow']

Y = df.loc[:,'RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=56)
X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)
classifier_models = {

    'CART': DecisionTreeClassifier(),

    'XGB': XGBClassifier(n_jobs=-1),

    'GNB': GaussianNB(),

    'LDA': LinearDiscriminantAnalysis(),

    'LR': LogisticRegression(),

    'KNN': KNeighborsClassifier()

}
def accuracy_report(models, X, y):

    results = []

    for name in models.keys():

        model = models[name]

        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        print("Accuracy: %.3f (+/- %.3f) [%s]-[%s]" %(scores.mean(), scores.std(), models[name], name))
accuracy_report(classifier_models, X_train, y_train)
from sklearn.metrics import confusion_matrix #for model evaluation

def confusion_martix_report(models, X_train, y_train, X_test, y_test):

    results = []

    for name in models.keys():

        model = models[name]

        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        cm = confusion_matrix(y_test, y_predict)

        plt.figure(figsize = (4,4))

        sns.heatmap(cm,fmt="d",annot=True)

        plt.title("Confusion Matrix")

        plt.xlabel("Predicted")

        plt.ylabel("Actuals")

        plt.show()
confusion_martix_report(classifier_models, X_train, y_train, X_test, y_test)
rf_classifier = RandomForestClassifier()

param_grid = {

    'n_estimators': [200,300,400,500,600],

    'random_state': [30,40,50,60,70]

}

gscv_rfc = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv= 5)

gscv_rfc.fit(X_train,y_train)
gscv_rfc.best_params_
rfc_modified = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=None,

            oob_score=False, random_state=60, verbose=0,

            warm_start=False)

scores = cross_val_score(rfc_modified, X_train, y_train, scoring='accuracy')

print("Accuracy: %.3f (+/- %.3f)" %(scores.mean(), scores.std()))
lrclf = LogisticRegression()

# Create regularization penalty space

penalty = ['l1', 'l2']

# Create regularization hyperparameter space

C = np.logspace(0, 4, 10)

# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation

gscv_lr = GridSearchCV(lrclf, hyperparameters, cv=5, verbose=0)

gscv_lr.fit(X_train,y_train)
# View best hyperparameters

print("tuned hpyerparameters :(best parameters) ",gscv_lr.best_params_)

print("accuracy :",gscv_lr.best_score_)
import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance

#import shap #for SHAP values
rfc_modified.fit(X_train, y_train)

perm = PermutationImportance(rfc_modified, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())