import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression

from scikitplot.metrics import plot_roc_curve as prc

from scikitplot.metrics import plot_lift_curve as plc

from scikitplot.metrics import plot_cumulative_gain as pcg

from sklearn.metrics import classification_report
data = pd.read_csv('../input/heart.csv')
data.info()

data.head(10)
data.columns = ['age', 'sex', 'chest_pain', 'rest_bp', 'cholesterol', 'fast_bs', 'rest_ecg', 'max_heart_rate',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'n_major_vessels', 'thalassemia', 'diagnose']
data = data[data.thalassemia != 0]
sex = { 0:'Female', 1:'Male'}

cp = {0:'Typical Angina', 1:'Atypical Angina', 2:'Non-anginal Pain', 3:'Asymptomatic'}

fbs = { 0:'<120mg/dl', 1:'>120mg/dl'}

ecg = {0 : 'Normal', 1:'ST-T', 2:'Left Ventricular Hypertrophy'}

ex = { 0:'no', 1:'yes'}

st_slope = {0:'Upsloping', 1:'Flat', 2:'Downsloping'}

thal = {1:'Normal', 2:'Fixed Defect', 3:'Reversable Defect'}
data['sex'] = [sex[i] for i in data.sex]

data['chest_pain'] = [cp[i] for i in data.chest_pain]

data['fast_bs'] = [fbs[i] for i in data.fast_bs]

data['rest_ecg'] = [ecg[i] for i in data.rest_ecg]

data['exercise_induced_angina'] = [ex[i] for i in data.exercise_induced_angina]

data['st_slope'] = [st_slope[i] for i in data.st_slope]

data['thalassemia'] = [thal[i] for i in data.thalassemia]
data['sex'] = data['sex'].astype('category')

data['chest_pain'] = data['chest_pain'].astype('category')

data['fast_bs'] = data['fast_bs'].astype('category')

data['rest_ecg'] = data['rest_ecg'].astype('category')

data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('category')

data['st_slope'] = data['st_slope'].astype('category')

data['thalassemia'] = data['thalassemia'].astype('category')
suspected = data[data['diagnose'] == 1]

sns.distplot(suspected.age, bins=15, rug=True)

print("80% Percentile of heart disease suspected patients' age : ", np.percentile(suspected.age, [10,90]))
suspected = data[data['diagnose'] == 1]

sns.distplot(suspected.rest_bp, bins=15, rug=True)

print("80% Percentile of heart disease suspected patients' rest blood pressure : ", np.percentile(suspected.rest_bp, [10,90]))
sns.countplot(suspected.chest_pain)
suspected = data[data['diagnose'] == 1]

sns.distplot(suspected.cholesterol, bins=15, rug=True)

print("80% Percentile of heart disease suspected patients' cholesterol : ", np.percentile(suspected.cholesterol, [10,90]))
sns.countplot(suspected.fast_bs)
suspected = data[data['diagnose'] == 1]

sns.distplot(suspected.max_heart_rate, bins=15, rug=True)

print("80% Percentile of heart disease suspected patients' maximum heart rate : ", np.percentile(suspected.max_heart_rate, [10,90]))
sns.countplot(suspected.thalassemia)
data = pd.get_dummies(data, drop_first=True)

data.head(10)
y = data[['diagnose']]

X = data.drop('diagnose',axis=1)
lr = LogisticRegression()

def cv_score(variables, X, y):

    cv_scores = cross_val_score(lr, X[variables], y, cv=10, scoring='roc_auc')

    cv_scores = cv_scores.mean()

    return(cv_scores)

    

def sel_var(candidate_variables, X, y):

    best_auc = 0

    best_var = []

    for v in candidate_variables:

        auc_var = cv_score(best_var + [v], X, y)

        if auc_var > best_auc:

            best_auc = auc_var

            best_var = best_var + [v]

    return best_auc, best_var

        
variables = X.columns.values

auc, var = sel_var(variables, X, y)
print(auc)

print(var)
X_train, X_test, y_train, y_test = train_test_split(X[var], y, test_size=0.2)

lr.fit(X_train, y_train)

y_pred = lr.predict_proba(X_test)

y_pred_bin = lr.predict(X_test)

prc(y_test, y_pred)

pcg(y_test, y_pred)

plc(y_test, y_pred)

print(classification_report(y_test, y_pred_bin))