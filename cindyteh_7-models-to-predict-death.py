import pandas as pd
import numpy as np

import plotly.express as px 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Read csv file into dataframe
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

# Show dataframe
df.head()
# Understand data
df.info()
# Check for missing data
df.isnull().sum()
sns.set()

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df.corr(), linewidths=.5, ax=ax, cmap='Blues')
sns.set_style('white') 
fig, ax = plt.subplots(3,2,figsize=(13,15))
sns.countplot(df['anaemia'], palette='Pastel1', ax=ax[0][0])
sns.countplot(df['diabetes'], palette='Set3', ax=ax[0][1])
sns.countplot(df['high_blood_pressure'], palette='Set2', ax=ax[1][0])
sns.countplot(df['sex'], palette='Set1', ax=ax[1][1])
sns.countplot(df['smoking'], palette='Pastel2', ax=ax[2][0])
sns.countplot(df['DEATH_EVENT'], palette='Accent', ax=ax[2][1])
fig, ax = plt.subplots(6,1,figsize=(13,20))
plt.suptitle('Bivariate Analysis (Hue=Sex)', fontsize=20)
plt.tight_layout(4)

sns.lineplot(data=df, x='age', y='creatinine_phosphokinase', hue='sex', lw=2, ax=ax[0])
sns.lineplot(data=df, x='age', y='ejection_fraction', hue='sex', lw=2, ax=ax[1])
sns.lineplot(data=df, x='age', y='platelets', hue='sex', lw=2, ax=ax[2])
sns.lineplot(data=df, x='age', y='serum_creatinine', hue='sex', lw=2, ax=ax[3])
sns.lineplot(data=df, x='age', y='serum_sodium', hue='sex', lw=2, ax=ax[4])
sns.lineplot(data=df, x='age', y='time', hue='sex', lw=2, ax=ax[5])
fig, ax = plt.subplots(6,1,figsize=(13,20))
plt.suptitle('Bivariate Analysis (Hue=Death)', fontsize=20)
plt.tight_layout(4)

sns.lineplot(data=df, x='age', y='creatinine_phosphokinase', hue='DEATH_EVENT', lw=2, ax=ax[0])
sns.lineplot(data=df, x='age', y='ejection_fraction', hue='DEATH_EVENT', lw=2, ax=ax[1])
sns.lineplot(data=df, x='age', y='platelets', hue='DEATH_EVENT', lw=2, ax=ax[2])
sns.lineplot(data=df, x='age', y='serum_creatinine', hue='DEATH_EVENT', lw=2, ax=ax[3])
sns.lineplot(data=df, x='age', y='serum_sodium', hue='DEATH_EVENT', lw=2, ax=ax[4])
sns.lineplot(data=df, x='age', y='time', hue='DEATH_EVENT', lw=2, ax=ax[5])
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
Y = df['DEATH_EVENT']
X = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]
# SMOTE: Synthetic Minority Over-sampling Technique
X_smote,Y_smote = SMOTE().fit_sample(X,Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, stratify = Y_smote, test_size=0.2, random_state=52)
print('Shape of X_train:', X_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of Y_train:', Y_train.shape)
print('Shape of Y_test:', Y_test.shape)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt

logis = LogisticRegression(random_state=0, solver='lbfgs')
model = logis.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

skplt.metrics.plot_confusion_matrix(Y_test, Y_predict, figsize=(8,8), 
                                    title='Confusion Matrix: Logistic Regression',
                                    normalize=True,
                                    cmap='Blues')

print(classification_report(Y_test, Y_predict))
from sklearn import svm

svm = svm.SVC(kernel='linear', C = 1)
model = svm.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

skplt.metrics.plot_confusion_matrix(Y_test, Y_predict, figsize=(8,8), 
                                    title='Confusion Matrix: SVM',
                                    normalize=True,
                                    cmap='Blues')

print(classification_report(Y_test, Y_predict))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
model = dt.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

skplt.metrics.plot_confusion_matrix(Y_test, Y_predict, figsize=(8,8), 
                                    title='Confusion Matrix: Decision Tree',
                                    normalize=True,
                                    cmap='Blues')

print(classification_report(Y_test, Y_predict))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50)
model = rf.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

skplt.metrics.plot_confusion_matrix(Y_test, Y_predict, figsize=(8,8), 
                                    title='Confusion Matrix: Random Forest',
                                    normalize=True,
                                    cmap='Blues')

print(classification_report(Y_test, Y_predict))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
model = gb.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

skplt.metrics.plot_confusion_matrix(Y_test, Y_predict, figsize=(8,8), 
                                    title='Confusion Matrix: Gradient Boosting Classifier',
                                    normalize=True,
                                    cmap='Blues')

print(classification_report(Y_test, Y_predict))
import lightgbm as lgb
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
model = lgbm.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

skplt.metrics.plot_confusion_matrix(Y_test, Y_predict, figsize=(8,8), 
                                    title='Confusion Matrix: LGBM',
                                    normalize=True,
                                    cmap='Blues')

print(classification_report(Y_test, Y_predict))
# Show importance features
plt.figure()
lgb.plot_importance(model)
plt.title("Feature Importances")
plt.show()
import xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier()
model = xgb.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

skplt.metrics.plot_confusion_matrix(Y_test, Y_predict, figsize=(8,8), 
                                    title='Confusion Matrix: XgBoost',
                                    normalize=True,
                                    cmap='Blues')

print(classification_report(Y_test, Y_predict))
plt.figure()
xgboost.plot_importance(model)
plt.title("Feature Importances")
plt.show()