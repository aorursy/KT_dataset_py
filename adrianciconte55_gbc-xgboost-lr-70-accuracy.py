import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime as dt
import calendar
%matplotlib inline
from time import sleep
from random import randint
import io
import re
import time
from random import randint

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import average_precision_score
from sklearn import model_selection
import xgboost as xgb
import re
import io

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.set(style="ticks", color_codes=True)
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 5000)
# pd.set_option('display.width', 1000)
fontsize = 25

cm = sns.light_palette("orange", as_cmap=True)
g_color = ['#ea4335ff','#4285f4ff', '#34a853ff', '#fbbc05ff', 'k', 'grey']
#read in CSV
df = pd.read_csv("../input/StudentsPerformance.csv")
#create total score
df['total score'] = df['math score'] + df['reading score'] + df['writing score']

#create percent across all three sections
df['percent'] = round(df['total score'].div(300)*100, 2)

#create percent bins for easier visualisations
df['percent bin'] = pd.cut(df['percent'], range(0,105,5))

#rename columns
df.rename(columns={'race/ethnicity':'race', 'parental level of education':'level of education'}, inplace=True)
#creating a higher 'pass' mark to ensure there is no class imbalance. if pass mark is 50, then class imbalance is 89/11

def pass_fail(df):
  if df['percent'] > 70:
    return 'Pass'
  else:
    return 'Fail'

df['status'] = df.apply(pass_fail, axis=1)
df['Target_Pass'] = df['status'].map({'Pass':1, 'Fail':0})
df.groupby(['Target_Pass']).size()
plt.figure(figsize=(30,12))

ax1 = plt.subplot(211)

sns.heatmap(pd.crosstab(df['race'], df['percent bin']), annot=True, cmap='Greens', linewidths=1, fmt='.0f', 
            annot_kws={"size": 20}, cbar_kws={'label':'', 'orientation':'vertical'}, ax=ax1)

plt.xticks(fontsize=fontsize, rotation=90)
plt.yticks (fontsize=fontsize, rotation=0)
plt.xlabel('Bin', fontsize=fontsize)
plt.ylabel('Race', fontsize=fontsize)

ax2 = plt.subplot(212)

sns.heatmap(pd.crosstab(df['level of education'], df['percent bin']), annot=True, cmap='Greens', linewidths=1, fmt='.0f', 
            annot_kws={"size": 20}, cbar_kws={'label':'', 'orientation':'vertical'}, ax=ax2)

plt.xticks(fontsize=fontsize, rotation=90)
plt.yticks (fontsize=fontsize)
plt.xlabel('Bin', fontsize=fontsize)
plt.ylabel('Education', fontsize=fontsize)

plt.tight_layout()
plt.show()
plt.figure(figsize=(30,12))

ax1 = plt.subplot(211)

sns.heatmap(pd.crosstab(df['gender'], df['percent bin']), annot=True, cmap='Greens', linewidths=1, fmt='.0f', 
            annot_kws={"size": 20}, cbar_kws={'label':'', 'orientation':'vertical'}, ax=ax1)

plt.xticks(fontsize=fontsize, rotation=90)
plt.yticks (fontsize=fontsize, rotation=0)
plt.xlabel('Bin', fontsize=fontsize)
plt.ylabel('Gender', fontsize=fontsize)

ax2 = plt.subplot(212)

sns.heatmap(pd.crosstab(df['test preparation course'], df['percent bin']), annot=True, cmap='Greens', linewidths=1, fmt='.0f', 
            annot_kws={"size": 20}, cbar_kws={'label':'', 'orientation':'vertical'}, ax=ax2)

plt.xticks(fontsize=fontsize, rotation=90)
plt.yticks (fontsize=fontsize, rotation=0)
plt.xlabel('Bin', fontsize=fontsize)
plt.ylabel('Preparation', fontsize=fontsize, rotation=90)

plt.tight_layout()
plt.show()
ax1 = plt.subplot(221)

round(df.groupby(['gender', 'race'])['math score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax1)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Math Score', fontsize=fontsize)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax2 = plt.subplot(222)

round(df.groupby(['gender', 'race'])['reading score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax2)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Reading Score', fontsize=fontsize)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax3 = plt.subplot(223)

round(df.groupby(['gender', 'race'])['writing score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax3)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Writing Score', fontsize=fontsize)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax4 = plt.subplot(224)

round(df.groupby(['gender', 'race'])['total score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax4)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Total Score', fontsize=fontsize)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

plt.tight_layout()
plt.show()
ax1 = plt.subplot(221)

round(df.groupby(['gender', 'level of education'])['math score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax1)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Math Score', fontsize=fontsize)
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax2 = plt.subplot(222)

round(df.groupby(['gender', 'level of education'])['reading score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax2)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Reading Score', fontsize=fontsize)
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax3 = plt.subplot(223)

round(df.groupby(['gender', 'level of education'])['writing score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax3)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Writing Score', fontsize=fontsize)
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax4 = plt.subplot(224)

round(df.groupby(['gender', 'level of education'])['total score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax4)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Total Score', fontsize=fontsize)
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

plt.tight_layout()
plt.show()
ax1 = plt.subplot(221)

round(df.groupby(['gender', 'test preparation course'])['math score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax1)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Math Score', fontsize=fontsize)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax2 = plt.subplot(222)

round(df.groupby(['gender', 'test preparation course'])['reading score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax2)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Reading Score', fontsize=fontsize)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax3 = plt.subplot(223)

round(df.groupby(['gender', 'test preparation course'])['writing score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax3)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Writing Score', fontsize=fontsize)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

ax4 = plt.subplot(224)

round(df.groupby(['gender', 'test preparation course'])['total score'].agg('mean').unstack(0).fillna(0), 2).plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax4)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('')
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Gender Total Score', fontsize=fontsize)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.40), ncol=6, fontsize=fontsize)

plt.tight_layout()
plt.show()
def school_uni(df):
  if df['level of education'] == "bachelor's degree":
    return 'University'
  elif df['level of education'] == "some college":
    return 'High School'
  elif df['level of education'] == "master's degree":
    return 'University'
  elif df['level of education'] == "associate's degree":
    return 'University'
  elif df['level of education'] == "high school":
    return 'High School'
  elif df['level of education'] == "some high school":
    return 'High School'
  else:
    return 'Error'
  
df['Education'] = df.apply(school_uni, axis=1)
ax1 = plt.subplot(211)

colors = {'female':'green', 'male':'blue'}
df[df['gender'] == 'female']['total score'].plot(kind='hist', alpha=0.3, color='blue', bins=20, label='Female', figsize=(29,14), ax=ax1)
df[df['gender'] == 'male']['total score'].plot(kind='hist', alpha=0.3, color='green', bins=20, label='Male', figsize=(29,14), ax=ax1)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('')
plt.legend()

ax2 = plt.subplot(212)

df.groupby(['percent bin', 'gender']).size().unstack().plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=['#ea4335ff','#4285f4ff', '#34a853ff'], ax=ax2)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')

plt.tight_layout()
plt.show()
ax1 = plt.subplot(211)

df.groupby(['percent bin', 'race']).size().unstack().plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=g_color, ax=ax1)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')

ax2 = plt.subplot(212)

df.groupby(['percent bin', 'Education']).size().unstack().plot(kind='bar', width=0.85, figsize=(29,14), grid=True, color=g_color, ax=ax2)
plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(linestyle='--', linewidth=1, alpha=0.3, color='lightgrey')

plt.tight_layout()
plt.show()
df.head()
X = pd.get_dummies(df[['gender', 'race', 'level of education', 'lunch', 'test preparation course']], drop_first=False)
y = df['Target_Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
models = []
models.append(('LR1', LogisticRegression()))
models.append(('LR2', LogisticRegression(C=100, penalty='l2', solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=20)))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('MLP', MLPClassifier()))
models.append(('ETC', ExtraTreeClassifier()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('BCL', BaggingClassifier()))
models.append(('XGB', xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

parameters = ["accuracy", "average_precision", "f1", "f1_micro", 'f1_macro', 'f1_weighted', 'precision', "roc_auc"]

for name, model in models:
    kfold = model_selection.KFold(n_splits = 15, random_state = 7)
    cv_results = model_selection.cross_val_score(model, X, y, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison

boxprops = dict(linestyle='-', linewidth=3, color='k')
medianprops = dict(linestyle='-', linewidth=3, color='k')
red_square = dict(markerfacecolor='r', marker='s')
box_color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

fig = plt.figure(figsize=(27,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results, showfliers=True, flierprops=red_square, vert=True)
ax.set_xticklabels(names)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.show()
clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=4,
              max_features=0.1, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=20, min_samples_split=400,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              n_iter_no_change=None, presort='auto', random_state=None,
              subsample=1.0, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
kfold = model_selection.KFold(n_splits = 10, random_state = 7)
scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

print('GBC Classifier is prediciting at: {}'.format(metrics.accuracy_score(y_test, predictions)))
print('Cross Validation Scores are: {}'.format(scores))
print('Cross Validation Score Averages are: {}'.format(scores.mean()))
#create list of feature importance for GBC Classifier

df_f = pd.DataFrame(clf.feature_importances_, columns=["Importance"])
df_f['Labels'] = X_train.columns
df_f.sort_values("Importance", inplace=True, ascending=True)
df_f.set_index('Labels').sort_values(by='Importance', ascending=False)[:15].plot(kind='bar', figsize=(15,14), width=0.85, color='#34a853ff', alpha=1)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.tight_layout()
plt.show()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#create list of feature importance using SelectKBest and chi2

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores.set_index('Specs').sort_values(by='Score', ascending=False).plot(kind='bar', width=0.85, color='#34a853ff', figsize=(15,14))

plt.rcParams["axes.linewidth"]  = 2
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
# gb_grid_params = {'learning_rate': [0.1, 0.01, 0.02, 0.05, 0.001],
#               'max_depth': [4, 6, 8],
#               'min_samples_leaf': [20, 50, 100, 150],
#               'max_features': [1.0, 0.3, 0.1],
#               'n_estimators':range(20, 101, 10),
#               'min_samples_split':range(200, 1001, 200) 
#               }

# grid = GridSearchCV(clf, param_grid=gb_grid_params, cv=10, scoring="accuracy", n_jobs=3)

# grid.fit(X,y)

# print(np.mean(grid.cv_results_['mean_test_score']))
# print(grid.best_score_)
# print(grid.best_estimator_)
# print(grid.best_params_)
# print(grid.n_splits_)
clf = xgb.XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=500, silent=True, objective='binary:logistic', 
                        booster='dart', n_jobs=5, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, 
                        colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, importance_type='total_gain')

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
kfold = model_selection.KFold(n_splits = 10, random_state = 42)
scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

print('XGBoost Classifier is prediciting at: {}'.format(metrics.accuracy_score(y_test, predictions)))
print('Cross Validation Scores are: {}'.format(scores))
print('Cross Validation Score Averages are: {}'.format(scores.mean()))
# gb_grid_params = {"learning_rate" : [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] ,
#                   "max_depth"        : [3, 4, 5, 6, 8, 10, 12, 15],
#                   "min_child_weight" : [1, 3, 5, 7 ],
#                   "gamma"            : [0.0, 0.1, 0.2 , 0.3, 0.4 ],
#                   "colsample_bytree" : [0.3, 0.4, 0.5 , 0.7 ] }

# grid = GridSearchCV(clf, param_grid=gb_grid_params, cv=10, scoring="accuracy", n_jobs=3)

# grid.fit(X,y)

# print(np.mean(grid.cv_results_['mean_test_score']))
# print(grid.best_score_)
# print(grid.best_estimator_)
# print(grid.best_params_)
# print(grid.n_splits_)
clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='liblinear',
          tol=0.0001, verbose=0, warm_start=False)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
kfold = model_selection.KFold(n_splits = 10, random_state = 42)
scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

print('Logistic Regression Classifier is prediciting at: {}'.format(metrics.accuracy_score(y_test, predictions)))
print('Cross Validation Scores are: {}'.format(scores))
print('Cross Validation Score Averages are: {}'.format(scores.mean()))
# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
grid = GridSearchCV(clf, hyperparameters, cv=10, scoring="accuracy", n_jobs=3)

grid.fit(X,y)

print(np.mean(grid.cv_results_['mean_test_score']))
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)
print(grid.n_splits_)
