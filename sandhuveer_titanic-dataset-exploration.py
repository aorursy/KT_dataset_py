import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as skpre 
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

df_titanic = pd.read_csv("../input/titanic/train.csv")
df_titanic.isnull().sum()
df_titanic.mean()
df_titanic.describe()
df_titanic.mode()
df_titanic = df_titanic.drop(columns = ['Cabin'])
df_titanic
df_titanic.isnull().sum()
df_titanic = df_titanic.dropna(axis = 0, how = 'any').copy()
df_titanic
df_titanic.isnull().sum()
df_titanic['is_male'] = df_titanic['Sex'] == 'male'
df_titanic['is_female'] = df_titanic['Sex'] == 'female'
df_titanic
df_titanic['Sex-Class'] = df_titanic['Sex'] + df_titanic['Pclass'].astype(str) 
df_titanic
df_titanic.mean()
people_in_firstclass = (df_titanic['Pclass'] == 1).sum()
people_in_firstclass
people_in_secondclass = (df_titanic['Pclass'] == 2).sum()
people_in_secondclass
people_in_thirdclass = (df_titanic['Pclass'] == 3).sum()
people_in_thirdclass
females_survived = ((df_titanic['is_female']) & (df_titanic['Survived'])).sum()
females_total = (df_titanic['is_female']).sum()
females_survive_percent = (females_survived / females_total) * 100
females_survive_percent
males_survived = ((df_titanic['is_male']) & (df_titanic['Survived'])).sum()
males_total = (df_titanic['is_male']).sum()
males_survive_percent = (males_survived / males_total) * 100
males_survive_percent
female1_survive = ((df_titanic['Sex-Class'] == 'female1') & (df_titanic['Survived'])).sum()
female1_total = (df_titanic['Sex-Class'] == 'female1').sum()
female1_survive_percent = (female1_survive / female1_total) * 100
female1_survive_percent # 96% of the females in first class survived
female2_survive = ((df_titanic['Sex-Class'] == 'female2') & (df_titanic['Survived'])).sum()
female2_total = (df_titanic['Sex-Class'] == 'female2').sum()
female2_survive_percent = (female2_survive / female2_total) * 100
female2_survive_percent
female3_survive = ((df_titanic['Sex-Class'] == 'female3') & (df_titanic['Survived'])).sum()
female3_total = (df_titanic['Sex-Class'] == 'female3').sum()
female3_survive_percent = (female3_survive / female3_total) * 100
female3_survive_percent
male1_survive = ((df_titanic['Sex-Class'] == 'male1') & (df_titanic['Survived'])).sum()
male1_total = (df_titanic['Sex-Class'] == 'male1').sum()
male1_survive_percent = (male1_survive / male1_total) * 100
male1_survive_percent
male2_survive = ((df_titanic['Sex-Class'] == 'male2') & (df_titanic['Survived'])).sum()
male2_total = (df_titanic['Sex-Class'] == 'male2').sum()
male2_survive_percent = (male2_survive / male2_total) * 100
male2_survive_percent
male3_survive = ((df_titanic['Sex-Class'] == 'male3') & (df_titanic['Survived'])).sum()
male3_total = (df_titanic['Sex-Class'] == 'male3').sum()
male3_survive_percent = (male3_survive / male3_total) * 100
male3_survive_percent
sns.catplot(data = df_titanic, x = 'Pclass', y = 'Age', hue = 'Survived')
fig = plt.gcf()
fig.suptitle('Titanic Dataset Plot (Class, Age, Survived)', y = 1.02)
fig.show()
sns.catplot(data = df_titanic, x = 'Sex', y = 'Age', hue = 'Survived')
fig = plt.gcf()
fig.suptitle('Titanic Dataset Plot (Sex, Age, Survived)', y = 1.02)
fig.show()
sns.catplot(data = df_titanic, x = 'Sex-Class', y = 'Age', hue = 'Survived')
fig = plt.gcf()
fig.suptitle('Titanic Dataset Plot (Sex-Class, Age, Survived)', y = 1.02)
fig.show()
df_titanic2 = df_titanic.copy()
df_titanic2 = df_titanic.drop(columns  = ['Name', 'Ticket', 'Sex-Class', 'is_male', 'is_female'])
df_titanic2
df_titanic2['Sex'] = df_titanic2['Sex'].map({'male':1,'female':0})
df_titanic2['Embarked'] = df_titanic2['Embarked'].map({'S':0,'C':1, 'Q':2})
df_titanic2
true_survival = df_titanic2['Survived']
df_titanic2 = df_titanic2.drop(columns = ['Survived'])
df_titanic2
X_train, X_test, y_train, y_test = train_test_split(df_titanic2, true_survival, test_size = 0.20)
rf_classifier = RandomForestClassifier(n_estimators = 50, max_depth=1, random_state=5)
rf_precision_scores = cross_val_score(rf_classifier, df_titanic2, true_survival, scoring = 'precision', cv=5)
rf_precision_scores.mean()
rf_recall_scores = cross_val_score(rf_classifier, df_titanic2, true_survival, scoring = 'recall', cv=5)
rf_recall_scores.mean()
rf_classifier2 = RandomForestClassifier()
param_space = { 'n_estimators' : [3,10,30,50,100], 
               'max_depth' : [5,10,50,100], 
               'min_samples_split' : [2,10,50], 
               'min_samples_leaf' : [1, 10, 100], 
               'max_features' : [2,4,6,8] }

rf_classifier2_cv = RandomizedSearchCV(rf_classifier2, param_space, n_iter=10, scoring = 'neg_root_mean_squared_error', cv = 5 )
search = rf_classifier2_cv.fit(X_train, y_train)
search.best_params_
rf_classifier2 = RandomForestClassifier(max_depth=100, max_features=2, min_samples_leaf=1, min_samples_split=10, n_estimators=30)
rf_classifier2.fit(X_train, y_train)
rf2_predictions = rf_classifier2.predict(X_test)
mean_squared_error(y_test, rf2_predictions)
rf2_precision_scores = cross_val_score(rf_classifier2, df_titanic2, true_survival, scoring = 'precision', cv=5)
rf2_precision_scores.mean()
rf2_recall_scores = cross_val_score(rf_classifier2, df_titanic2, true_survival, scoring = 'recall', cv=5)
rf2_recall_scores.mean()
accuracy_score(y_test, rf2_predictions)
f1score_rf2  = 2 * (( rf2_precision_scores.mean() * rf2_recall_scores.mean()) / (rf2_precision_scores.mean() + rf2_recall_scores.mean()))
# 2*((precision*recall)/(precision+recall))
f1score_rf2
logistic_regression_model = linear_model.LogisticRegression(max_iter=800)
logistic_regression_model.fit(X = X_train, y = y_train)
predictions = logistic_regression_model.predict(X = X_test)
mean_absolute_error(y_test, predictions)
logistic_regression_model.coef_
lr_precision_scores = cross_val_score(logistic_regression_model, df_titanic2, true_survival, scoring = 'precision', cv=5)
lr_precision_scores.mean()
lr_recall_scores = cross_val_score(logistic_regression_model, df_titanic2, true_survival, scoring = 'recall', cv=5)
lr_recall_scores.mean()
accuracy_score(y_test, predictions)
f1score_lr  = 2 * (( lr_precision_scores.mean() * lr_recall_scores.mean()) / (lr_precision_scores.mean() + lr_recall_scores.mean()))
f1score_lr
cf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Greens')
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
predictions_tree = tree_model.predict(X_test)
predictions_tree
mean_absolute_error(y_test, predictions_tree)
tm_precision_scores = cross_val_score(tree_model, df_titanic2, true_survival, scoring = 'precision', cv=5)
tm_precision_scores.mean()
tm_recall_scores = cross_val_score(tree_model, df_titanic2, true_survival, scoring = 'recall', cv=5)
tm_recall_scores.mean()
accuracy_score(y_test, predictions_tree)
f1score_tm  = 2 * (( tm_precision_scores.mean() * tm_recall_scores.mean()) / (tm_precision_scores.mean() + tm_recall_scores.mean()))
f1score_tm
cf_matrix2 = confusion_matrix(y_test, predictions_tree)
sns.heatmap(cf_matrix2/np.sum(cf_matrix2), annot=True, fmt='.2%', cmap='Reds')