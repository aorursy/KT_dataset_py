##importing required libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

import numpy as np

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
##reading the Wine Quality data

df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
print("data dimensions: ", df.shape)

df.head()
df.info()
df.describe()
df.isnull().sum()##checking missing values
sns.pairplot(df)

plt.show()
sns.countplot(df['quality'])
sns.barplot(x = 'quality', y = 'alcohol', data = df)
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)
sns.barplot(x = 'quality', y = 'volatile acidity', data = df)
sns.barplot(x = 'quality', y = 'citric acid', data = df)
sns.barplot(x = 'quality', y = 'sulphates', data = df)
sns.barplot(x = 'quality', y = 'pH', data = df)
##Correlation Matrix

corr = df.corr()

plt.subplots(figsize=(15,10))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
# Create Classification version of target variable

df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]

##now the target variable is goodquality

# Separate feature variables and target variable

X = df.drop(['quality','goodquality'], axis = 1)

y = df['goodquality']

print("Independent Variables: ",X.shape)

print("Dependent Variables: ",y.shape)
# See proportion of good vs bad wines

print(df['goodquality'].value_counts())

sns.countplot(df['goodquality'],label="Count")
sns.lmplot(x='density',y='fixed acidity',data=df,fit_reg=False,hue='goodquality') 

sns.lmplot(x='pH',y='fixed acidity',data=df,fit_reg=False,hue='goodquality') 

plt.show()
### Standardize the data

from sklearn.preprocessing import StandardScaler

X_features = X

X = StandardScaler().fit_transform(X)
# Splitting the data into Train and test - 80-20 ratio

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

print("Training Data: ",X_train.shape, y_train.shape)

print("Test Data: ",X_test.shape, y_test.shape)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

## checking the metrics of model on test data

acc_log = round(logreg.score(X_test, y_test)*100, 2)

print("Accuracy on test data: ",acc_log)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, make_scorer

pred_log = logreg.predict(X_test)

auc_log = round(roc_auc_score(y_test, pred_log)*100,2)

print("AUC: ",  auc_log)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, y_train)

## checking the metrics of model on test data

acc_svc = round(svc.score(X_test, y_test)*100, 2)

print("Accuracy on test set: ",acc_svc)
pred_svc = svc.predict(X_test)

auc_svc = round(roc_auc_score(y_test, pred_svc)*100,2)

print("AUC: ",  auc_svc)
knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(X_train, y_train)

## checking the metrics of model on test data

acc_knn = round(knn.score(X_test, y_test)*100, 2)

print("Accuracy on test data: ",acc_knn)
### Predicting on test data and checking the metrics

pred_knn = knn.predict(X_test)

auc_knn = round(roc_auc_score(y_test, pred_knn)*100,2)

print("AUC: ",  auc_knn)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

acc_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)

acc_gaussian
pred_nb = gaussian.predict(X_test)

auc_nb = round(roc_auc_score(y_test, pred_nb)*100,2)

print("AUC: ",  auc_nb)
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, y_train)

acc_p = round(perceptron.score(X_test, y_test) * 100, 2)

acc_p
pred_p = perceptron.predict(X_test)

auc_p = round(roc_auc_score(y_test, pred_p)*100,2)

print("AUC: ",  auc_p)
from sklearn.tree import DecisionTreeClassifier

#Building the model

model_ct = DecisionTreeClassifier(criterion='gini',random_state=1)

model_ct.fit(X_train,y_train) ## training the model

## checking the metrics of model on test data

acc_ct=round(model_ct.score(X_test, y_test)*100,2)

print("Accuracy on test data: ",acc_ct)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

pred_ct = model_ct.predict(X_test)

auc_ct = round(roc_auc_score(y_test, pred_ct)*100,2)

print("AUC: ",  auc_ct)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=101, random_state=0)

rf.fit(X_train, y_train)

## checking the metrics of model on train/test data

acc_rf=round(rf.score(X_test, y_test)*100,2)

print("Accuracy on test data: ",acc_rf)
pred_rf = rf.predict(X_test)

auc_rf = round(roc_auc_score(y_test, pred_rf)*100,2)

print("AUC: ",  auc_rf)
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=1)

gb.fit(X_train, y_train)

## checking the metrics of model on test data

acc_gb=round(gb.score(X_test, y_test)*100,2)

print("Accuracy on test set: ",acc_gb)
pred_gb = gb.predict(X_test)

auc_gb = round(roc_auc_score(y_test, pred_gb)*100,2)

print("AUC: ",  auc_gb)
import xgboost as xgb

xgb = xgb.XGBClassifier(random_state=1)

xgb.fit(X_train, y_train)

## checking the metrics of model on test data

acc_xgb=round(xgb.score(X_test, y_test)*100,2)

print("Accuracy on test data: ",acc_xgb)
pred_xgb = xgb.predict(X_test)

auc_xgb = round(roc_auc_score(y_test, pred_xgb)*100,2)

print("AUC: ",  auc_xgb)
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

sgd.fit(X_train, y_train)

## checking the metrics of model on test data

acc_sgd=round(sgd.score(X_test, y_test)*100,2)

print("Accuracy on test set: ",acc_sgd)
pred_sgd = sgd.predict(X_test)

auc_sgd = round(roc_auc_score(y_test, pred_sgd)*100,2)

print("AUC: ",  auc_sgd)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Gradient Boosting', 

              'Decision Tree','XGBoosting'],

    'Accuracy': [acc_svc, acc_knn, acc_log, 

              acc_rf, acc_sgd, acc_p, 

              acc_sgd, acc_gb, acc_ct, acc_xgb],

    'AUC' : [auc_svc, auc_knn, auc_log, 

              auc_rf, auc_sgd, auc_p, 

              auc_sgd, auc_gb, auc_ct,auc_xgb]})

models.sort_values(by='Accuracy', ascending=False)
rf_importances = pd.Series(rf.feature_importances_, index=X_features.columns)

xgb_importances = pd.Series(xgb.feature_importances_, index=X_features.columns)

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

rf_importances.nlargest(25).sort_values(ascending=True).plot(kind='barh', title='Importance by Rf')

plt.subplot(1,2,2)

xgb_importances.nlargest(25).sort_values(ascending=True).plot(kind='barh',title='Importance by XGB')
### Finding best param for SVM model

param = {

    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],

    'kernel':['linear', 'rbf'],

    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]

}

grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)

grid_svc.fit(X_train, y_train)

print("Accuracy after grid search: ",round(grid_svc.score(X_test, y_test)*100,2))
best=grid_svc.best_params_

print(best)
#Let's run our SVC again with the best parameters.

svc2 = SVC(C = 1.2, gamma =  1.2, kernel= 'rbf')

svc2.fit(X_train, y_train)

pred_svc2 = svc2.predict(X_test)

acc_svc2 =round(svc2.score(X_test, y_test)*100,2)

print("Accuracy on test set: ",acc_svc2)

auc_svc2 = round(roc_auc_score(y_test, pred_svc2)*100,2)

print("AUC: ",  auc_svc2)
#Now lets try to do some evaluation for random forest model using cross validation.

rfc_eval = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 10)

rfc_eval.mean()
param_grid = {'n_estimators': [101,201,251], 'max_features': [6,7,8], 'max_depth':[7,8,9]}

rf1 = GridSearchCV(RandomForestClassifier(), param_grid, cv=10, 

                   scoring=make_scorer(accuracy_score))

rf1.fit(X_train, y_train)

print("Accuracy :",round(rf1.score(X_test, y_test)*100,2))
print('---Comparison Of Best 2 Models---')

print('RF Model Accuracy:',acc_rf,',Auc:',auc_rf)

print('SVM Model Accuracy:',acc_svc2,',Auc:',auc_svc2)
print("=== Confusion Matrix of RF Model ===")

print(confusion_matrix(y_test, pred_rf))

print("=== Classification Report of RF Model ===")

print(classification_report(y_test, pred_rf))

print('\n')

print("=== Confusion Matrix of SVM Model ===")

print(confusion_matrix(y_test, pred_svc2))

print("=== Classification Report of SVM Model ===")

print(classification_report(y_test, pred_svc2))