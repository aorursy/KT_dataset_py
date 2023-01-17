# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
##importing required libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

import numpy as np

##Model

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier



##Performance metrics

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score,accuracy_score, make_scorer

##reading the bank data

df=pd.read_csv('/kaggle/input/personal-loan-modeling/Bank_Personal_Loan_Modelling.csv')

df.head()
df.tail()
df.info()
#Removing ID column which is of no relevance

df1= df.drop(columns =['ID', 'ZIP Code'])
df1.isnull().sum()##checking missing values
df1.describe()
df1.loc[df1.Experience < 0, ['Experience']] = 0

df1.describe()
##Lets see the distribution of target column- Personal Loan

print(df1.groupby('Personal Loan').size())

sns.countplot(df1['Personal Loan'],label="Count")

plt.title("Distribution of Target Variable")

plt.show()
sns.distplot(df1["Age"])
sns.distplot(df1["Experience"])
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)

df1.groupby('Personal Loan')['Income'].mean().plot(kind='bar',title='Income')

plt.subplot(2,3,2)

df1.groupby('Personal Loan')['CCAvg'].mean().plot(kind='bar', title='Average CC Spend')

plt.subplot(2,3,3)

df1.groupby('Personal Loan')['Age'].mean().plot(kind='bar', title='Age')

plt.subplot(2,3,4)

df1.groupby('Personal Loan')['Experience'].mean().plot(kind='bar', title='Experience')

plt.subplot(2,3,5)

df1.groupby('Personal Loan')['Mortgage'].mean().plot(kind='bar', title='Mortagage')
sns.heatmap(df1.corr())

plt.show()
sns.lmplot(x='Income',y='CCAvg',data=df1,fit_reg=False,hue='Personal Loan') 

sns.lmplot(x='Income',y='Mortgage',data=df1,fit_reg=False,hue='Personal Loan') 

plt.show()
pd.crosstab(df1['Securities Account'],df['Personal Loan']).plot(kind='bar',stacked=True,title='Securities')

pd.crosstab(df1['CD Account'],df['Personal Loan']).plot(kind='bar',stacked=True,title='CD Account')
pd.crosstab(df1['Online'],df1['Personal Loan']).plot(kind='bar',stacked=True,title='Online')

pd.crosstab(df1['CreditCard'],df1['Personal Loan']).plot(kind='bar',stacked=True,title='Credit Card')
##Plotting family

edu=pd.crosstab(df1['Family'],df1['Personal Loan'])

edu.div(edu.sum(1).astype(float),axis=0).plot(kind='bar',

                                              stacked=True,title='% Family')
##Plotting education

edu=pd.crosstab(df1['Education'],df1['Personal Loan'])

edu.div(edu.sum(1).astype(float),axis=0).plot(kind='bar',

                                              stacked=True,title='% Education')
#splitting the data into train-test in 80-20 ratio

X_train, X_test, Y_train, Y_test = train_test_split(df1.loc[:, df1.columns != 'Personal Loan'], df1['Personal Loan'], 

                                                    stratify=df1['Personal Loan'], 

                                                    random_state=66, test_size =0.2)

print("Training Data: ",X_train.shape, Y_train.shape)

print("Test Data: ",X_test.shape, Y_test.shape)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print("Accuracy on test data: ",acc_log)
coeff_df = pd.DataFrame(df1.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
##Predicting on test data

pred_log = logreg.predict(X_test)

auc_log = round(roc_auc_score(Y_test, pred_log)*100,2)

recall_log = round(recall_score(Y_test, pred_log)*100,2)

print("AUC: ",  auc_log)

print("Recall: ",  recall_log)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

## checking the metrics of model on test data

acc_svc = round(svc.score(X_test, Y_test)*100, 2)

print("Accuracy on test set: ",acc_svc)
##Predicting on test data

pred_svc = svc.predict(X_test)

auc_svc = round(roc_auc_score(y_test, pred_svc)*100,2)

recall_svc = round(recall_score(y_test, pred_svc)*100,2)

print("AUC: ",  auc_svc)

print("Recall: ",  recall_svc)
knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(X_train, Y_train)

## checking the metrics of model on test data

acc_knn = round(knn.score(X_test, Y_test)*100, 2)

print("Accuracy on test data: ",acc_knn)
### Predicting on test data and checking the metrics

pred_knn = knn.predict(X_test)

auc_knn = round(roc_auc_score(Y_test, pred_knn)*100,2)

recall_knn = round(recall_score(Y_test, pred_knn)*100,2)

print("AUC: ",  auc_knn)

print("Recall: ",  recall_knn)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)

acc_gaussian
pred_nb = gaussian.predict(X_test)

auc_nb = round(roc_auc_score(Y_test, pred_nb)*100,2)

recall_nb = round(recall_score(Y_test, pred_nb)*100,2)

print("AUC: ",  auc_nb)

print("Recall: ",  recall_nb)
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

acc_p = round(perceptron.score(X_test, Y_test) * 100, 2)

acc_p
### Predicting on test data

pred_p = gaussian.predict(X_test)

auc_p = round(roc_auc_score(Y_test, pred_p)*100,2)

recall_p = round(recall_score(Y_test, pred_p)*100,2)

print("AUC: ",  auc_p)

print("Recall: ",  recall_p)
#Building the model

model_ct = DecisionTreeClassifier(criterion='gini',random_state=1)

model_ct.fit(X_train,Y_train) ## training the model

## checking the metrics of model on test data

acc_ct=round(model_ct.score(X_test, Y_test)*100,2)

print("Accuracy on test data: ",acc_ct)
model_ct = DecisionTreeClassifier(criterion='gini',random_state=1, max_depth=5)

model_ct.fit(X_train,Y_train) ## training the model

## checking the metrics of model on test data

acc_ct=round(model_ct.score(X_test, Y_test)*100,2)

print("Accuracy on test data: ",acc_ct)
### Predicting on test data

pred_ct = model_ct.predict(X_test)

auc_ct = round(roc_auc_score(Y_test, pred_ct)*100,2)

recall_ct = round(recall_score(Y_test, pred_ct)*100,2)

print("AUC: ",  auc_ct)

print("Recall: ",  recall_ct)
rf = RandomForestClassifier(n_estimators=101, random_state=1)

rf.fit(X_train, Y_train)

## checking the metrics of model on test data

acc_rf=round(rf.score(X_test, Y_test)*100,2)

print("Accuracy on test set: ",acc_rf)
### Predicting on test data

pred_rf = rf.predict(X_test)

auc_rf = round(roc_auc_score(Y_test, pred_rf)*100,2)

recall_rf = round(recall_score(Y_test, pred_rf)*100,2)

print("AUC: ",  auc_rf)

print("Recall: ",  recall_rf)
gb = GradientBoostingClassifier(random_state=1)

gb.fit(X_train, Y_train)

## checking the metrics of model on test data

acc_gb=round(gb.score(X_test, Y_test)*100,2)

print("Accuracy on test set: ",acc_gb)
### Predicting on test data

pred_gb = gb.predict(X_test)

auc_gb = round(roc_auc_score(Y_test, pred_gb)*100,2)

recall_gb = round(recall_score(Y_test, pred_gb)*100,2)

print("AUC: ",  auc_gb)

print("Recall: ",  recall_gb)
import xgboost as xgb

xgb = xgb.XGBClassifier(random_state=1)

xgb.fit(X_train, Y_train)

## checking the metrics of model on test data

acc_xgb=round(xgb.score(X_test, Y_test)*100,2)

print("Accuracy on test data: ",acc_xgb)
### Predicting on test data

pred_xgb = xgb.predict(X_test)

auc_xgb = round(roc_auc_score(Y_test, pred_xgb)*100,2)

recall_xgb = round(recall_score(Y_test, pred_xgb)*100,2)

print("AUC: ",  auc_xgb)

print("Recall: ",  recall_xgb)
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)



## checking the metrics of model on test data

acc_sgd=round(sgd.score(X_test, Y_test)*100,2)

print("Accuracy on test set: ",acc_sgd)
### Predicting on test data

pred_sgd = sgd.predict(X_test)

auc_sgd = round(roc_auc_score(Y_test, pred_sgd)*100,2)

recall_sgd = round(recall_score(Y_test, pred_sgd)*100,2)

print("AUC: ",  auc_sgd)

print("Recall: ",  recall_sgd)
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

              auc_sgd, auc_gb, auc_ct,auc_xgb],

    'Recall': [recall_svc, recall_knn, recall_log, 

              recall_rf, recall_sgd, recall_p, 

              recall_sgd, recall_gb, recall_ct,recall_xgb]})

models.sort_values(by='Accuracy', ascending=False)
rf_importances = pd.Series(rf.feature_importances_, index=X_train.columns)

xgb_importances = pd.Series(xgb.feature_importances_, index=X_train.columns)

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

rf_importances.nlargest(25).sort_values(ascending=True).plot(kind='barh', title='Importance by Rf')

plt.subplot(1,2,2)

xgb_importances.nlargest(25).sort_values(ascending=True).plot(kind='barh',title='Importance by XGB')
##lets try to do some evaluation for random forest model using cross validation.

rfc_eval = cross_val_score(estimator = rf, X = X_train, y = Y_train, cv = 10)

rfc_eval.mean()
##lets try to find best hyperparameter for random forest model using GridSearchCV.

param_grid = {'n_estimators': [101,201,251], 'max_features': [6,7,8], 'max_depth':[7,8,9]}

rf1 = GridSearchCV(RandomForestClassifier(), param_grid, cv=10, 

                   scoring=make_scorer(accuracy_score))

rf1.fit(X_train, Y_train)

print("Accuracy :",round(rf1.score(X_test, Y_test)*100,2))
best=rf1.best_params_

print(best)
##Tuning the Decision Tree model

model_t = DecisionTreeClassifier(random_state=1,max_depth=5)

model_t.fit(X_train,Y_train) ## training the model

## checking the accuracy of model on test data

acc_t=round(model_t.score(X_test, Y_test)*100,2)

print("Accuracy on test data: ",acc_t)
##Predicting on test data



pred_t = model_t.predict(X_test)

print("=== Confusion Matrix ===")

print(confusion_matrix(Y_test, pred_t))

print('\n')

print("=== Classification Report ===")

print(classification_report(Y_test, pred_t))

print('\n')

auc_t = round(roc_auc_score(Y_test, pred_t)*100,2)

print("AUC: ",  auc_t)

recall_t = round(recall_score(Y_test, pred_t)*100,2)

print("Recall: ",  recall_t)
##Plotting the tree

plt.figure(figsize=(25,10))

a= plot_tree(model_t, 

             feature_names=X_train.columns,

             filled=True, 

              rounded=True, 

              fontsize=14)
print('---Comparison Of Best 2 Models---')

print('Decision Tree Model Accuracy:',acc_t,',Auc:',auc_t,',Recall:',recall_t)

print(' XGBoosting Model Accuracy:',acc_xgb,',Auc:',auc_xgb,',Recall:',recall_xgb)