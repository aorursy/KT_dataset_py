# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings                       

warnings.filterwarnings("ignore")

dataset = pd.read_csv('../input/advertising.csv')

dataset.head()
dataset.info()
#Check for duplicates

dataset.duplicated().sum()
#Check for missing features

dataset.isnull().sum()
#Correlation analysis

corrm = dataset.corr()

corrm['Clicked on Ad'].sort_values(ascending = False)
#Exploring target

dataset['Clicked on Ad'].value_counts()
sns.countplot(x = 'Clicked on Ad', data = dataset)
#Statistical information on the numeric features

dataset.describe()
#Statistical information on the categorical features

categ_cols = ['Ad Topic Line', 'City', 'Country']

dataset[categ_cols].describe(include = ['O'])
dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

dataset['Timestamp']

dataset['Year'] = dataset['Timestamp'].dt.year

dataset['Month'] = dataset['Timestamp'].dt.month

dataset['Day'] = dataset['Timestamp'].dt.day

dataset['Hour'] = dataset['Timestamp'].dt.hour

dataset['Weekday'] = dataset['Timestamp'].dt.dayofweek

dataset = dataset.drop(['Timestamp'], axis=1)

dataset.head(10)
#Relationship between numerical featuers

sns.pairplot(dataset, hue = 'Clicked on Ad', 

             vars = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage'], palette = 'Greens_r')
dataset = dataset.drop(['Year'], axis=1)

#Correlation heatmap with new features

fig = plt.figure(figsize = (12,10))

sns.heatmap(dataset.corr(), cmap='Greens', annot = True)
X = dataset.iloc[:,[0,1,2,3,6,9,10,11,12]].values

y = dataset.iloc[:,8].values

#Splitting the data into train and test sets 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Scaling the data

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

X_train = standardScaler.fit_transform(X_train)

X_test = standardScaler.transform(X_test)
#Initiate and fit the model of Logistic Regression on training data

from sklearn.linear_model import LogisticRegression

log_rg = LogisticRegression()

log_rg.fit(X_train, y_train)

#Prediction

y_log_rg = log_rg.predict(X_test)
#Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_log_rg)

sns.heatmap(cm,annot=True,fmt='3.0f',cmap="Greens")

plt.title('Confusion matrix for Logistic Regression', y=1.05, size=15)
#Classification report

from sklearn.metrics import classification_report

cr = classification_report(y_test, y_log_rg)

print(cr)
#Initiate and fit the model of Naive Bayes on training data

from sklearn.naive_bayes import GaussianNB

naive_b = GaussianNB()

naive_b.fit(X_train, y_train)

#Prediction

y_naive = naive_b.predict(X_test)
#Confusion matrix

from sklearn.metrics import confusion_matrix

naive_cm = confusion_matrix(y_test, y_naive)

sns.heatmap(naive_cm,annot=True,fmt='3.0f',cmap="Blues")

plt.title('Confusion matrix for Naive Bayes', y=1.05, size=15)
#Classification report

from sklearn.metrics import classification_report

naive_cr = classification_report(y_test, y_naive)

print(naive_cr)
#Initiate and fit the model of Random Forest on training data

from sklearn.ensemble import RandomForestClassifier

randm_frst = RandomForestClassifier()

randm_frst.fit(X_train, y_train)

#Prediction

y_frst = randm_frst.predict(X_test)
#Confusion matrix

from sklearn.metrics import confusion_matrix

frst_cm = confusion_matrix(y_test, y_frst)

sns.heatmap(frst_cm,annot=True,fmt='3.0f',cmap="Reds")

plt.title('Confusion matrix for Random Forest', y=1.05, size=15)
#Classification report

from sklearn.metrics import classification_report

frst_cr = classification_report(y_test, y_frst)

print(frst_cr)
#Initiate and fit the model of K-Nearest Neighbors on training data

from sklearn.neighbors import KNeighborsClassifier

kneighbors = KNeighborsClassifier()

kneighbors.fit(X_train, y_train)

#Prediction

y_knn = kneighbors.predict(X_test)
#Confusion matrix

from sklearn.metrics import confusion_matrix

knn_cm = confusion_matrix(y_test, y_knn)

sns.heatmap(knn_cm,annot=True,fmt='3.0f',cmap="mako")

plt.title('Confusion matrix for K-Nearest Neighbors', y=1.05, size=15)
#Classification report

from sklearn.metrics import classification_report

knn_cr = classification_report(y_test, y_knn)

print(knn_cr)
from sklearn.metrics import f1_score

f1_log = f1_score(y_test, y_log_rg)

f1_naive = f1_score(y_test, y_naive)

f1_frst = f1_score(y_test, y_frst)

f1_knn = f1_score(y_test, y_knn)

from pandas import DataFrame

scores = {'Model':  ['Logistic Regression','Naive_Bayes', 'Random Forest', 'KNN'], 

          'f1 score': [f1_log, f1_naive, f1_frst, f1_knn]}

f1_scores = DataFrame (scores, columns = ['Model','f1 score'])

f1_scores
sns.barplot(x="Model", y="f1 score", data=f1_scores, palette="Greens_r")
#Choose hyperparameters for Random Forest model

from sklearn.model_selection import GridSearchCV

param_frst = [{"n_estimators": [10,100,200,300,500], "criterion": ["gini", "entropy"]}]

grid_search_frst = GridSearchCV(estimator=randm_frst,

                          param_grid=param_frst,

                          scoring = 'accuracy',

                          cv=10)

grid_search_frst = grid_search_frst.fit(X_train, y_train)
#Calculation best accuracy for Random Forest model

best_acc_frst = grid_search_frst.best_score_

best_acc_frst
#Calculation best parameters for Random Forest model

best_params_frst = grid_search_frst.best_params_

best_params_frst
#Choose hyperparameters for K-Nearest Neighbors model

from sklearn.model_selection import GridSearchCV

param_knn = [{"n_neighbors": range(1,10), "weights": ["uniform", "distance"]}]

grid_search_knn = GridSearchCV(estimator=kneighbors,

                          param_grid=param_knn,

                          scoring = 'accuracy',

                          cv=10)

grid_search_knn = grid_search_knn.fit(X_train, y_train)
#Calculation best accuracy for K-Nearest Neighbors model

best_acc_knn = grid_search_knn.best_score_

best_acc_knn
#Calculation best parameters for K-Nearest Neighbors model

best_params_knn = grid_search_knn.best_params_

best_params_knn
#Initiate and fit the model of Random Forest on training data with hyperparamets

from sklearn.ensemble import RandomForestClassifier

randm_frst_imp = RandomForestClassifier(n_estimators=100, criterion='gini')

randm_frst_imp.fit(X_train, y_train)

#Prediction

y_frst_imp = randm_frst_imp.predict(X_test)
#Initiate and fit the model of K-Nearest Neighbors on training data with hyperparamets

from sklearn.neighbors import KNeighborsClassifier

kneighbors_imp = KNeighborsClassifier(n_neighbors=5, weights= 'uniform')

kneighbors_imp.fit(X_train, y_train)

#Prediction

y_knn_imp = kneighbors_imp.predict(X_test)
#Recalculation f1 score

f1_frst_imp = f1_score(y_test, y_frst_imp)

f1_knn_imp = f1_score(y_test, y_knn_imp)

scores_imp = {'Model':  ['Logistic Regression','Naive_Bayes', 'Random Forest', 'KNN'], 

          'f1 score': [f1_log, f1_naive, f1_frst_imp, f1_knn_imp]}

f1_scores_imp = DataFrame (scores_imp, columns = ['Model','f1 score'])

f1_scores_imp.sort_values(by=['f1 score'], ascending=False)
#Confusion matrix for best model

frst_imp_cm = confusion_matrix(y_test, y_frst_imp)

sns.heatmap(frst_imp_cm,annot=True,fmt='3.0f',cmap="PuBu_r")

plt.title('Confusion matrix for Random Forest with hyperparameters', y=1.05, size=15)

from sklearn.metrics import roc_auc_score

lr_auc = roc_auc_score(y_test, log_rg.predict(X_test))

rf_roc_auc = roc_auc_score(y_test, randm_frst_imp.predict(X_test))



# Create ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, log_rg.predict_proba(X_test)[:,1])

rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, randm_frst_imp.predict_proba(X_test)[:,1])



plt.figure()

# Plot Random Forest ROC

plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)

# Plot Logistic Regression ROC

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Base Rate ROC

plt.plot([0,1], [0,1],label='Base Rate')



plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.10])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
