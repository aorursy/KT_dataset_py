import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import time
from contextlib import contextmanager
churn_data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
churn_data.info() # NAN value bulunmamaktadir.
churn_data.head()
churn_data.drop(labels=['RowNumber','CustomerId','Surname'],
                axis=1,
                inplace=True)
churn_data.head()
categorical_features = ["Geography","Gender","NumOfProducts","HasCrCard","IsActiveMember"]

numerical_features = ["CreditScore","Age","Tenure","Balance","EstimatedSalary"]

target = "Exited"
churn_data[numerical_features].describe()
churn_data[numerical_features].hist(bins=30, figsize=(10, 10));
fig, ax = plt.subplots(1, 5, figsize=(30, 5))
churn_data[churn_data.Exited == 0][numerical_features].hist(bins=30, color="blue", alpha=0.5, ax=ax);
churn_data[churn_data.Exited == 1][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax);
g = sns.pairplot(churn_data,hue = 'Exited')
churn_data_cleaned = pd.get_dummies(churn_data,
                                    prefix_sep='_', 
                                    columns=categorical_features,
                                    drop_first=True,
                                    dtype=int)
churn_data_cleaned.head()
churn_data_cleaned[numerical_features].describe()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(churn_data_cleaned[numerical_features])
churn_data_cleaned[numerical_features] = data_scaled
churn_data_cleaned.head()
df = churn_data_cleaned
X = df.drop(['Exited'], axis=1)
y = df["Exited"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
names = ["LogisticRegression","GaussianNB","KNeighborsClassifier","LinearSVC","SVC",
         "DecisionTreeClassifier","RandomForestClassifier","GradientBoostingClassifier",
         "XGBClassifier","LGBMClassifier","CatBoostClassifier"]
    
    
classifiers = [LogisticRegression(), GaussianNB(), KNeighborsClassifier(), LinearSVC(), SVC(),
               DecisionTreeClassifier(),RandomForestClassifier(), GradientBoostingClassifier(),
               XGBClassifier(), LGBMClassifier(), CatBoostClassifier(verbose = False)]
for name, clf in zip(names, classifiers):
    
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    msg = "%s: %f" % (name, acc)
    print(msg)
results = []
A = []

for name, clf in zip(names, classifiers):
        
    kfold = KFold(n_splits=10, random_state=1001)
    cv_results = cross_val_score(clf, X, y, cv = kfold, scoring = "accuracy")
    results.append(cv_results)
    A.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean() , cv_results.std())
    print(msg)
cv_result = {}
best_estimators = {}
best_params = {}
    
clf = GridSearchCV(RandomForestClassifier(), 
                   param_grid = {"max_features": ["log2","Auto","None"],
                 "min_samples_split":[2,3,5],
                 "min_samples_leaf":[1,3,5],
                 "bootstrap":[True,False],
                 "n_estimators":[50,100,150],
                 "criterion":["gini","entropy"]},
                   cv =10, scoring = "accuracy", 
                   n_jobs = -1, 
                   verbose = False)

clf.fit(X_train,y_train)
cv_result = clf.best_score_
best_estimators = clf.best_estimator_
best_params = clf.best_params_
print('cross validation accuracy : %.3f'%cv_result)
print(best_estimators)
print(best_params)
y_pred =  best_estimators.fit(X_train,y_train).predict(X_test)

accuracy=accuracy_score(y_pred, y_test)

print('accuracy score :', "%.3f" %accuracy)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_test)
from sklearn.metrics import precision_score
precision_score(y_pred, y_test)
from sklearn.metrics import f1_score
f1_score(y_pred, y_test)
from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(y_pred, y_test)
auc_LR = roc_auc_score(y_pred, y_test)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_pred, y_test)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))