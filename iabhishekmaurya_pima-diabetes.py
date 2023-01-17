import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns
diab_df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
diab_df.head()
diab_df.info()
diab_df.describe()
plt.figure(figsize=(16,9))
sns.heatmap(diab_df)
diab_df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(diab_df.corr(), annot = True, cmap ='coolwarm', linewidths=2)
diab_df.shape
diab_df.isnull().sum()
X = diab_df.drop(['Outcome'], axis = 1) 
X.head()
y = diab_df['Outcome']
y.head()
y.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  
from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(penalty = 'l1',solver='liblinear')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuarcy_lr=accuracy_score(y_test, y_pred_lr)
print(accuarcy_lr)
from sklearn.svm import SVC

svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuarcy_svm=accuracy_score(y_test, y_pred_scv)
print(accuarcy_svm)
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion ='entropy')
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(accuracy_dt)
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators = 10)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(accuracy_rf)
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(accuracy_xgb)
from sklearn.ensemble import AdaBoostClassifier

ada_classifier = AdaBoostClassifier()
ada_classifier.fit(X_train, y_train)
y_pred_ada = ada_classifier.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(accuracy_ada)
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 5)]

rf_params = {'n_estimators' : n_estimators,
              'criterion' : ['gini', 'entropy'],
              'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15]}

print(rf_params)
xgb_params={
 "learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma" : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}
from sklearn.model_selection import RandomizedSearchCV

xgb_random_search = RandomizedSearchCV(xgb_classifier, param_distributions = xgb_params,
                                       scoring= 'roc_auc',
                                       n_jobs= -1, verbose= 3)
rf_random_search = RandomizedSearchCV(rf_classifier, param_distributions = rf_params,
                                       scoring= 'roc_auc',
                                       n_jobs= -1, verbose= 3)

xgb_random_search.fit(X_train, y_train)
rf_random_search.fit(X_train, y_train)
rf_random_search.best_params_
tuned_rf_classifier = RandomForestClassifier(n_estimators = 752, 
                                             max_depth = 8, 
                                             criterion = 'entropy')
tuned_rf_classifier.fit(X_train, y_train)
y_pred_tuned_rf = tuned_rf_classifier.predict(X_test)
accuracy_tuned_rf = accuracy_score(y_test, y_pred_tuned_rf)

print("Accuracy is : ",accuracy_tuned_rf)