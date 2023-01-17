import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import scipy
from scipy.stats import randint

import lime
import lime.lime_tabular

data = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
data.head()
data.info()
data.isna().sum()
data['EDUCATION']=np.where(data['EDUCATION'] == 5, 4, data['EDUCATION'])
data['EDUCATION']=np.where(data['EDUCATION'] == 6, 4, data['EDUCATION'])
data['EDUCATION']=np.where(data['EDUCATION'] == 0, 4, data['EDUCATION'])
data['MARRIAGE']=np.where(data['MARRIAGE'] == 0, 3, data['MARRIAGE'])
data['MARRIAGE'].unique()
X = data.drop('default.payment.next.month',axis=1)
Y = data['default.payment.next.month']
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.85,random_state=42)
feature_names = (x_train.columns)
feature_names
class_names = data['default.payment.next.month'].unique()
class_names
param_dist = {"max_depth": [1,2,3,4,5,6,7,8,9],
              "max_features": [1,2,3,4,5,6,7,8,9],
              "min_samples_leaf": [1,2,3,4,5,6,7,8,9],
              "criterion": ["gini", "entropy"]}


dt = DecisionTreeClassifier()


dt_cv = RandomizedSearchCV(dt, param_distributions=param_dist, cv=5, random_state=0)


dt_cv.fit(x_train, y_train)


print("Tuned Decision Tree Parameters: {}".format(dt_cv.best_params_))
tuned_tree = DecisionTreeClassifier(criterion= 'gini', max_depth= 7, 
                                     max_features= 9, min_samples_leaf= 2, 
                                     random_state=0)
tuned_tree.fit(x_train, y_train)
y_pred = tuned_tree.predict(x_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))
plt.figure(figsize=(8,6))
ConfMatrix = confusion_matrix(y_test,tuned_tree.predict(x_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 
            xticklabels = ['Non-default', 'Default'], 
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Decision Tree")

print(classification_report(y_test, y_pred))
x_test_test = x_test.values
predict_fn_dt = lambda x: tuned_tree.predict_proba(x).astype(float)
explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, feature_names=feature_names, class_names=['Wont Default','Will Default'], verbose=False, mode='classification')

choosen_instance = X.iloc[[2]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_dt, num_features=10)
exp.show_in_notebook(show_table=True)
param_dist = {'n_estimators': [50,100,150,200,250],
               "max_features": [1,2,3,4,5,6,7,8,9],
               'max_depth': [1,2,3,4,5,6,7,8,9],
               "criterion": ["gini", "entropy"]}

rf = RandomForestClassifier()

rf_cv = RandomizedSearchCV(rf, param_distributions = param_dist, 
                           cv = 5, random_state=0, n_jobs = -1)

rf_cv.fit(x_train, y_train)

print("Tuned Random Forest Parameters: %s" % (rf_cv.best_params_))
tuned_rf = RandomForestClassifier(criterion= 'entropy', max_depth= 5, 
                                     max_features= 6, n_estimators= 200, 
                                     random_state=0)
tuned_rf.fit(x_train, y_train)
y_pred = tuned_rf.predict(x_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))
plt.figure(figsize=(8,6))
ConfMatrix = confusion_matrix(y_test,tuned_rf.predict(x_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 
            xticklabels = ['Non-default', 'Default'], 
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Random Forest")

print(classification_report(y_test, y_pred))
predict_fn_rf = lambda x: tuned_rf.predict_proba(x).astype(float)
choosen_instance = X.iloc[[546]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf, num_features=10)
exp.show_in_notebook(show_table=True)
xgb = XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': scipy.stats.randint(150, 1000),
              'learning_rate': scipy.stats.uniform(0.01, 0.6),
              'subsample': scipy.stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': scipy.stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }

xgb_cv = RandomizedSearchCV(xgb, 
                         param_distributions = param_dist,  
                         n_iter = 5, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)

xgb_cv.fit(x_train, y_train)
print("Tuned Random Forest Parameters: %s" % (xgb_cv.best_params_))
tuned_xgb = XGBClassifier(colsample_bytree=0.7870,min_child_weight=1, learning_rate=0.0772, 
                                     max_depth= 3, n_estimators= 947, 
                                     subsample=0.413676)
tuned_xgb.fit(x_train, y_train)
y_pred = tuned_xgb.predict(x_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))
plt.figure(figsize=(8,6))
ConfMatrix = confusion_matrix(y_test,tuned_xgb.predict(x_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 
            xticklabels = ['Non-default', 'Default'], 
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - XGBOOST")

print(classification_report(y_test, y_pred))
predict_fn_xgb = lambda x: tuned_xgb.predict_proba(x).astype(float)
choosen_instance = X.iloc[[4564]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_xgb, num_features=10)
exp.show_in_notebook(show_table=True)