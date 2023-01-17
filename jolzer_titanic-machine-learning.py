import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import os
print(os.listdir("../input"))
#importing functions
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
#models
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
#read table
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_combined = pd.concat([df_train.drop("Survived",axis = 1),df_test])
df_combined.info()
df_combined.describe()
#determine which variables seem related to survived.
df_train.corr()['Survived']
#Pclass and Fare have the highest correlation with survived.
#visual analysis to determine which variables seem correlated.
f, ax = plt.subplots(figsize=(5, 4))
corr_combined = df_combined.corr()
sns.heatmap(corr, mask=np.zeros_like(corr_combined, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
#remove cabin column. Get dummy for Pclass, sex and embarked
df_combined_new = pd.get_dummies(\
        pd.get_dummies(df_combined.drop(['PassengerId','Name','Ticket','Cabin'], axis=1),columns = ['Sex'],drop_first = True)\
                        , columns = ['Embarked'])

#fill NaN values in Age and Fare as the median.
df_combined_new = df_combined_new.fillna(df_combined_new.median())
X = df_combined_new[:891].values
y = df_train['Survived'].values
#split the training and holding set.
X_train, X_hold, y_train, y_hold = train_test_split(X,y,test_size = 0.2, random_state = 42)
# Setup the pipeline steps.
steps_svc = [
         ('scaler', StandardScaler()),
         ('svc', SVC(probability = True, verbose = False))]

pipeline_svc = Pipeline(steps_svc)
# Specify the hyperparameter space
parameters_svc = {'svc__C':np.logspace(-3,3,10),
                  'svc__gamma':np.logspace(-3,3,10)
                 }
cv_svc = GridSearchCV(pipeline_svc, parameters_svc, cv=5)
#fit model
cv_svc.fit(X_train,y_train)
# Compute and print metrics
y_pred_svc = cv_svc.predict(X_hold)

print("Accuracy: {}".format(cv_svc.score(X_hold, y_hold)))
print(classification_report(y_hold, y_pred_svc))
print("Tuned Model Parameters: {}".format(cv_svc.best_params_))
#print auc-roc metric

# Compute predicted probabilities
y_pred_prob_svc = cv_svc.predict_proba(X_hold)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_hold, y_pred_prob_svc)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC: "+ str(roc_auc_score(y_hold, y_pred_prob_svc)))
# Setup the pipeline steps.
steps_logit = [
         ('scaler', StandardScaler()),
         ('logit', LogisticRegression())]

pipeline_logit = Pipeline(steps_logit)
# Specify the hyperparameter space
parameters_logit = {'logit__C':np.logspace(-4,4,20),
                 'logit__penalty':['l1','l2']}
cv_logit = GridSearchCV(pipeline_logit, parameters_logit, cv=5)
#fit model
cv_logit.fit(X_train,y_train)
# Compute and print metrics
y_pred_logit = cv_logit.predict(X_hold)

print("Accuracy: {}".format(cv_logit.score(X_hold, y_hold)))
print(classification_report(y_hold, y_pred_logit))
print("Tuned Model Parameters: {}".format(cv_logit.best_params_))
#print auc-roc metric

# Compute predicted probabilities
y_pred_prob_logit = cv_logit.predict_proba(X_hold)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_hold, y_pred_prob_logit)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC: "+ str(roc_auc_score(y_hold, y_pred_prob_logit)))
# Setup the pipeline steps.
steps_knn = [
         ('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]

pipeline_knn = Pipeline(steps_knn)
# Specify the hyperparameter space
parameters_knn = {'knn__n_neighbors':np.arange(1,20)
                 }
cv_knn = GridSearchCV(pipeline_knn, parameters_knn, cv=5)
#fit model
cv_knn.fit(X_train,y_train)
# Compute and print metrics
y_pred_knn = cv_knn.predict(X_hold)

print("Accuracy: {}".format(cv_knn.score(X_hold, y_hold)))
print(classification_report(y_hold, y_pred_knn))
print("Tuned Model Parameters: {}".format(cv_knn.best_params_))
#print auc-roc metric

# Compute predicted probabilities
y_pred_prob_knn = cv_knn.predict_proba(X_hold)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_hold, y_pred_prob_knn)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC: "+ str(roc_auc_score(y_hold, y_pred_prob_knn)))
# Setup the pipeline steps.
steps_xgb = [
         ('scaler', StandardScaler()),
         ('xgb', xgb.XGBClassifier())]

pipeline_xgb = Pipeline(steps_xgb)
# Specify the hyperparameter space
parameters_xgb = {'xgb__learning_rate':[0.01,0.015,0.025,0.05,0.1],
                 'xgb__gamma':[0.05,0.07,0.09,0.12,0.18,0.25,0.3,0.5,0.7,0.9,1.0],
                   'xgb__max_depth':[3,5,7,9,12,15,17,25],
                   'xgb__min_child_weight':[1,3,5,7],
                   'xgb__subsample':[0.5,0.6,0.7,0.8,0.9,1.0],
                   'xgb__colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1.0],
                   'xgb__reg_lambda':np.logspace(-3,2,10),
                   'xgb__reg_alpha':[0,0.1,0.5,1.0,2]}
cv_rand_xgb = RandomizedSearchCV(pipeline_xgb,param_distributions=parameters_xgb, cv=5, n_iter=1000)
#fit model
cv_rand_xgb.fit(X_train,y_train)
# Compute and print metrics
y_pred_xgb = cv_rand_xgb.predict(X_hold)

print("Accuracy: {}".format(cv_rand_xgb.score(X_hold, y_hold)))
print(classification_report(y_hold, y_pred_xgb))
print("Tuned Model Parameters: {}".format(cv_rand_xgb.best_params_))
#print auc-roc metric

# Compute predicted probabilities
y_pred_prob_xgb = cv_rand_xgb.predict_proba(X_hold)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_hold, y_pred_prob_xgb)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC: "+ str(roc_auc_score(y_hold, y_pred_prob_xgb)))
X_test = df_combined_new[891:].values
results = cv_rand_xgb.predict(X_test)
output_format = pd.read_csv('../input/gender_submission.csv')
output_format['Survived'] = pd.Series(results)
output_format.to_csv("output_xgb.csv",index = False)
