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
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.isnull().sum()
df.info()
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(df['DEATH_EVENT'])
sns.countplot(df['DEATH_EVENT'], hue=df['diabetes'])
plt.figure(figsize=(12,12))
sns.heatmap(df.corr())
plt.hist(df['age'], bins=20)
plt.show()
sns.pairplot(df)
df.columns

plt.hist(df['creatinine_phosphokinase'], bins=20)
plt.show()
sns.scatterplot(x=df['age'], y=df['serum_creatinine'], hue=df['DEATH_EVENT'])
sns.scatterplot(x=df['time'], y=df['serum_creatinine'], hue=df['DEATH_EVENT'])
sns.scatterplot(x=df['age'], y=df['platelets'], hue=df['DEATH_EVENT'])
sns.scatterplot(x=df['age'], y=df['serum_sodium'], hue=df['DEATH_EVENT'])
sns.countplot(df['diabetes'], hue=df['DEATH_EVENT'])
sns.countplot(df['high_blood_pressure'], hue=df['DEATH_EVENT'])
sns.countplot(df['sex'], hue=df['DEATH_EVENT'])
sns.countplot(df['smoking'], hue=df['DEATH_EVENT'])
sns.countplot(df['anaemia'], hue=df['DEATH_EVENT'])
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_recall_curve, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0, stratify=y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# example of grid searching key hyperparametres for logistic regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2', 'l1']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=3, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
best_model1 = LogisticRegression(C=1.0, solver='liblinear', penalty = 'l1')
best_model1.fit(X_train, y_train)
y_pred = best_model1.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve

print(confusion_matrix(y_test, y_pred))
print('Accuracy of our model is: ', accuracy_score(y_test, y_pred)*100)
# get importance
importance = best_model1.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in X.columns], importance)
plt.xticks(rotation=90)
plt.show()
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier()
n_estimators = [100,200, 300, 400]
max_depth = [6,5,7]
min_samples_split = [8,9,7,6]

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split)

gridF = GridSearchCV(rf1, hyperF, cv = 5, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (bestF.best_score_, bestF.best_params_))
rf_best = RandomForestClassifier(max_depth= 5, min_samples_split= 8, n_estimators= 300)
rf_best.fit(X_train, y_train)
y_pred = rf_best.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print('Accuracy of our model is: ', accuracy_score(y_test, y_pred))
features = pd.DataFrame()
features['Feature'] = X.columns
features['Importance'] = rf_best.feature_importances_
features.sort_values(by=['Importance'], ascending=False, inplace=True)
features.set_index('Feature', inplace=True)

features.plot(kind='bar', figsize=(20, 10))
from sklearn.svm import SVC 
# defining parameter range 
param_grid = {'C': [ 1, 10, 100], 
            'gamma': [ 0.1, 0.01, 0.001], 
            'kernel': ['rbf', 'poly']} 

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, cv=5, scoring='accuracy') 

# fitting the model for grid search 
grid.fit(X_train, y_train)
# print best parameter after tuning 
print(grid.best_params_) 

# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)
y_pred_svm = grid.predict(X_test)

print(confusion_matrix(y_test, y_pred_svm))
print('Accuracy of our model is: ', accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=200, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy of our model is: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))
features = pd.DataFrame()
features['Feature'] = X.columns
features['Importance'] = clf.feature_importances_
features.sort_values(by=['Importance'], ascending=False, inplace=True)
features.set_index('Feature', inplace=True)

features.plot(kind='bar', figsize=(20, 10))
from sklearn.neighbors import KNeighborsClassifier
error_rate = []

for i in range(1, 40):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  pred_i = knn.predict(X_test)
  error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12,12))

plt.plot(range(1,40), error_rate, color='b', linestyle='dashed', marker='o', markerfacecolor='red')
plt.title('Error rate vs K')
knn_11 = KNeighborsClassifier(n_neighbors=11)
knn_11.fit(X_train, y_train)
pred = knn_11.predict(X_test)

print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(f1_score(y_test, pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#ROC for Ada Boost
ada_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

#ROC for Random Forrest
rf_roc_auc = roc_auc_score(y_test, rf_best.predict(X_test))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_best.predict_proba(X_test)[:,1])

#ROC Curve for Random Forest & Ada Boost
plt.figure()
plt.plot(fpr, tpr, label='Ada Boost (area = %0.2f)' % ada_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()
