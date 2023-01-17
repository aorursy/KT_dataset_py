import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline 
sns.set_palette('Set1')
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/heart.csv')
data.head()
data.isnull().sum()
data.info()
sns.distplot(data['age']);
# 1 = male; 0 = female

sns.countplot(data['sex']);
sns.countplot(data['cp']);
sns.distplot(data['trestbps'])
sns.distplot(data['chol'])
sns.countplot(data['fbs']);
sns.countplot(data['restecg']);
sns.distplot(data['thalach'])
sns.countplot(data['exang']);
sns.distplot(data['oldpeak']);
sns.countplot(data['slope']);
sns.countplot(data['ca']);
sns.countplot(data['thal']);
plt.figure(figsize=(14, 5))

sns.distplot(data[data['target'] == 1]['age'], label= "Disease - Yes")

sns.distplot(data[data['target'] == 0]['age'], label= "Disease - No")

plt.legend();
# 1 = male; 0 = female

sns.countplot(data['target'], hue = data['sex']);
sns.countplot(data['target'], hue = data['cp']);
plt.figure(figsize=(14, 5))

sns.distplot(data[data['target'] == 1]['trestbps'], label= "Disease - Yes")

sns.distplot(data[data['target'] == 0]['trestbps'], label= "Disease - No")

plt.legend();
plt.figure(figsize=(14, 5))

sns.distplot(data[data['target'] == 1]['chol'], label= "Disease - Yes")

sns.distplot(data[data['target'] == 0]['chol'], label= "Disease - No")

plt.legend();
sns.countplot(data['target'], hue = data['fbs']);
sns.countplot(data['target'] ,hue = data['restecg']);
plt.figure(figsize=(14, 5))

sns.distplot(data[data['target'] == 1]['thalach'], label= "Disease - Yes")

sns.distplot(data[data['target'] == 0]['thalach'], label= "Disease - No")

plt.legend();
sns.countplot(data['target'], hue = data['exang']);
plt.figure(figsize=(14, 5))

sns.distplot(data[data['target'] == 1]['oldpeak'], label= "Disease - Yes")

sns.distplot(data[data['target'] == 0]['oldpeak'], label= "Disease - No")

plt.legend();
sns.countplot(data['target'], hue = data['slope']);
sns.countplot(data['target'], hue = data['ca']);
sns.countplot(data['target'], hue = data['thal']);
sns.jointplot(x= 'oldpeak' , y= 'chol' ,data= data, kind= 'kde');
plt.figure(figsize=(8,6))

sns.violinplot(x = 'fbs',y= 'trestbps', data = data, hue = 'sex', split=True);
plt.figure(figsize=(8,6))

sns.boxplot(x = 'exang',y= 'trestbps', data = data, hue = 'sex');
plt.figure(figsize=(12,10))

sns.heatmap(data.corr(), annot= True, fmt='.2f')

plt.show();
sns.pairplot(data, hue = 'target');
sex = pd.get_dummies(data['sex'])

cp = pd.get_dummies(data['cp'])

fbs = pd.get_dummies(data['fbs'])

restecg = pd.get_dummies(data['restecg'])

exang = pd.get_dummies(data['exang'])

slope = pd.get_dummies(data['slope'])

ca = pd.get_dummies(data['ca'])

thal = pd.get_dummies(data['thal'])
data = pd.concat([data, sex, cp, fbs, restecg, exang, slope, ca, thal], axis = 1)
data.head()
data.drop(['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], axis = 1, inplace= True)
from sklearn.svm import SVC
model = SVC(probability=True)
from sklearn.model_selection import train_test_split
X = data.drop('target', axis = 1)

y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
print(classification_report(y_test, y_pred))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
y_prob = model.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])

roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(SVC(), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)
train_scores = np.mean(train_scores, axis = 1)

test_scores = np.mean(test_scores, axis = 1)
plt.plot(train_sizes, train_scores, 'o-', label="Training score")

plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")

plt.legend();
train_sizes, train_scores, test_scores = learning_curve(SVC(C=2), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)

train_scores = np.mean(train_scores, axis = 1)

test_scores = np.mean(test_scores, axis = 1)

plt.plot(train_sizes, train_scores, 'o-', label="Training score")

plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")

plt.legend();
train_sizes, train_scores, test_scores = learning_curve(SVC(C=3), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)

train_scores = np.mean(train_scores, axis = 1)

test_scores = np.mean(test_scores, axis = 1)

plt.plot(train_sizes, train_scores, 'o-', label="Training score")

plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")

plt.legend();
train_sizes, train_scores, test_scores = learning_curve(SVC(C=3, gamma=0.1), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)

train_scores = np.mean(train_scores, axis = 1)

test_scores = np.mean(test_scores, axis = 1)

plt.plot(train_sizes, train_scores, 'o-', label="Training score")

plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")

plt.legend();
train_sizes, train_scores, test_scores = learning_curve(SVC(C=3, gamma=0.01), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)

train_scores = np.mean(train_scores, axis = 1)

test_scores = np.mean(test_scores, axis = 1)

plt.plot(train_sizes, train_scores, 'o-', label="Training score")

plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")

plt.legend();
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[1,2,3,4,5,6,7,8,14], 'gamma':[0.1, 0.01, 0.001, 0.0001], 'kernel':['linear', 'poly', 'rbf'], 'degree': [1,2,3,4,5]}

grid = GridSearchCV(param_grid= param_grid, estimator= SVC(), scoring='f1', refit= True, verbose=1)
grid.fit(X_train, y_train)
grid.best_params_
param_grid = {'C':[6,7,8], 'gamma':np.linspace(0.01, 0.02, 10), 'kernel':['rbf'], 'degree': [1,2,3,4,5]}

grid = GridSearchCV(param_grid= param_grid, estimator= SVC(probability= True), scoring='f1', refit= True, verbose=1)

grid.fit(X_train, y_train)

grid.best_params_
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
y_prob = grid.predict_proba(X_test)



fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
result = pd.DataFrame({'Test':y_test, 'Prediction':y_pred, 'Probability': y_prob[:,1]})
result.to_csv('Result.csv')