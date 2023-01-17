import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense, Input, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers
traindata = pd.read_csv('../input/creditcardfraud/creditcard.csv')
traindata.head(10)

#traindata= traindata.sample(frac = 0.1,random_state=1)
#샘플로 하이퍼 파라미터 찾기

traindata.describe()
f, (ax1) = plt.subplots(1, figsize=(24,10))

corr = traindata.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)
print(traindata.isnull().sum())
print(traindata.columns)
print('Normal', round(traindata['Class'].value_counts()[0]/len(traindata) * 100,2), '% of the dataset')
print('Frauds', round(traindata['Class'].value_counts()[1]/len(traindata) * 100,2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=traindata, palette=colors)
plt.title('Class Distributions \n (0: Normal || 1: Fraud)', fontsize=14)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
traindata['scaled_V2'] = sc.fit_transform(traindata['V2'].values.reshape(-1,1))
traindata['scaled_V3'] = sc.fit_transform(traindata['V3'].values.reshape(-1,1))
traindata['scaled_V4'] = sc.fit_transform(traindata['V4'].values.reshape(-1,1))
traindata['scaled_V7'] = sc.fit_transform(traindata['V7'].values.reshape(-1,1))
traindata['scaled_V8'] = sc.fit_transform(traindata['V8'].values.reshape(-1,1))
traindata['scaled_V10'] = sc.fit_transform(traindata['V10'].values.reshape(-1,1))
traindata['scaled_V11'] = sc.fit_transform(traindata['V11'].values.reshape(-1,1))
traindata['scaled_V12'] = sc.fit_transform(traindata['V12'].values.reshape(-1,1))
traindata['scaled_V14'] = sc.fit_transform(traindata['V14'].values.reshape(-1,1))
traindata['scaled_V16'] = sc.fit_transform(traindata['V16'].values.reshape(-1,1))
traindata['scaled_V17'] = sc.fit_transform(traindata['V17'].values.reshape(-1,1))

traindata.drop(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'], axis=1, inplace=True)

Class = traindata['Class']

traindata.drop(['Class'], axis=1, inplace=True)
traindata.insert(0, 'Class', Class)
f, (ax1) = plt.subplots(1, figsize=(24,10))

corr = traindata.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)
fraud_traindata = traindata.loc[traindata['Class'] == 1]
normal_traindata = traindata.loc[traindata['Class'] == 0]

X_normal = normal_traindata.drop('Class', axis=1)
Y_normal = normal_traindata['Class']
X_fraud_test = fraud_traindata.drop('Class', axis=1)
Y_fraud_test = fraud_traindata['Class']

from sklearn.model_selection import train_test_split

X_normal_train, X_normal_test, Y_normal_train, Y_normal_test = train_test_split(X_normal, Y_normal, test_size=0.3, random_state=42)
print(X_normal_train.shape)
print(Y_normal_train.shape)
print(X_normal_test.shape)
print(Y_normal_test.shape)
print(X_fraud_test.shape)
print(Y_fraud_test.shape)

X_normal_train = X_normal_train.values
Y_normal_train = Y_normal_train.values
X_normal_test = X_normal_test.values
Y_normal_test = Y_normal_test.values
X_fraud_test = X_fraud_test.values
Y_fraud_test = Y_fraud_test.values
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

clf = OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, max_iter=-1).fit(X_normal_train)
kfold = StratifiedKFold(n_splits=5, random_state=13, shuffle=True)

#svm_params = {'degree': [1, 3, 5], 'gamma': [0.1, 0.5, 1], 'kernel': ['rbf', 'poly', 'linear']}
#최적의 하이퍼 파라미터 {'degree': 1, 'gamma': 0.1, 'kernel': 'rbf'}
svm_params = {'degree': [1], 'gamma': [0.1], 'kernel': ['rbf']}

grid_svm = GridSearchCV(clf, svm_params, scoring="accuracy")
grid_svm.fit(X_normal_train, Y_normal_train)
svm = grid_svm.best_estimator_

print('최적의 하이퍼 파라미터', grid_svm.best_params_)
print('최적 모델의 교차검증 스코어', grid_svm.best_score_)
print('최적 모델', grid_svm.best_estimator_)
y_pred = svm.predict(X_normal_train)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y_normal_train).sum()

from sklearn.metrics import classification_report,accuracy_score

print("{}: {}".format(clf,n_errors))
print("Accuracy Score :")
print(accuracy_score(Y_normal_train,y_pred))
y_pred = svm.predict(X_normal_test)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y_normal_test).sum()

print("{}: {}".format(clf,n_errors))
print("Accuracy Score :")
print(accuracy_score(Y_normal_test,y_pred))
acc_X_normal_test=accuracy_score(Y_normal_test,y_pred)
y_pred = svm.predict(X_fraud_test)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y_fraud_test).sum()

print("{}: {}".format(clf,n_errors))
print("Accuracy Score :")
print(accuracy_score(Y_fraud_test,y_pred))
acc_X_fraud_test=accuracy_score(Y_fraud_test,y_pred)
total_accuracy=acc_X_normal_test+acc_X_fraud_test
print(total_accuracy/2)