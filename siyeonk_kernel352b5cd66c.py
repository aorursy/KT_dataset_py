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
#환경 준비
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#데이터 불러오기
data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
#데이터 구조 확인하기
data.head()
#데이터 정보 확인하기
data.info()
#데이터에 대한 보다 자세한 정보 확인하기
data.describe()
#와인의 품질을 나타내는 값과 개수 확인하기
data['quality'].value_counts()
#품질 분포 확인하기
sns.countplot(x='quality', data=data)
#모든 성분 분포 확인하기
data.hist(bins=50, figsize=(20,15))
plt.show()
#상관관계 확인하기
corr_matrix = data.corr()
corr_matrix['quality'].sort_values(ascending=False)
#상관관계 확인하기
sns.pairplot(data)
#상관관계 히트맵 보기
plt.figure(figsize=(15,10))
heatmap = sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.show()
from pandas.plotting import scatter_matrix

attributes = ['quality','alcohol','citric acid','sulphates','fixed acidity']
scatter_matrix(data[attributes], figsize=(12,8))
#박스플롯 그리기
fig, ax = plt.subplots(4, 3, figsize=(20, 20))
for var, subplot in zip(data.columns, ax.flatten()):
    if var == "quality":
        continue
    else:
        sns.boxplot(x=data['quality'], y=data[var], data=data, ax=subplot)
bins = (2, 6.5, 8)
group_names = ['Bad', 'Good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
data.head()
data['quality'].value_counts()
#문자를 숫자로 맵핑해주기
from sklearn.preprocessing import LabelEncoder
label_quality = LabelEncoder()
data['quality'] = label_quality.fit_transform(data['quality'])
data.head()
data['quality'].value_counts()
sns.countplot(data['quality'])
X = data.drop('quality', axis = 1)
y = data['quality']
print(X)
print(y)
#트레이닝 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#계층적 샘플링
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['quality']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
strat_test_set['quality'].value_counts()/len(strat_test_set)
strat_train_set['quality'].value_counts()/len(strat_train_set)
#표준 스케일링 적용하기
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print(X_train)
print(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_confusion_matrix = confusion_matrix(y_test, lr_predict)
lr_accuracy_score = accuracy_score(y_test, lr_predict)
lr_precision_score = precision_score(y_test, lr_predict)
lr_recall_score = recall_score(y_test, lr_predict)
lr_f1_score = f1_score(y_test, lr_predict)
print(lr_confusion_matrix)
print(lr_accuracy_score)
print(lr_precision_score)
print(lr_recall_score)
print(lr_f1_score)
tp_lr = confusion_matrix(y_test, lr_predict)[0,0]
fp_lr = confusion_matrix(y_test, lr_predict)[0,1]
tn_lr = confusion_matrix(y_test, lr_predict)[1,1]
fn_lr = confusion_matrix(y_test, lr_predict)[1,0]
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_predict = dt.predict(X_test)
dt_confusion_matrix = confusion_matrix(y_test, dt_predict)
dt_accuracy_score = accuracy_score(y_test, dt_predict)
dt_precision_score = precision_score(y_test, dt_predict)
dt_recall_score = recall_score(y_test, dt_predict)
dt_f1_score = f1_score(y_test, dt_predict)
print(dt_confusion_matrix)
print(dt_accuracy_score)
print(dt_precision_score)
print(dt_recall_score)
print(dt_f1_score)
tp_dt = confusion_matrix(y_test, dt_predict)[0,0]
fp_dt = confusion_matrix(y_test, dt_predict)[0,1]
tn_dt = confusion_matrix(y_test, dt_predict)[1,1]
fn_dt = confusion_matrix(y_test, dt_predict)[1,0]
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_predict = rf.predict(X_test)
rf_confusion_matrix = confusion_matrix(y_test, rf_predict)
rf_accuracy_score = accuracy_score(y_test, rf_predict)
rf_precision_score = precision_score(y_test, rf_predict)
rf_recall_score = recall_score(y_test, rf_predict)
rf_f1_score = f1_score(y_test, rf_predict)
print(rf_confusion_matrix)
print(rf_accuracy_score)
print(rf_precision_score)
print(rf_recall_score)
print(rf_f1_score)
tp_rf = confusion_matrix(y_test, rf_predict)[0,0]
fp_rf = confusion_matrix(y_test, rf_predict)[0,1]
tn_rf = confusion_matrix(y_test, rf_predict)[1,1]
fn_rf = confusion_matrix(y_test, rf_predict)[1,0]
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
svc_predict = svc.predict(X_test)
svc_confusion_matrix = confusion_matrix(y_test, svc_predict)
svc_accuracy_score = accuracy_score(y_test, svc_predict)
svc_precision_score = precision_score(y_test, svc_predict)
svc_recall_score = recall_score(y_test, svc_predict)
svc_f1_score = f1_score(y_test, svc_predict)
print(svc_confusion_matrix)
print(svc_accuracy_score)
print(svc_precision_score)
print(svc_recall_score)
print(svc_f1_score)
tp_svc = confusion_matrix(y_test, svc_predict)[0,0]
fp_svc = confusion_matrix(y_test, svc_predict)[0,1]
tn_svc = confusion_matrix(y_test, svc_predict)[1,1]
fn_svc = confusion_matrix(y_test, svc_predict)[1,0]
models = [('Logistic Regression', tp_lr, fp_lr, tn_lr, fn_lr, lr_accuracy_score, lr_precision_score, lr_recall_score, lr_f1_score),
          ('Decision Tree Classifier', tp_dt, fp_dt, tn_dt, fn_dt, dt_accuracy_score, dt_precision_score, dt_recall_score, dt_f1_score),
          ('Random Forest Classifier', tp_rf, fp_rf, tn_rf, fn_rf, rf_accuracy_score, rf_precision_score, rf_recall_score, rf_f1_score),
          ('SVM (Linear)', tp_svc, fp_svc, tn_svc, fn_svc, svc_accuracy_score, svc_precision_score, svc_recall_score, svc_f1_score),
         ]
predict = pd.DataFrame(data = models, columns=['Model', 'True Positive', 'False Positive', 'True Negative', 'False Negative', 'Accuracy', 'Precision', 'Recall','f1_score'])
predict
f, axe = plt.subplots(1,1, figsize=(12,4))

predict.sort_values(by=['Accuracy'], ascending=False, inplace=True)

sns.barplot(x='Accuracy', y='Model', data = predict, palette='Blues_d')
axe.set_xlabel('Accuracy Score', size=16)
axe.set_ylabel('Model', size=16)
axe.set_xticks(np.arange(0, 1.1, 0.1))
plt.show()
f, axe = plt.subplots(1,1, figsize=(12,4))

predict.sort_values(by=['f1_score'], ascending=False, inplace=True)

sns.barplot(x='f1_score', y='Model', data = predict, palette='Greens_d')
axe.set_xlabel('f1_score', size=16)
axe.set_ylabel('Model', size=16)
axe.set_xticks(np.arange(0, 1.1, 0.1))
plt.show()