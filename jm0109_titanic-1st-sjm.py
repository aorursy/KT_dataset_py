# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# 라이브러리 로드
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra # 수학 연산 관련 함수 사용하기 위한 라이브러리 https://numpy.org/
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) # 데이터 프로세싱 관련한 라이브러리, 여기서는 CSV 파일 처리 위해 사용 https://pandas.pydata.org/

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# import data
traindata = pd.read_csv('/kaggle/input/titanic/train.csv')
testdata = pd.read_csv('/kaggle/input/titanic/test.csv')
# 불러온 데이터 구조 확인하기
traindata.head(10) # pandas 함수, XXX.head() 라고 입력하면 위의 5개줄 출력, 숫자 지정하면 지정한만큼 출력
traindata.count() 

### 각 열의 데이터 갯수 확인
# 데이터 분석
traindata.describe()  # 이것도 pandas 함수, XXX.describe() 라고 입력하면 데이터의 개수, 평균 분산 등등 출력함
# encoding string data columns. 문자열로 되어 있는 데이터를 숫자로 변환
from sklearn.preprocessing import LabelEncoder # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.transform
le = LabelEncoder()
traindata['Sex'] = le.fit_transform(traindata['Sex'])
testdata['Sex'] = le.fit_transform(testdata['Sex'])
traindata['Embarked'] = le.fit_transform(traindata['Embarked'].astype(str))
testdata['Embarked'] = le.fit_transform(testdata['Embarked'].astype(str))
### 변환이 잘 되었나 확인

traindata["Sex"].head() 
traindata['Embarked'].head()
### 원래로 되돌리는건 아래 코드가 아닌가?? 에러 남. 
# traindata['Sex'] = le.inverse_transform(traindata['Sex'])
# traindata['Sex'].head()
### 원래 데이터: PassengerId / Survived / Pclass / Name / Sex / Age / SibSp / Parch / Ticket / Fare / Cabin / Embarked

# 데이터 리스트들 중에 쓸만한 것들로만 학습시키자. 의미없는 열은 제거하자
X = traindata[['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked']].values
y = traindata.iloc[:,1].values
X_real_test = testdata[['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked']].values # X 와 X_real_test와 같은거 아닌가?? 차이가 뭐지??
X_real_test # 한번 출력해 봄. 
y
# 데이터 중간 중간에 빠진 값들을 찾아보자
print(traindata.isnull().sum())
# 비어있는 데이터, NA 들을 채워보자
from sklearn.impute import SimpleImputer  # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
imputer = SimpleImputer()
X_transformed = imputer.fit_transform(X)
X_real_test = imputer.fit_transform(X_real_test)
print(X_transformed)
# Splitting the dataset into the Training set and Test set for evaluation
from sklearn.model_selection import train_test_split # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X_transformed,y,test_size=0.25,random_state=1313)
print(X_transformed.shape)
print(X_train.shape)
# 891 * 0.75 = 668
# 데이터의 scale을 맞춰준다.
from sklearn.preprocessing import StandardScaler # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_real_test = sc.fit_transform(X_real_test)
X_train
# SVM classifier with Gaussian RBF kernel
from sklearn.svm import SVC
classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# 원래 코드    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    decision_function_shape='ovr', degree=3, gamma='auto',
    kernel='rbf', max_iter=-1, probability=False, random_state=0,
    shrinking=True, tol=0.001, verbose=False)
classifier.fit(X_train,y_train)
# 검증 데이터로 ML 모델의 성능을 테스트 해보자.
y_pred = classifier.predict(X_test)
# evaluation using confusion matrix
from sklearn.metrics import confusion_matrix # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
cm = confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.model_selection import GridSearchCV # Hyper parameters 를 optimization 해보자 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.ensemble import RandomForestClassifier
Forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=2, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
#print(clf)
scores = ['precision', 'recall']
parameters = {'n_estimators':[10, 50, 100, 500, 1000], 'max_depth':[2, 4, 8, 16]}
for score in scores:
    clf = GridSearchCV(Forest, parameters, n_jobs=4, cv=5,  scoring='%s_macro' % score)
    clf.fit(X_train,y_train)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
#print(clf)
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
       print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
print(clf.best_params_)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
# generating predictions on provided test data
y_pred_test = clf.predict(X_real_test)
pid = testdata[['PassengerId']].values
res = np.expand_dims(y_pred_test,axis=1)
f = np.hstack((pid,res))
df = pd.DataFrame(f, columns = ['PassengerId', 'Survived']) 
df.to_csv('gender_submission.csv', index=False)