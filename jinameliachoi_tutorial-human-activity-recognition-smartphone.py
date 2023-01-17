# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# features.txt 파일에는 feature 이름 index와 feature명이 공백으로 분리되어 있음. 이를 dataframe으로 load
feature_name_df = pd.read_csv('../input/human-activity-recognition-using-smart-phone/UCI HAR Dataset/features.txt',
                             sep = '\s+',
                             header = None,
                             names = ['column_index', 'column_name'])

# feature명 index를 제거하고, feature명만 list 객체로 생성한 뒤 sample로 10개만 추출
feature_name = feature_name_df.iloc[:, 1].values.tolist()
print('전체 feature명에서 10개만 추출: ', feature_name[:10])
feature_dup_df =feature_name_df.groupby('column_name').count()
print(feature_dup_df[feature_dup_df['column_index'] > 1].count())
feature_dup_df[feature_dup_df['column_index'] > 1].head()
# 총 42개의 feature명이 중복되어 있음
# 이 중복된 feature명에 대해서는 원본 피처명에 숫자를 추가로 부여해 새로운 feature명을 가지는 dataframe 생성

def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data = old_feature_name_df.groupby('column_name').cumcount(),
                                 columns = ['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(),
                                  feature_dup_df,
                                  how = 'outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x: x[0] + '_' + str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df
# train과 test directory에 있는 data file을 각각 train/test dataframe에 할당
# dataframe 생성하는 간단한 함수 생성

def get_human_dataset():
    
    # 각 데이터 파일은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep로 할당
    feature_name_df = pd.read_csv('../input/human-activity-recognition-using-smart-phone/UCI HAR Dataset/features.txt',
                                 sep = '\s+',
                                 header = None,
                                 names = ['column_index', 'column_name'])
    
    # 중복된 feature명을 수정하는 get_new_feature_name_df()를 이용, 신규 feature명 dataframe 생성
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    # dataframe에서 feature명을 column으로 부여하기 위해 list 객체로 반환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    
    # train feature dataset와 test feature dataset을 dataframe으로 load. column명은 feature_name 적용
    X_train = pd.read_csv('../input/human-activity-recognition-using-smart-phone/UCI HAR Dataset/train/X_train.txt',
                         sep = '\s+',
                         names = feature_name)
    X_test = pd.read_csv('../input/human-activity-recognition-using-smart-phone/UCI HAR Dataset/test/X_test.txt',
                        sep = '\s+',
                        names = feature_name)
    
    # column명을 action으로 부여
    y_train = pd.read_csv('../input/human-activity-recognition-using-smart-phone/UCI HAR Dataset/train/y_train.txt',
                         sep = '\s+',
                         header = None,
                         names = ['action'])
    y_test = pd.read_csv('../input/human-activity-recognition-using-smart-phone/UCI HAR Dataset/test/y_test.txt',
                        sep = '\s+',
                        header = None,
                        names = ['action'])
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()
print('## train feature dataset info()')
print(X_train.info())
X_train.head(3)
print(y_train['action'].value_counts())
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 예제 반복 시마다 동일한 예측 결과 도출하기 위해 random_state 값 부여
dt = DecisionTreeClassifier(random_state=156)
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('결정 트리 예측 정확도: {0:.4f}'.format(accuracy))

# DecisionTreeClassifier의 hyper-parameter 추출
print('DecisionTreeClassifier 기본 hyper-parameter: \n', dt.get_params())
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth' : [6, 8, 10, 12, 16, 20, 24]
}

grid_cv = GridSearchCV(dt,
                      param_grid = params,
                      scoring = 'accuracy',
                      cv = 5, 
                      verbose = 1) # verbose: log 출력의 level 조정 (숫자가 클수록 많은 log 출력)
grid_cv.fit(X_train, y_train)
print('GridSearchCV 최고 평균 정확도 수치: {0:.4f}'.format(grid_cv.best_score_))
print('GridSearchCV 최적 하이퍼 파라미터: ', grid_cv.best_params_)
# GridSearchCV 객체의 cv_results_ 속성 dataframe 변환
cv_results_df = pd.DataFrame(grid_cv.cv_results_)

# max_depth parameter 값과 그때의 train/test datsaset의 정확도 수치 추출
cv_results_df[['param_max_depth', 'mean_test_score']]
# mean_test_score는 cv=5에서 test dataset의 정확도 평균 수치
# 현재 max_depth params 값들에서는 85.48%를 능가하는 정확도가 나오지 않음
# max_depth의 변화에 따른 값 측정을 확인해보자

max_depths = [6, 8, 10, 12, 16, 20, 24]
for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth = depth,
                               random_state = 156)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print('max_depth = {0} 정확도: {1:.4f}'.format(depth, accuracy))
params = {
    'max_depth' : [8, 12, 16, 20],
    'min_samples_split': [16, 24],
}

grid_cv = GridSearchCV(dt,
                      param_grid = params,
                      scoring = 'accuracy', 
                      cv = 5,
                      verbose = 1)
grid_cv.fit(X_train, y_train)
print('GridSearchCV 최고 평균 정확도 수치: {0:.4f}'.format(grid_cv.best_score_))
print('GridSearchCV 최적 하이퍼 파라미터: ', grid_cv.best_params_)
best_dt = grid_cv.best_estimator_
pred1 = best_dt.predict(X_test)
accuracy = accuracy_score(y_test, pred1)
print('결정 트리 예측 정확도: {0:.4f}'.format(accuracy))
# 중요도가 높은 순서대로 feature_importance_ 확인하기
import seaborn as sns

feature_importance_values = best_dt.feature_importances_
# top 중요도로 정렬을 쉽게 하고, 막대그래프로 쉽게 표한하기 위해 series 변환
feature_importances = pd.Series(feature_importance_values,
                              index = X_train.columns)
# 중요도값 순으로 series를 정렬
feature_top20 = feature_importances.sort_values(ascending = False)[:20]
plt.figure(figsize=(8, 6))
plt.title('Feature Importance Top 20')
sns.barplot(x = feature_top20,
           y = feature_top20.index)
plt.show()