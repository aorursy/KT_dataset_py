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
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')



dt_clf = DecisionTreeClassifier(random_state=156)



iris_data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, 

                                                    test_size=0.2, random_state=11)



dt_clf.fit(X_train, y_train)
from sklearn.tree import export_graphviz



export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, 

                feature_names= iris_data.feature_names, impurity=True, filled = True)
import graphviz

with open("tree.dot") as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)
import seaborn as sns

import numpy as np

%matplotlib inline





print("Feature importances:\n{0}".format(np.round(dt_clf.feature_importances_, 3)))



for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):

    print('{0}: {1:.3f}'.format(name, value))

    

sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt



plt.title("3 Class values with 2 Features Sample data creation")



X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2,

                                           n_classes=3, n_clusters_per_class=1, random_state=0)

plt.scatter(X_features[:, 0], X_features[:, 1], marker='o', c=y_labels, s=25, edgecolor='k')
import numpy as np



# Classifier의 Decision Boundary를 시각화 하는 함수

def visualize_boundary(model, X, y):

    fig,ax = plt.subplots()

    

    # 학습 데이타 scatter plot으로 나타내기

    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',

               clim=(y.min(), y.max()), zorder=3)

    ax.axis('tight')

    ax.axis('off')

    xlim_start , xlim_end = ax.get_xlim()

    ylim_start , ylim_end = ax.get_ylim()

    

    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 

    model.fit(X, y)

    # meshgrid 형태인 모든 좌표값으로 예측 수행. 

    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    

    # contourf() 를 이용하여 class boundary 를 visualization 수행. 

    n_classes = len(np.unique(y))

    contours = ax.contourf(xx, yy, Z, alpha=0.3,

                           levels=np.arange(n_classes + 1) - 0.5,

                           cmap='rainbow', clim=(y.min(), y.max()),

                           zorder=1)
from sklearn.tree import DecisionTreeClassifier



# 특정한 트리 생성 제약없는 결정 트리의 Decsion Boundary 시각화.

dt_clf = DecisionTreeClassifier().fit(X_features, y_labels)

visualize_boundary(dt_clf, X_features, y_labels)
# min_samples_leaf=6 으로 트리 생성 조건을 제약한 Decision Boundary 시각화

dt_clf = DecisionTreeClassifier( min_samples_leaf=6).fit(X_features, y_labels)

visualize_boundary(dt_clf, X_features, y_labels)
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# features.txt 파일에는 피처 이름 index와 피처명이 공백으로 분리되어 있음. 이를 DataFrame으로 로드.

feature_name_df = pd.read_csv('../input/human-activity/human_activity/features.txt',sep='\s+',

                        header=None,names=['column_index','column_name'])



# 피처명 index를 제거하고, 피처명만 리스트 객체로 생성한 뒤 샘플로 10개만 추출

feature_name = feature_name_df.iloc[:, 1].values.tolist()

print('전체 피처명에서 10개만 추출:', feature_name[:10])

feature_dup_df = feature_name_df.groupby('column_name').count()

print(feature_dup_df[feature_dup_df['column_index']>1].count())

feature_dup_df[feature_dup_df['column_index']>1].head()
def get_new_feature_name_df(old_feature_name_df):

    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),

                                  columns=['dup_cnt'])

    feature_dup_df = feature_dup_df.reset_index()

    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')

    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) 

                                                                                         if x[1] >0 else x[0] ,  axis=1)

    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)

    return new_feature_name_df
import pandas as pd



def get_human_dataset( ):

    feature_name_df = pd.read_csv('../input/human-activity/human_activity/features.txt',sep='\s+',

                        header=None,names=['column_index','column_name'])



    new_feature_name_df = get_new_feature_name_df(feature_name_df)

    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    X_train = pd.read_csv('../input/human-activity/human_activity/train/X_train.txt',sep='\s+', names=feature_name )

    X_test = pd.read_csv('../input/human-activity/human_activity/test/X_test.txt',sep='\s+', names=feature_name)

    y_train = pd.read_csv('../input/human-activity/human_activity/train/y_train.txt',sep='\s+',header=None,names=['action'])

    y_test = pd.read_csv('../input/human-activity/human_activity/test/y_test.txt',sep='\s+',header=None,names=['action'])



    return X_train, X_test, y_train, y_test





X_train, X_test, y_train, y_test = get_human_dataset()
print('##학습 피처 데이터셋info()')

print(X_train.info())
print(y_train['action'].value_counts())
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



dt_clf = DecisionTreeClassifier(random_state=156)

dt_clf.fit(X_train, y_train)

pred=dt_clf.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print('결정 트리 예측 정확도: {0:.4f}'.format(accuracy))



print('DecisionTreeClassifier 기본 하이퍼 파라미터:\n', dt_clf.get_params())
from sklearn.model_selection import GridSearchCV



params = {

    'max_depth' : [ 6, 8 ,10, 12, 16 ,20, 24]

}



grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring='accuracy', cv=5, verbose=1 )

grid_cv.fit(X_train , y_train)

print('GridSearchCV 최고 평균 정확도 수치:{0:.4f}'.format(grid_cv.best_score_))

print('GridSearchCV 최적 하이퍼 파라미터:', grid_cv.best_params_)





cv_results_df = pd.DataFrame(grid_cv.cv_results_)

cv_results_df[['param_max_depth', 'mean_test_score']]

max_depths = [6,]