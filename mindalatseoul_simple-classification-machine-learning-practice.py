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
import warnings 
warnings.filterwarnings(action='ignore')

import pandas as pd 
import numpy as np
# 지도 학습 : 훈련 데이터로부터 하나의 함수를 유추해내기 위한 기계 학습의 한 방법 
# 분류 예측 : 주어진 입력 벡터가 어떤 종류의 값인지 표식하는 것을 분류라고 함

# 이해하기 쉬운 분류 알고리즘 4 : 
# K-Nearest Neighbors
# Decision Tree/Random Forest 
# Support Vector Machine(SVM)
# Neural Network 

# 알고리즘 import  
from sklearn.neighbors import KNeighborsClassifier # KNN 모델
from sklearn.tree import DecisionTreeClassifier # 의사결정나무 모델
from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트 모델
from sklearn.svm import SVC # 서포트백터머신

# 모델을 테스트하기 위해 필요한 iris 데이터 불러오기
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris['data'],columns=iris['feature_names'])
df['target']=iris['target']
df.head()
# 데이터셋을 test와 valid셋으로 분리 
from sklearn.model_selection import train_test_split
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33)
# 모델별로 데스트 해보기
from sklearn.model_selection import GridSearchCV
param_grid = {
    'KNeighborsClassifier':{
        # n_neighbors, n개의 근접 이웃을 기준으로 판단할지 결정 
        'n_neighbors':[i for i in np.arange(5,20)],
        # 근접 이웃에 따라서 동일하게 고려할지, 거리에 따라서 가중치를 고려할지를 결정 
        'weights':['uniform','distance'],
        # 데이터간 거리를 측정하는 방법을 결정, kdtree를 좀 더 최적화한 것이 balltree
        'algorithm':['auto','ball_tree','kd_tree']
    }
}
    
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
model, model_name = KNeighborsClassifier(),'KNeighborsClassifier'
gcv = GridSearchCV(model,param_grid=param_grid[model_name],scoring='accuracy')
gcv.fit(X_train,Y_train)
# print(gcv.best_params_)
# print(gcv.best_score_)
model = gcv.best_estimator_
result = model.predict(X_test)
accuracy_score(Y_test,result)

# 정확도(accuracy) : 예측이 얼마나 정확한가
# 정밀도(precision) : 예측한 것 중에서 정답의 비율은? 
# 재현율(recall) : 찾아야할 것 중에서 실제로 찾은 비율은? 
# f1 스코어 : 정밀도와 재현율의 평균 

# 번호 : [  1,    2,    3,    4,    5,    6  ]
# 정답 : [음치,음치,음치,음치,정상,정상] 
# 예측 : [음치,음치,정상,정상,정상,정상]

# 정확도 : 예측이 맞은 비율은?
#         1,2,5,6 번 맞추고 3,4번은 틀렸다. 6명중 4명 맞췄으므로 4/6 = 2/3 = 0.66 
#정밀도 : 음치라고 예측한 사람들 중에 진짜 음치가 얼마나 있는가?
#          내가 음치라고 예측한 1,2번 이 둘다 음치가 맞았다. 2/2 = 1.00
#재현율 : 전체 음치 중에서 내가 맞춘 음치의 비율은?
#          원래 음치가 4명 있는데 나는 그중에서 2명을 맞췄다. 2/4 = 0.5
#F1 Score : 정밀도와 재현율의 평균 
#            2 * 정밀도 * 재현율 /(정밀도+재현율) = 2 * 1.00 * 0.5 / (1.00 + 0.5) = 0.66


# sklearn 을 이용하면 전부 계산해준다.
print('accuracy', accuracy_score(Y_test,result) )
print('precision', precision_score(Y_test,result,average=None) )
print('recall', recall_score(Y_test,result,average=None) )
print('f1', f1_score(Y_test,result,average=None) )
# 모델별로 데스트 해보기
from sklearn.model_selection import GridSearchCV
param_grid = {
    'DecisionTreeClassifier':{
        # 트리를 만들 때 불순도 impurity가 가장 낮은 방향으로 트리를 만들어야함
        'criterion':['gini', 'entropy'],
        'splitter':['best','random'],
        'max_depth':[i for i in np.arange(1,5,1)],
        # 노트를 분할하기 위한 최소한의 샘플 수 
        'min_samples_split':[i for i in np.arange(5,11,1)],
        # 리프 노드가 되기 위한 최소한의 샘플 수
        'min_samples_leaf':[i for i in np.arange(5,11,1)]
        
    }
}
    
from sklearn.metrics import accuracy_score
model, model_name = DecisionTreeClassifier(),'DecisionTreeClassifier'
gcv = GridSearchCV(model,param_grid=param_grid[model_name],scoring='accuracy')
gcv.fit(X_train,Y_train)
print(gcv.best_params_)
print(gcv.best_score_)
model = gcv.best_estimator_
result = model.predict(X_test)
accuracy_score(Y_test,result)

# sklearn 을 이용하면 전부 계산해준다.
print('accuracy', accuracy_score(Y_test,result) )
print('precision', precision_score(Y_test,result,average=None) )
print('recall', recall_score(Y_test,result,average=None) )
print('f1', f1_score(Y_test,result,average=None) )
# 모델별로 데스트 해보기
from sklearn.model_selection import GridSearchCV
param_grid = {
    'RandomForestClassifier':{
        'criterion':['gini', 'entropy'],
        'max_depth':[i for i in np.arange(1,5,1)],
        'min_samples_split':[i for i in np.arange(5,11,1)]
#         'min_samples_leaf':[i for i in np.arange(5,11,1)]
        
    }
}
    
from sklearn.metrics import accuracy_score
model, model_name = RandomForestClassifier(),'RandomForestClassifier'
gcv = GridSearchCV(model,param_grid=param_grid[model_name],scoring='accuracy')
gcv.fit(X_train,Y_train)
print(gcv.best_params_)
print(gcv.best_score_)
model = gcv.best_estimator_
result = model.predict(X_test)
accuracy_score(Y_test,result)

# sklearn 을 이용하면 전부 계산해준다.
print('accuracy', accuracy_score(Y_test,result) )
print('precision', precision_score(Y_test,result,average=None) )
print('recall', recall_score(Y_test,result,average=None) )
print('f1', f1_score(Y_test,result,average=None) )
# 모델별로 데스트 해보기
from sklearn.model_selection import GridSearchCV
param_grid = {
    'SVC':{
        'kernel':['linear', 'poly','rbf'],
        'degree':[i for i in np.arange(5,11,1)],
        'gamma':['scale','auto']
    }
}
    
from sklearn.metrics import accuracy_score
model, model_name = SVC(),'SVC'
gcv = GridSearchCV(model,param_grid=param_grid[model_name],scoring='accuracy')
gcv.fit(X_train,Y_train)
print(gcv.best_params_)
print(gcv.best_score_)
model = gcv.best_estimator_
result = model.predict(X_test)
accuracy_score(Y_test,result)

# sklearn 을 이용하면 전부 계산해준다.
print('accuracy', accuracy_score(Y_test,result) )
print('precision', precision_score(Y_test,result,average=None) )
print('recall', recall_score(Y_test,result,average=None) )
print('f1', f1_score(Y_test,result,average=None) )
# 실험용 데이터 셋 정제 
# 타이타닉 데이터셋 
df = pd.read_csv('/kaggle/input/titanic/train.csv')
# 성별, 승선항구를 숫자로 변경
df['Sex'] = df['Sex'].astype('category').cat.codes
df['Embarked'] = df['Embarked'].astype('category').cat.codes
# 불필요한 컬럼은 drop 
df = df.drop(columns=['Name','Ticket','Cabin']).copy() 
# 빈 값은 0으로 대체 
df = df.fillna(0).copy() 
titanic = df.copy() 
# 버섯 분류 데이터셋
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df = df.astype('category').apply(lambda x: x.cat.codes).copy() 
df = df.fillna(0).copy()
df.head()
mushroom = df.copy() 
# IBM HR 데이터 셋 
df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
for col in df.columns:
    if df[col].dtype.name == 'object':
        df[col]=df[col].astype('category').cat.codes
df.head()
ibmhr = df.copy() 
# 심장병 사망 사례 데이터셋 
df= pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df = df.fillna(0).copy()
df.head()
heart = df.copy() 
# 머신러닝 모델링 - 타이타닉 데이터 활용
# 데이터셋을 test와 valid셋으로 분리 
from sklearn.model_selection import train_test_split
# 모델별로 데스트 해보기
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
X = titanic.drop(columns=['Survived'])
y = titanic['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33)

param_grid = {
    'KNeighborsClassifier':{
        # n_neighbors, n개의 근접 이웃을 기준으로 판단할지 결정 
        'n_neighbors':[i for i in np.arange(1,20)],
        # 근접 이웃에 따라서 동일하게 고려할지, 거리에 따라서 가중치를 고려할지를 결정 
        'weights':['uniform','distance'],
        # 데이터간 거리를 측정하는 방법을 결정, kdtree를 좀 더 최적화한 것이 balltree
        'algorithm':['auto','ball_tree','kd_tree']
    },
    'DecisionTreeClassifier':{
        # 트리를 만들 때 불순도 impurity가 가장 낮은 방향으로 트리를 만들어야함
        'criterion':['gini', 'entropy'],
        'splitter':['best','random'],
        'max_depth':[i for i in np.arange(1,5,1)],
        # 노트를 분할하기 위한 최소한의 샘플 수 
        'min_samples_split':[i for i in np.arange(5,11,1)]
        # 리프 노드가 되기 위한 최소한의 샘플 수
#         'min_samples_leaf':[i for i in np.arange(5,11,1)]
    },
    'RandomForestClassifier':{
        'criterion':['gini', 'entropy'],
        'max_depth':[i for i in np.arange(1,5,1,)],
        'min_samples_split':[i for i in np.arange(5,11,1)]
#         'min_samples_leaf':[i for i in np.arange(5,11,1)]  
    },
    'SVC':{
        'kernel':['linear','rbf'],
#         'degree':[i for i in np.arange(5,11,2)]
#         'gamma':['scale','auto']
    }
}
    

models = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),SVC()] 
model_names = ['KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier','SVC']
    
for model, model_name in zip(models,model_names):
    print()
    print('======='+model_name+'========')
    gcv = GridSearchCV(model,param_grid=param_grid[model_name],scoring='accuracy')
    gcv.fit(X_train,Y_train)
    print('train set의 최고 예측 정확도 : {}'.format(gcv.best_score_))
    print('train set 예측 정확도가 가장 높은 경우의 파라미터 : {}'.format(gcv.best_params_))
    model = gcv.best_estimator_
    result = model.predict(X_test)

    # sklearn 을 이용하면 전부 계산해준다.
    print(model_name,'test set의 예측 정확도 accuracy', accuracy_score(Y_test,result) )
    print(model_name,'test set의 예측 정확도 precision', precision_score(Y_test,result,average=None) )
    print(model_name,'test set의 예측 정확도 recall', recall_score(Y_test,result,average=None) )
    print(model_name,'test set의 예측 정확도 f1', f1_score(Y_test,result,average=None) )
# 머신러닝 모델링 - 타이타닉 데이터 활용
# 데이터셋을 test와 valid셋으로 분리 
from sklearn.model_selection import train_test_split
# 모델별로 데스트 해보기
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
X = mushroom.drop(columns=['class'])
y = mushroom['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33)

param_grid = {
    'KNeighborsClassifier':{
        # n_neighbors, n개의 근접 이웃을 기준으로 판단할지 결정 
        'n_neighbors':[i for i in np.arange(1,20)],
        # 근접 이웃에 따라서 동일하게 고려할지, 거리에 따라서 가중치를 고려할지를 결정 
        'weights':['uniform','distance'],
        # 데이터간 거리를 측정하는 방법을 결정, kdtree를 좀 더 최적화한 것이 balltree
        'algorithm':['auto','ball_tree','kd_tree']
    },
    'DecisionTreeClassifier':{
        # 트리를 만들 때 불순도 impurity가 가장 낮은 방향으로 트리를 만들어야함
        'criterion':['gini', 'entropy'],
        'splitter':['best','random'],
        'max_depth':[i for i in np.arange(1,5,1)],
        # 노트를 분할하기 위한 최소한의 샘플 수 
        'min_samples_split':[i for i in np.arange(5,11,1)]
        # 리프 노드가 되기 위한 최소한의 샘플 수
#         'min_samples_leaf':[i for i in np.arange(5,11,1)]
    },
    'RandomForestClassifier':{
        'criterion':['gini', 'entropy'],
        'max_depth':[i for i in np.arange(1,5,1,)],
        'min_samples_split':[i for i in np.arange(5,11,1)]
#         'min_samples_leaf':[i for i in np.arange(5,11,1)]  
    },
    'SVC':{
        'kernel':['linear','rbf'],
#         'degree':[i for i in np.arange(5,11,2)]
#         'gamma':['scale','auto']
    }
}
    

models = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),SVC()] 
model_names = ['KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier','SVC']
    
for model, model_name in zip(models,model_names):
    print()
    print('======='+model_name+'========')
    gcv = GridSearchCV(model,param_grid=param_grid[model_name],scoring='accuracy')
    gcv.fit(X_train,Y_train)
    print('train set의 최고 예측 정확도 : {}'.format(gcv.best_score_))
    print('train set 예측 정확도가 가장 높은 경우의 파라미터 : {}'.format(gcv.best_params_))
    model = gcv.best_estimator_
    result = model.predict(X_test)

    # sklearn 을 이용하면 전부 계산해준다.
    print(model_name,'test set의 예측 정확도 accuracy', accuracy_score(Y_test,result) )
    print(model_name,'test set의 예측 정확도 precision', precision_score(Y_test,result,average=None) )
    print(model_name,'test set의 예측 정확도 recall', recall_score(Y_test,result,average=None) )
    print(model_name,'test set의 예측 정확도 f1', f1_score(Y_test,result,average=None) )
# 머신러닝 모델링 - 타이타닉 데이터 활용
# 데이터셋을 test와 valid셋으로 분리 
from sklearn.model_selection import train_test_split
# 모델별로 데스트 해보기
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
X = ibmhr.drop(columns=['Attrition'])
y = ibmhr['Attrition']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33)

param_grid = {
    'KNeighborsClassifier':{
        # n_neighbors, n개의 근접 이웃을 기준으로 판단할지 결정 
        'n_neighbors':[i for i in np.arange(1,20)],
        # 근접 이웃에 따라서 동일하게 고려할지, 거리에 따라서 가중치를 고려할지를 결정 
        'weights':['uniform','distance'],
        # 데이터간 거리를 측정하는 방법을 결정, kdtree를 좀 더 최적화한 것이 balltree
        'algorithm':['auto','ball_tree','kd_tree']
    },
    'DecisionTreeClassifier':{
        # 트리를 만들 때 불순도 impurity가 가장 낮은 방향으로 트리를 만들어야함
        'criterion':['gini', 'entropy'],
        'splitter':['best','random'],
        'max_depth':[i for i in np.arange(1,5,1)],
        # 노트를 분할하기 위한 최소한의 샘플 수 
        'min_samples_split':[i for i in np.arange(5,11,1)]
        # 리프 노드가 되기 위한 최소한의 샘플 수
#         'min_samples_leaf':[i for i in np.arange(5,11,1)]
    },
    'RandomForestClassifier':{
        'criterion':['gini', 'entropy'],
        'max_depth':[i for i in np.arange(1,5,1,)],
        'min_samples_split':[i for i in np.arange(5,11,1)]
#         'min_samples_leaf':[i for i in np.arange(5,11,1)]  
    },
    'SVC':{
        'kernel':['linear','rbf'],
#         'degree':[i for i in np.arange(5,11,2)]
#         'gamma':['scale','auto']
    }
}
    

models = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),SVC()] 
model_names = ['KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier','SVC']
    
for model, model_name in zip(models,model_names):
    print()
    print('======='+model_name+'========')
    gcv = GridSearchCV(model,param_grid=param_grid[model_name],scoring='accuracy')
    gcv.fit(X_train,Y_train)
    print('train set의 최고 예측 정확도 : {}'.format(gcv.best_score_))
    print('train set 예측 정확도가 가장 높은 경우의 파라미터 : {}'.format(gcv.best_params_))
    model = gcv.best_estimator_
    result = model.predict(X_test)
    print(model_name,'test set의 예측 정확도 : {}'.format(accuracy_score(Y_test,result)))
    
    # sklearn 을 이용하면 전부 계산해준다.
    print(model_name,'test set의 예측 정확도 accuracy', accuracy_score(Y_test,result) )
    print(model_name,'test set의 예측 정확도 precision', precision_score(Y_test,result,average=None) )
    print(model_name,'test set의 예측 정확도 recall', recall_score(Y_test,result,average=None) )
    print(model_name,'test set의 예측 정확도 f1', f1_score(Y_test,result,average=None) )
