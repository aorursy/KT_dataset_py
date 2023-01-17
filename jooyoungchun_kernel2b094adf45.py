# sklearn으로 bagging 만들기
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
import sklearn
import pandas as pd
import numpy as np
from sklearn import model_selection # cross-validation score를 가져오기 위함
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier # bagging
from sklearn.tree import DecisionTreeClassifier # 의사 결정 나무
from collections import Counter # count
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter("ignore", UserWarning)
filename = '../input/pima-indians-diabetes.data.csv'
# 
dataframe = pd.read_csv(filename, header=None)
dataframe.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class']
dataframe.head()
array = dataframe.values # 손 쉬운 indexing을 위하여 array로 변형
array
X = array[:,0:8].astype(float)  # 0 - 7 column은 독립변수
Y = array[:,8].astype(int) # 마지막 column은 종속변수

print('X:',X[:5])
print('y:',Y[:5])
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)
print('Number of train set:', len(train_x))
print('Number of test set:', len(test_x))
assert len(train_x) == len(train_y)
assert len(test_x) == len(test_y)
# hyperparameters
param_grid = {'n_estimators': [100, 200],
              'max_features': [1.0], 
              'bootstrap_features': [False], # no replacement
              'oob_score': [True], # compute out of bag error
              'n_jobs':[-1], 
              'base_estimator__max_depth': [3, 5]
              }
# 1) 모델 선언
DT = DecisionTreeClassifier()
DT
sklearn.metrics.SCORERS.keys()
# 2) 여러 모델들을 ensemble: bagging
bag_model = BaggingClassifier(base_estimator=DT, random_state=1, max_samples=0.5)

# hyperparameter search
grid_search = GridSearchCV(bag_model, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit( train_x, train_y)
grid_search.best_params_
opt_model = grid_search.best_estimator_
opt_model
# 검증데이터에 대한 f1-score
opt_model.oob_score_
# 4) 예측
test_pred_y = opt_model.predict(test_x)
test_pred_y
# 테스트 데이터에 대한 f1-score
bag_f1 = f1_score(y_true= test_y, y_pred= test_pred_y)
bag_f1
def get_variable_importance(model):
    return np.mean([tree.feature_importances_ for tree in model.estimators_], axis =0)

var_df = pd.Series(get_variable_importance(opt_model), index = dataframe.columns[:-1])

var_df.sort_values(ascending=False)
dataframe.columns
# sklearn으로 random forest 만들기
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
# hyperparameters
param_grid = {'n_estimators': [100, 200],
              'oob_score': [True], # compute out of bag error
              'n_jobs':[-1], 
              'max_depth': [3, 5]
              }
# 1) 모델 선언 & 2) 여러 모델들을 ensemble: randomforest
rf_model = RandomForestClassifier()

# hyperparameter search
grid_search = GridSearchCV(rf_model, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit( train_x, train_y)
grid_search.best_params_
opt_model = grid_search.best_estimator_
opt_model
# 검증데이터에 대한 f1-score
opt_model.oob_score_
# 4) 예측
test_pred_y = opt_model.predict(test_x)
test_pred_y
# 테스트 데이터에 대한 f1-score
rf_f1 = f1_score(y_true= test_y, y_pred= test_pred_y)
rf_f1
opt_model.feature_importances_
var_df = pd.Series(opt_model.feature_importances_, index = dataframe.columns[:-1])
var_df.sort_values(ascending=False)
pd.Series([bag_f1,rf_f1],index =['bag', 'f1'], name = 'f1-score')
