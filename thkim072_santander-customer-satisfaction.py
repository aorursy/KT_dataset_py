import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn 라이브러리 세팅
plt.style.use('seaborn')   # matplot 기본 그림 말고 seaborn 그림 스타일 사용
sns.set(font_scale=2.5)    # 폰트 사이즈 2.5로 고정

# null 데이터를 시각화하여 보여주는 라이브러리
import missingno as msno   

# 오류 무시하는 코드 
import warnings
warnings.filterwarnings('ignore')

# matplot 라이브러리 사용해 시각화한 뒤 show했을 때 새로운 창이 아닌 노트북에서 바로 확인 가능하도록
%matplotlib inline
train = pd.read_csv('../input/santander-customer-satisfaction/train.csv')
test = pd.read_csv('../input/santander-customer-satisfaction/test.csv')
train.head()
test.head()
print(train.shape)
print(test.shape)
train.info()

# dtypes를 통해 모든 피처가 숫자형임을 알 수 있음
# null값 없음
test.info()
train.describe()
# var3열의 min -9999는 NaN이나 특정 예외 값을 -9999로 변환한 것 같음
test.describe()
print(train.var3.value_counts()[:10])
# -999999값이 116개 존재
# var3은 숫자형이고, 다른 값에 비해 편차가 심하므로 가장 값이 많은 2로 변환
train['var3'].replace(-999999, 2, inplace=True)
test['var3'].replace(-999999, 2, inplace=True)
train.drop('ID',axis=1 , inplace=True)
test.drop('ID',axis=1 , inplace=True)
# 이진 분류인 경우 타겟값의 분류가 불균형한지 아닌지 확인해야 함
print(train['TARGET'].value_counts())  
# 불만족인 데이터 건수
unsatisfied= train[train['TARGET'] == 1].TARGET.count()
print(unsatisfied)
# 전체 데이터 건수
total= train.TARGET.count()
print(total)
# 불만족 데이터와 전체 데이터 건수의 비율 확인
print(unsatisfied / total)
# 불만족 비율 4%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
X = train.drop('TARGET', axis=1)  # TARGET을 제외한 피처들
Y = train['TARGET']   
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=11)
train_count = y_train.count()
print(y_train.value_counts()/train_count)
# 타겟값 불만족이 원본 데이터와 유사하게 4%유지
val_count = y_val.count()
print(y_val.value_counts()/val_count)
# 타겟값 불만족이 원본 데이터와 유사하게 4%유지
dt = DecisionTreeClassifier(random_state=11)

# 학습
dt.fit(X_train , y_train)

# 예측
dt_pred = dt.predict(X_val)

# 평가
dt_roc = roc_auc_score(y_val, dt_pred)
print('ROC AUC: {0:.4f}'.format(dt_roc))
# 하이퍼 파라미터 설정
parameters = {'max_depth':[2,3,5,10],'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]} 

# 하이퍼 파라미터를 5개의 train, val로 나누어 테스트 수행 설정
grid_dt = GridSearchCV(dt, param_grid = parameters, scoring = 'accuracy', cv=5, verbose=1 , refit = True)  #  verbose: 얼마나 자세히 정보를 표시할 것인가 0,1,2로 나눠짐

# 튜닝된 하이퍼 파라미터로 학습
grid_dt.fit(X_train, y_train)

# 최고 성능을 낸 하이퍼 파라미터 값과 그때의 평가값 저장
print('GridSearchCV 최적 하이퍼 파라미터:', grid_dt.best_params_)       # 최적 하이퍼 파라미터
print('GridSearchCV 최고 정확도:{0:.4f}'.format(grid_dt.best_score_))  # 최적 하이퍼 파라미터일 때 정확도
# refit = True로 최적 하이퍼 파라미터 미리 학습하여 best_estimator_로 저장됨(별도로 fit할 필요없음)
dt1= grid_dt.best_estimator_   

# 재예측
dt1_pred = dt1.predict(X_val)   

# 재평가
dt1_roc = roc_auc_score(y_val , dt1_pred)
print('ROC AUC:{0:.4f}'.format(dt1_roc))
rf = RandomForestClassifier(random_state=11)

# 학습
rf.fit(X_train , y_train)

# 예측
rf_pred = rf.predict(X_val)

# 평가
rf_roc = roc_auc_score(y_val ,rf_pred)
print('AUC_ROC: {0:.4f}'.format(rf_roc))
# 하이퍼 파라미터 설정
parameters = {'n_estimators':[10], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [8, 12, 18 ],'min_samples_split' : [8, 16, 20]}

# 하이퍼 파라미터를 2개의 train, val로 나누어 테스트 수행 설정
grid_rf = GridSearchCV(rf, param_grid = parameters , cv=2)     # 이번에는 refit 안해봄

# 튜닝된 하이퍼 파라미터로 학습
grid_rf.fit(X_train, y_train)

# 최고 성능을 낸 하이퍼 파라미터 값과 그때의 평가값 저장
print('AUC_ROC:\n', grid_rf.best_params_)              # 최적 하이퍼 파라미터
print('AUC_ROC: {0:.4f}'.format(grid_rf.best_score_))  # 최적 하이퍼 파라미터일 때 정확도
# 최적 하이퍼 파라미터 적용
rf1 = RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=8, min_samples_split=8, random_state=0)

# 재학습
rf1.fit(X_train , y_train)    # refit 안했으므로 fit도 수행

# 재예측
rf1_pred = rf1.predict(X_val)

# 재평가
print(roc_auc_score(y_val , rf1_pred))
xgb = XGBClassifier(n_estimators=500, random_state=156)

# 평가 데이터 세트는 앞에서 분리한 테스트 데이터 세트 이용 -> 1이 4%밖에 없어서 테스트 세트 이용해 검증  
evals = [(X_train, y_train), (X_val, y_val)]

# 학습
xgb.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=0)
                                                       
# 예측 및 평가
xgb_roc_score = roc_auc_score(y_val, xgb.predict_proba(X_val)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))
# 수행시간 조절을 위해 100으로 줄임
xgb = XGBClassifier(n_estimators=100)

# 하이퍼 파라미터 설정
parameters = {'max_depth':[5, 7] , 'min_child_weight':[1,3] ,'colsample_bytree':[0.5, 0.75] }    # 칼럼을 샘플링해서 적용(칼럼이 많으므로 조절)

# 평가 데이터 세트는 앞에서 분리한 테스트 데이터 세트 이용 -> 1이 4%밖에 없어서 테스트 세트 이용해 검증  
evals = [(X_train, y_train), (X_val, y_val)]

# 하이퍼 파라미터의 수행속도를 향상시키기 위해 cv설정 안함
grid_xgb = GridSearchCV(xgb, param_grid=parameters)

# 튜닝된 하이퍼 파라미터로 학습
grid_xgb.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc", eval_set=evals, verbose =0)

# 예측 및 평가
xgb_roc_score = roc_auc_score(y_val, grid_xgb.predict_proba(X_val)[:,1], average='macro')
print('GridSearchCV 최적 파라미터:',grid_xgb.best_params_) 
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))
# n_estimators는 1000으로 증가시키고, learning_rate=0.02로 감소, reg_alpha=0.03으로 추가
xgb1 = XGBClassifier(n_estimators=1000, random_state=156, learning_rate=0.02, max_depth=5,min_child_weight=3, colsample_bytree=0.5, reg_alpha=0.03)

# 재학습
xgb1.fit(X_train, y_train, early_stopping_rounds=200, eval_metric="auc",eval_set=[(X_train, y_train), (X_val, y_val)],  verbose =0)

# 재예측 및 재평가                                                                                                                        
xgb1_roc_score = roc_auc_score(y_val, xgb1.predict_proba(X_val)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(xgb1_roc_score))
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(1,1,figsize=(10,8))
plot_importance(xgb1, ax=ax , max_num_features=20,height=0.4)

# 칼럼이름이 나오는 이유는 numpy가 아닌 dataframe으로 했기 때문
lgbm = LGBMClassifier(n_estimators=500)

evals = [(X_val, y_val)]

# 학습
lgbm.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=0)

# 예측 및 평가
lgbm_roc_score = roc_auc_score(y_val, lgbm.predict_proba(X_val)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
from sklearn.model_selection import GridSearchCV

# 하이퍼 파라미터 테스트의 수행 속도를 향상시키기 위해 n_estimators를 100으로 감소
lgbm = LGBMClassifier(n_estimators=200)

parameters = {'num_leaves': [32, 64 ],'max_depth':[128, 160],'min_child_samples':[60, 100],'subsample':[0.8, 1]}


# 수행속도위해 cv저장 안함
grid_lgbm = GridSearchCV(lgbm, param_grid=parameters)

# 학습
grid_lgbm.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc", eval_set=[(X_train, y_train), (X_val, y_val)],verbose=0)

# 예측 및 평가
lgbm_roc_score = roc_auc_score(y_val,grid_lgbm.predict_proba(X_val)[:,1],average='macro')

print('GridSearchCV 최적 파라미터:', grid_lgbm.best_params_)
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
lgbm1 = LGBMClassifier(n_estimators=1000, num_leaves=32, sumbsample=0.8, min_child_samples=100,max_depth=128)

evals = [(X_val, y_val)]

lgbm1.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=0)

lgbm1_roc_score = roc_auc_score(y_val, lgbm1.predict_proba(X_val)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(lgbm1_roc_score))
submission = pd.read_csv('../input/santander-customer-satisfaction/sample_submission.csv')
prediction = xgb1.predict(test)  # 실제 예측
submission['TARGET'] = prediction  
submission.to_csv('submission.csv', index = False)  # 캐글 커널 서버에 csv파일 저장
