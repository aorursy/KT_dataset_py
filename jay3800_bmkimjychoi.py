import pandas as pd

import numpy as np
human = pd.read_csv('../input/human1/human.csv', encoding = 'cp949')

human_new = pd.read_csv('../input/human1/human_new.csv', encoding = 'cp949')
human
# 특정한 1개의 나라 열 삭제

human = human[human['모국'] !=' Holand-Netherlands']

human
# 각 특성 값 숫자화

obj = ['노동 계급', '학력', '혼인 상태', '직업', '관계', '인종','모국','성별']

human[obj] = human[obj].apply(lambda x: x.astype('category').cat.codes)
obj = ['노동 계급', '학력', '혼인 상태', '직업', '관계', '인종','모국']

human_new[obj] = human_new[obj].apply(lambda x: x.astype('category').cat.codes)
# 영문화

human = human.rename(columns={'아이디': 'ID', '나이':'age', '노동 계급':'workclass', 'fnlwgt':'fnlwgt', '학력':'education', 

              '교육 수':'education_num', '혼인 상태':'marital_status', '직업':'occupation',

              '관계':'relationship', '인종':'race', '성별':'sex', '자본 이득':'capital_gain',

              '자본 손실':'capital_loss', '주당 시간':'hours_per_week', '모국':'native_country'})



human_new = human_new.rename(columns={'아이디': 'ID', '나이':'age', '노동 계급':'workclass', 'fnlwgt':'fnlwgt', '학력':'education', 

              '교육 수':'education_num', '혼인 상태':'marital_status', '직업':'occupation',

              '관계':'relationship', '인종':'race', '자본 이득':'capital_gain',

              '자본 손실':'capital_loss', '주당 시간':'hours_per_week', '모국':'native_country'})



human.info()
corr = human.corr()

corr
human[['sex', 'relationship']]
# gender 와의 상관계수 정렬

# 전반적으로 저조한 상관계수

# 가장 큰 상관계수를 지닌 '아침_구매건수' 확인

pd.DataFrame(abs(corr['sex']).sort_values(ascending = False))
# one hot

human = pd.get_dummies(human, columns=['workclass', 'education', 'education_num', 'marital_status',

                                       'occupation', 'relationship', 'race', 'native_country'])
human_new = pd.get_dummies(human_new, columns=['workclass', 'education', 'education_num', 'marital_status',

                                       'occupation', 'relationship', 'race', 'native_country'])
human
human_new
t_final = human_new.copy()

human = human.drop(['ID'], axis=1)

human_new = human_new.drop(['ID'], axis=1)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(human)
x = scaler.transform(human)

human = pd.DataFrame(x, columns=human.columns)
scaler = MinMaxScaler()

scaler.fit(human_new)
x = scaler.transform(human_new)

human_new = pd.DataFrame(x, columns=human_new.columns)
human
# 데이터 분할(Split data)

from sklearn.model_selection import train_test_split



X = human.drop(['sex'], axis=1)

y = human['sex']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,

                     weights='uniform')



knn.fit(X_train, y_train)
print('훈련 정확도: {:.2f}'.format(knn.score(X_train, y_train)))

print('테스트 정확도: {:.2f}'.format(knn.score(X_test, y_test)))
from lightgbm import LGBMClassifier



LGBM = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

               importance_type='split', lambda_l1=0.01, lambda_l2=0.01,

               learning_rate=0.01, max_depth=50, metric='auc',

               min_child_samples=20, min_child_weight=0.001,

               min_data_in_leaf=20, min_split_gain=0.0, n_estimators=300,

               n_jobs=-1, num_boost_round=4000, num_leaves=150,

               objective='binary', random_state=0, reg_alpha=0.1,

               reg_lambda=0.01, silent=True, subsample=1,

               subsample_for_bin=200000, subsample_freq=0)



LGBM.fit(X_train, y_train)
print('훈련 정확도: {:.2f}'.format(LGBM.score(X_train, y_train)))

print('테스트 정확도: {:.2f}'.format(LGBM.score(X_test, y_test)))
#Grid Search on Logistic Regression

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import ShuffleSplit



## C_vals is the alpla value of lasso and ridge regression(as alpha increases the model complexity decreases,)

## remember effective alpha scores are 0<alpha<infinity 

C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]

## Choosing penalties(Lasso(l1) or Ridge(l2))

penalties = ['l1','l2']

## Choose a cross validation strategy. 

cv = ShuffleSplit(n_splits = 10, test_size = .25)



## setting param for param_grid in GridSearchCV. 

param = {'penalty': penalties, 'C': C_vals}



logreg = LogisticRegression(solver='liblinear')

## Calling on GridSearchCV object. 

grid = GridSearchCV(estimator=LogisticRegression(), 

                           param_grid = param,

                           scoring = 'accuracy',

                            n_jobs =-1,

                           cv = cv)

## Fitting the model

grid.fit(X_train, y_train)
print('정확도: {:.2f}'.format(grid.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier(n_estimators=10, random_state=0)



RF.fit(X_train, y_train)
print('정확도: {:.2f}'.format(RF.score(X_test, y_test)))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform



logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,

                              random_state=0)

distributions = dict(C=uniform(loc=0, scale=4),

                     penalty=['l2', 'l1'])

clf = RandomizedSearchCV(logistic, distributions, random_state=0)

search = clf.fit(X_train, y_train)

print('정확도: {:.2f}'.format(clf.score(X_test, y_test)))
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold





params = {'max_features':list(np.arange(1, X.shape[1])), 'bootstrap':[False],

          'n_estimators': [50], 'criterion':['gini','entropy']}

model = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=10,

                           cv=KFold, scoring='roc_auc',n_jobs=-1, verbose=1)



print('MODELING.............................................................................')



model.fit(X, target)

print('========BEST_AUC_SCORE = ', model.best_score_)



model = model.best_estimator

test = model.predict_proba(Y)[:,1]
t_final['SEX'] = LGBM.predict(human_new.values)

# cp_test['gender'] = bc.predict(cp_test.loc[:,'총구매액':'구매브랜드가치'])
# 특정 범위 Drop

drop = t_final.columns[1:121]

t_final = t_final.drop(drop, axis=1)
t_final.to_csv('t_final4.csv', encoding='cp949')