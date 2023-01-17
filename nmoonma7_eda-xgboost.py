import pandas as pd

import numpy as np

import xgboost as xgb
train= pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
# 훈련 데이터의 데이터 크기를 확인

train.shape
train.head()
# 변수별 데이터 및 타입 확인

for col in train.columns:

    print('{}\n'.format(train[col].head()))
train.info()
#수치형 변수 살펴보기

num_cols = [col for col in train.columns if train[col].dtype in ['int64','float64']]

train[num_cols].describe()
# 범주형 변수 살펴보기

cat_cols = [col for col in train.columns if train[col].dtype in ['O']]

train[cat_cols].describe()
# 범주형 변수의 고유값 출력

for col in cat_cols:

    uniq = np.unique(train[col].astype(str))

    print('-'*50)

    print('# col {}, n_uniq {}, uniq {}'.format(col,len(uniq), uniq))
import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline

import seaborn as sns
# 변수를 막대 그래프로 시각화하기

skip_cols = ['PassengerId','Name']

for col in train.columns:

    

    # PassengerId와 Name은 의미가 없으므로 생략한다

    if col in skip_cols:

        continue

    

    print('-'* 50)

    print('col: ', col)

    

    f, ax = plt.subplots(figsize=(20,15))

    sns.countplot(x=col, data=train, alpha=0.5)

    plt.show()
np.random.seed(2019)



# 데이터 전처리



# train data와 test data의 통합

df= pd.concat([train,test],sort=False)



# 학습에 이용할 feature 담기

features = []



# 범주형 변수 label encoding

categorical_cols = ['Sex','Cabin','Embarked']



for col in categorical_cols:

    df[col],_= df[col].factorize(na_sentinel=-99)



features += categorical_cols



# Survied test 데이터는 우선 -1로 채운다

df['Survived'].fillna(-1, inplace= True)



# 수치형 변수 전처리

# 결측치를 모두 -99로 대체한다

df.fillna(-99, inplace =True)



features += ['Pclass','Age','SibSp','Parch','Fare']
features
# xgboost 모델 학습

param = {

    'booster': 'gbtree',

    'max_depth': 8,

    'nthread':4,

    'objective':'binary:hinge',

    'silent':1,

    'eval_metric':'error',

    'eta':0.1,

    'min_child_weight':10,

    'colsample_bytree':0.8,

    'colsample_bylevel':0.9,

    'seed':2019,

}
# train과 test를 통합했던 df 에서 다시 분리한다



dtrn= df[:891]

dtest=df[891:]



X_trn= dtrn.as_matrix(columns=features)

Y_trn= dtrn.as_matrix(columns=['Survived'])

dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)
watch_list = [(dtrn,'train')]

model= xgb.train(param,dtrn,num_boost_round=1000, evals=watch_list, early_stopping_rounds=20)
# 테스트 데이터 값 예측

X_test= dtest.as_matrix(columns=features)

dtst=xgb.DMatrix(X_test, feature_names=features)

pred_tst= model.predict(dtst)
psgId=np.array(dtest['PassengerId'].tolist())



result=np.stack((psgId, pred_tst), axis=-1)



result=result.astype(int)



final_result = pd.DataFrame(result, columns=['PassengerId','Survived'])
final_result.to_csv('./submission.csv',index=False)