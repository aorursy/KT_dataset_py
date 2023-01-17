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
import numpy as np



def apk(actual, predicted, k=7, default=0.0):

    # MAP@7 이므로, 최대 7개만 사용한다

    if len(predicted) > k:

        predicted = predicted[:k]



    score = 0.0

    num_hits = 0.0



    for i, p in enumerate(predicted):

        # 점수를 부여하는 조건은 다음과 같다 :

        # 예측값이 정답에 있고 (‘p in actual’)

        # 예측값이 중복이 아니면 (‘p not in predicted[:i]’) 

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i+1.0)



    # 정답값이 공백일 경우, 무조건 0.0점을 반환한다

    if not actual:

        return default



    # 정답의 개수(len(actual))로 average precision을 구한다

    return score / min(len(actual), k)



def mapk(actual, predicted, k=7, default=0.0):

    # list of list인 정답값(actual)과 예측값(predicted)에서 고객별 Average Precision을 구하고, np.mean()을 통해 평균을 계산한다

    return np.mean([apk(a, p, k, default) for a, p in zip(actual, predicted)]) 

%%time

import pandas as pd

import numpy as np

import xgboost as xgb



np.random.seed(2018)



# 데이터를 불러온다.

train_full = pd.read_csv("/kaggle/input/santander-product-recommendation/train_ver2.csv.zip")

trn = train_full.sample(n=10000).copy()

del train_full
%%time

test_full = pd.read_csv("/kaggle/input/santander-product-recommendation/test_ver2.csv.zip")

tst = test_full.sample(n=10000).copy()

del test_full
tst.head()
%%time

## 데이터 전처리 ##



# 제품 변수를 별도로 저장해 놓는다.

prods = trn.columns[24:].tolist()



# 제품 변수 결측값을 미리 0으로 대체한다.

trn[prods] = trn[prods].fillna(0.0).astype(np.int8)



%%time

# 24개 제품 중 하나도 보유하지 않는 고객 데이터를 제거한다.

no_product = trn[prods].sum(axis=1) == 0

trn = trn[~no_product]

# 훈련 데이터와 테스트 데이터를 통합한다. 테스트 데이터에 없는 제품 변수는 0으로 채운다.

for col in trn.columns[24:]:

    tst[col] = 0
%%time

df = pd.concat([trn, tst], axis=0)





# 학습에 사용할 변수를 담는 list이다.

features = []



# 범주형 변수를 .factorize() 함수를 통해 label encoding한다.

categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'nomprov', 'segmento']

for col in categorical_cols:

    df[col], _ = df[col].factorize(na_sentinel=-99)

features += categorical_cols

%%time

# 수치형 변수의 특이값과 결측값을 -99로 대체하고, 정수형으로 변환한다.

df['age'].replace(' NA', -99, inplace=True)

df['age'] = df['age'].astype(np.int8)



df['antiguedad'].replace('     NA', -99, inplace=True)

df['antiguedad'] = df['antiguedad'].astype(np.int8)



df['renta'].replace('         NA', -99, inplace=True)

df['renta'].fillna(-99, inplace=True)

df['renta'] = df['renta'].astype(float).astype(np.int8)



df['indrel_1mes'].replace('P', 5, inplace=True)

df['indrel_1mes'].fillna(-99, inplace=True)

df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)

%%time

# 학습에 사용할 수치형 변수를 features에 추구한다.

features += ['age','antiguedad','renta','ind_nuevo','indrel','indrel_1mes','ind_actividad_cliente']



# (피쳐 엔지니어링) 두 날짜 변수에서 연도와 월 정보를 추출한다.

df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)

df['fecha_alta_year'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)

features += ['fecha_alta_month', 'fecha_alta_year']



df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)

df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)

features += ['ult_fec_cli_1t_month', 'ult_fec_cli_1t_year']



# 그 외 변수의 결측값은 모두 -99로 대체한다.

df.fillna(-99, inplace=True)
%%time

# (피쳐 엔지니어링) lag-1 데이터를 생성한다.

# 코드 2-12와 유사한 코드 흐름이다.



# 날짜를 숫자로 변환하는 함수이다. 2015-01-28은 1, 2016-06-28은 18로 변환된다

def date_to_int(str_date):

    Y, M, D = [int(a) for a in str_date.strip().split("-")] 

    int_date = (int(Y) - 2015) * 12 + int(M)

    return int_date



# 날짜를 숫자로 변환하여 int_date에 저장한다

df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)



# 데이터를 복사하고, int_date 날짜에 1을 더하여 lag를 생성한다. 변수명에 _prev를 추가한다.

df_lag = df.copy()

df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns ]

df_lag['int_date'] += 1



# 원본 데이터와 lag 데이터를 ncodper와 int_date 기준으로 합친다. Lag 데이터의 int_date는 1 밀려 있기 때문에, 저번 달의 제품 정보가 삽입된다.

df_trn = df.merge(df_lag, on=['ncodpers','int_date'], how='left')



# 메모리 효율을 위해 불필요한 변수를 메모리에서 제거한다

del df, df_lag



# 저번 달의 제품 정보가 존재하지 않을 경우를 대비하여 0으로 대체한다.

for prod in prods:

    prev = prod + '_prev'

    df_trn[prev].fillna(0, inplace=True)

df_trn.fillna(-99, inplace=True)



# lag-1 변수를 추가한다.

features += [feature + '_prev' for feature in features]

features += [prod + '_prev' for prod in prods]



###

### Baseline 모델 이후, 다양한 피쳐 엔지니어링을 여기에 추가한다.

###





## 모델 학습

# 학습을 위하여 데이터를 훈련, 테스트용으로 분리한다.

# 학습에는 2016-01-28 ~ 2016-04-28 데이터만 사용하고, 검증에는 2016-05-28 데이터를 사용한다.

use_dates = ['2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']

trn = df_trn[df_trn['fecha_dato'].isin(use_dates)]

tst = df_trn[df_trn['fecha_dato'] == '2016-06-28']

del df_trn



# 훈련 데이터에서 신규 구매 건수만 추출한다.

X = []

Y = []

for i, prod in enumerate(prods):

    prev = prod + '_prev'

    prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]

    prY = np.zeros(prX.shape[0], dtype=np.int8) + i

    X.append(prX)

    Y.append(prY)

XY = pd.concat(X)

Y = np.hstack(Y)

XY['y'] = Y



# 훈련, 검증 데이터로 분리한다. 

vld_date = '2016-05-28'

XY_trn = XY[XY['fecha_dato'] != vld_date]

XY_vld = XY[XY['fecha_dato'] == vld_date]





# XGBoost 모델 parameter를 설정한다.

param = {

    'booster': 'gbtree',

    'max_depth': 8,

    'nthread': 4,

    'num_class': len(prods),

    'objective': 'multi:softprob',

    'silent': 1,

    'eval_metric': 'mlogloss',

    'eta': 0.1,

    'min_child_weight': 10,

    'colsample_bytree': 0.8,

    'colsample_bylevel': 0.9,

    'seed': 2018,

    }





features
XY_trn[features].values
XY_trn[['ind_empleado']]
#XY_trn.values(columns=features)

XY_trn.values
%%time

# 훈련, 검증 데이터를 XGBoost 형태로 변환한다.

#X_trn = XY_trn.as_matrix(columns=features)

X_trn = XY_trn[features].values

#Y_trn = XY_trn.as_matrix(columns=['y'])

Y_trn = XY_trn[['y']].values



dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)



#X_vld = XY_vld.as_matrix(columns=features)

X_vld = XY_vld[features].values



#Y_vld = XY_vld.as_matrix(columns=['y'])

Y_vld = XY_vld[['y']].values



dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)
X_trn
%%time

# XGBoost 모델을 훈련 데이터로 학습한다!

watch_list = [(dtrn, 'train'), (dvld, 'eval')]

model = xgb.train(param, dtrn, num_boost_round=1000, evals=watch_list, early_stopping_rounds=20)
%%time

# 학습한 모델을 저장한다.

import pickle

pickle.dump(model, open("./xgb.baseline.pkl", "wb"))

best_ntree_limit = model.best_ntree_limit



# MAP@7 평가 척도를 위한 준비작업이다.

# 고객 식별 번호를 추출한다.

vld = trn[trn['fecha_dato'] == vld_date]

#ncodpers_vld = vld.as_matrix(columns=['ncodpers'])

ncodpers_vld = vld[['ncodpers']]



# 검증 데이터에서 신규 구매를 구한다.

for prod in prods:

    prev = prod + '_prev'

    padd = prod + '_add'

    vld[padd] = vld[prod] - vld[prev]    

#add_vld = vld.as_matrix(columns=[prod + '_add' for prod in prods])

add_vld = vld[ [prod + '_add' for prod in prods] ]

add_vld_list = [list() for i in range(len(ncodpers_vld))]

print (len(ncodpers_vld))

print (len(prods))
add_vld.iloc[0,0]
add_vld.head()
add_vld.reset_index(drop=True)
%%time

# 고객별 신규 구매 정답 값을 add_vld_list에 저장하고, 총 count를 count_vld에 저장한다.

count_vld = 0

for ncodper in range(len(ncodpers_vld)):

    for prod in range(len(prods)):

        if add_vld.iloc[ncodper, prod] > 0:

            add_vld_list[ncodper].append(prod)

            count_vld += 1

                        

# 검증 데이터에서 얻을 수 있는 MAP@7 최고점을 미리 구한다. (0.042663)

print(mapk(add_vld_list, add_vld_list, 7, 0.0))
vld
features
vld.columns


# 검증 데이터에 대한 예측 값을 구한다.

#X_vld = vld.as_matrix(columns=features)

X_vld = vld[features].values



#Y_vld = vld.as_matrix(columns=['y'])

Y_vld = vld[['y']].values



dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

preds_vld = model.predict(dvld, ntree_limit=best_ntree_limit)



# 저번 달에 보유한 제품은 신규 구매가 불가하기 때문에, 확률값에서 미리 1을 빼준다

#preds_vld = preds_vld - vld.as_matrix(columns=[prod + '_prev' for prod in prods])

preds_vld = preds_vld - vld[[prod + '_prev' for prod in prods]].values





# 검증 데이터 예측 상위 7개를 추출한다.

result_vld = []

for ncodper, pred in zip(ncodpers_vld, preds_vld):

    y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]

    y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]

    result_vld.append([ip for y,p,ip in y_prods])

    

# 검증 데이터에서의 MAP@7 점수를 구한다. (0.036466)

print(mapk(add_vld_list, result_vld, 7, 0.0))
%%time



# XGBoost 모델을 전체 훈련 데이터로 재학습한다!

X_all = XY.as_matrix(columns=features)

Y_all = XY.as_matrix(columns=['y'])

dall = xgb.DMatrix(X_all, label=Y_all, feature_names=features)

watch_list = [(dall, 'train')]

# 트리 개수를 늘어난 데이터 양만큼 비례해서 증가한다.

best_ntree_limit = int(best_ntree_limit * (len(XY_trn) + len(XY_vld)) / len(XY_trn))

# XGBoost 모델 재학습!

model = xgb.train(param, dall, num_boost_round=best_ntree_limit, evals=watch_list)



# 변수 중요도를 출력해본다. 예상하던 변수가 상위로 올라와 있는가?

print("Feature importance:")

for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv: kv[1], reverse=True):

    print(kv)



%%time

# 캐글 제출을 위하여 테스트 데이터에 대한 예측 값을 구한다.

X_tst = tst.as_matrix(columns=features)

dtst = xgb.DMatrix(X_tst, feature_names=features)

preds_tst = model.predict(dtst, ntree_limit=best_ntree_limit)

ncodpers_tst = tst.as_matrix(columns=['ncodpers'])

preds_tst = preds_tst - tst.as_matrix(columns=[prod + '_prev' for prod in prods])



# 제출 파일을 생성한다.

submit_file = open('./xgb.baseline.2015-06-28', 'w')

submit_file.write('ncodpers,added_products\n')

for ncodper, pred in zip(ncodpers_tst, preds_tst):

    y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]

    y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]

    y_prods = [p for y,p,ip in y_prods]

    submit_file.write('{},{}\n'.format(int(ncodper), ' '.join(y_prods)))