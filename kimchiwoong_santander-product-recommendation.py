import numpy as np 

import pandas as pd

import random 



import sys

import os

print(os.listdir("../input"))
pd.options.display.max_columns = 200

pd.options.display.max_rows = 1500
%%time

#df_train = pd.read_csv('../input/train_ver2.csv', nrows=1000000, encoding='ms949', engine='python') # RAM 사양 고려, rows 수 축소 : 13.6M -> 1M 

#df_test = pd.read_csv('../input/test_ver2.csv', nrows= 1000, encoding='ms949', engine='python')





filename = "../input/train_ver2.csv"

n = sum(1 for line in open(filename))  # 행의 개수

s = int(n * 0.0001) # 사용할 행의 개수

skip = sorted(random.sample(range(1,n+1),n-s)) #랜덤 샘플링해서 사용할 개수 제외 수



df_train = pd.read_csv(filename, skiprows=skip)

df_test = pd.read_csv("../input/test_ver2.csv", skiprows=skip)
sample_submission = pd.read_csv('../input/sample_submission.csv', nrows = 10)

sample_submission # 제출 시, added_products 에 상위 7개 추천 제품을 공백을 띄워서 저장 제출하면 된다.
sample_submission.to_csv('submission.csv', index=False)
df_train.shape, df_test.shape

# 1) Check Data Set

df_train.head()
df_test.head()
# 2) Change Column Names to Korean



tmp1 = ["날짜(fecha_dato)", "고객 고유식별번호(ncodpers)","고용 지표(ind_empleado)", "고객 거주 국가(pais_residencia)",

        "성별(sexo)", "나이(age)", "고객&은행 간 첫 계약 체결 날짜(fecha_alta)","신규 고객 지표(ind_nuevo)",

        "은행 거래 누적 기간(antiguedad)", "고객 등급(indrel)", "1등급 고객으로서 마지막 날짜(ult_fec_cli_1t)",

        "월초 기준 고객 등급(indrel_1mes)","월초 기준 고객 관계 유형(tiprel_1mes)", "거주 지표(indresi)",

        "외국인 지표(indext)", "배우자 지표(conyuemp)", "고객 유입 채널(canal_entrada)","고객 사망 여부(indfall)",

        "주소 유형(tipodom)", "지방 코드(cod_prov)", "지방 이름(nomprov)", "활발성 지표(ind_actividad_cliente)",

        "가구 총 수입(renta)", "분류(segmento)", "예금(ind_ahor_fin_ult1)", "보증(ind_aval_fin_ult1)",

        "당좌 예금(ind_cco_fin_ult1)", "파생 상품 계좌(ind_cder_fin_ult1)", "급여 계정(ind_cno_fin_ult1)",

        "청소년 계정(ind_ctju_fin_ult1)", "마스 특별 계정(ind_ctma_fin_ult1)", "특정 계정(ind_ctop_fin_ult1)",

        "특정 플러스 계정(int_ctpp_fin_ult1)","단기 예금(ind_deco_fin_ult1)", "중기 예금(ind_deme_fin_ult1)",

        "장기 예금(ind_dela_fin_ult1)", "e-계정(ind_ecue_fin_ult1)", "펀드(ind_fond_fin_ult1)",

        "부동산 대출(ind_hip_fin_ult1)", "연금(ind_plan_fin_ult1)", "대출(ind_pres_fin_ult1)", "세금(ind_reca_fin_ult1)",

        "신용카드(ind_tjcr_fin_ult1)","증권(ind_valo_fin_ult1)", "홈 계정(ind_viv_fin_ult1)", "급여(ind_nomina_ult1)",

        "연금2(ind_nom_pens_ult1)", "직불 카드(ind_recibo_ult1)"]



tmp2 = ["날짜(fecha_dato)", "고객 고유식별번호(ncodpers)","고용 지표(ind_empleado)", "고객 거주 국가(pais_residencia)",

        "성별(sexo)", "나이(age)", "고객&은행 간 첫 계약 체결 날짜(fecha_alta)","신규 고객 지표(ind_nuevo)",

        "은행 거래 누적 기간(antiguedad)", "고객 등급(indrel)", "1등급 고객으로서 마지막 날짜(ult_fec_cli_1t)",

        "월초 기준 고객 등급(indrel_1mes)","월초 기준 고객 관계 유형(tiprel_1mes)", "거주 지표(indresi)",

        "외국인 지표(indext)", "배우자 지표(conyuemp)", "고객 유입 채널(canal_entrada)","고객 사망 여부(indfall)",

        "주소 유형(tipodom)", "지방 코드(cod_prov)", "지방 이름(nomprov)", "활발성 지표(ind_actividad_cliente)",

        "가구 총 수입(renta)", "분류(segmento)"]



df_train.columns = tmp1

df_test.columns = tmp2
df_train.head()
df_train.tail()
df_test.head()
# 3) Check Column Data Type & Missing Data

def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(df_train)
missing_data(df_test)
df = pd.concat([df_train, df_test], axis=0)

df.head()
df_train.shape, df_test.shape, df.shape
missing_data(df)
# 4) Change Unpair Data Type



# '나이(age)' object -> int

df['나이(age)'].replace(' NA', -99, inplace=True)

df['나이(age)'].astype(int)



# '신규 고객 지표(ind_nuevo)' float -> int

df['신규 고객 지표(ind_nuevo)'].fillna(-99).astype(int)



# '은행 거래 누적 기간(antiguedad)' object -> int

df['은행 거래 누적 기간(antiguedad)'].replace('     NA', -99, inplace=True)

df['은행 거래 누적 기간(antiguedad)'].astype(int)



# '고객 등급(indrel)' float -> int

df['고객 등급(indrel)'].fillna(-99).astype(int)



# '월초 기준 고객 등급(indrel_1mes)' float -> int

df['월초 기준 고객 등급(indrel_1mes)'].fillna(-99).astype(int)



# '지방 코드(cod_prov)' float -> int

df['지방 코드(cod_prov)'].fillna(-99).astype(int)



# '활발성 지표(ind_actividad_cliente)' float -> int

df['활발성 지표(ind_actividad_cliente)'].fillna(-99).astype(int)



# '가구 총 수입(renta)' float -> int

#df['가구 총 수입(renta)'].fillna(-99).astype(int)





y_columns = []

for col in df.columns :

    if 'ult1' in col :

        y_columns.append(col)



print(len(y_columns))

print(y_columns)
df.columns
'활발성 지표(ind_actividad_cliente)' in df.columns
X_columns = []



for col in df.columns :

    if col in y_columns :

        continue

    X_columns.append(col)



print(len(X_columns))

print(X_columns)
# 4) Check Data Description



# 4-1) Numeric Columns



num_cols = [col for col in X_columns if df[col].dtype in ['int64', 'float64']]

df[num_cols].describe()

# 4-2) Category Columns



cat_cols = [col for col in X_columns if df[col].dtype in ['object']]

df[cat_cols].describe()
for col in cat_cols :

    uniq = np.unique(df[col].astype(str)) # astype : data 형태 변환 vs dtype : data 형태 확인

    print('-' * 100)

    print('# col {}, n_uniq {}, uniq {}'.format(col, len(uniq), uniq))
# 4-3) Data Visualization



import matplotlib.pyplot as plt

%matplotlib inline

# Jupyter Notebook 내부에 그래프 출력하도록 설정

import seaborn as sns



skip_column_because_to_many_variables = ['고객 고유식별번호(ncodpers)','가구 총 수입(renta)']

for col in df.columns:

    if col in skip_column_because_to_many_variables :

        continue

    print('-' *100)

    print('col :', col)

    

    f, ax = plt.subplots(figsize=(12,10))

    sns.countplot(x=col, data=df, alpha=0.5)

    plt.show()
# 4-4) Time Series Data Visualization



# 날짜 데이터를 기준으로 분석하기 위하여, 날짜 데이터를 별도로 추출한다.

months = df_train["날짜(fecha_dato)"].unique().tolist()



label_cols = df_train.columns[24:].tolist()



label_over_time = []

for i in range(len(label_cols)):

    # 매월, 각 제품의 총합을 groupby(..).agg('sum')으로 계산하여 label_sum 에 저장한다.

    label_sum = df_train.groupby(["날짜(fecha_dato)"])[label_cols[i]].agg('sum')

    label_over_time.append(label_sum.tolist())



print(label_over_time)
# 월별 금융 제품 보유 데이터를 누적 막대그래프로 시각화 (누적 증가 확인)

label_sum_over_time = []

for i in range(len(label_cols)):

    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))

print(label_sum_over_time)



# 누적 막대 그래프를 시각화하기 위해 n번째 제품의 총합을 1~n번째 제품의 총합으로 만든다.

color_list = ['#F5B7B1', '#D2B4DE', '#AED6F1', '#A2D9CE', '#ABEBC6', '#F9E79F', '#F5CBA7', '#CCD1D1']



f, ax = plt.subplots(figsize=(30,15))

for i in range(len(label_cols)):

    sns.barplot(x=months, y=label_sum_over_time[i], color = color_list[i%8], alpha=0.7)

plt.legend(

    [plt.Rectangle((0,0),1,1,fc = color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))],

    label_cols,

    loc=1,

    ncol=2,

    prop={'size':16})

    
# 월별 금융 제품 보유 데이터를 누적 막대그래프로 시각화 (상대 증가 확인)



label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))) *100 



#asarray -> input 을 array 로 변환

color_list = ['#F5B7B1', '#D2B4DE', '#AED6F1', '#A2D9CE', '#ABEBC6', '#F9E79F', '#F5CBA7', '#CCD1D1']



f, ax = plt.subplots(figsize=(30,15))

for i in range(len(label_cols)):

    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha=0.7)

plt.legend(

    [plt.Rectangle((0,0),1,1,fc = color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))],

    label_cols,

    loc=1,

    ncol=2,

    prop={'size':16})

    


#df_train = pd.read_csv('../input/train_ver2.csv', nrows = 1000000)



# 제품 컬럼을 리스트로 저장

prods = df_train.columns[24:].tolist()



# 제품 값 결측치를 0으로 저장

df_train[prods] = df_train[prods].fillna(0.0).astype(np.int8)

# 제품을 보유하고 있지 않은 고객은 제거

no_product = df_train[prods].sum(axis=1) == 0



df_train = df_train[~no_product]

df_train.shape
# test set 에는 y 값이 없으므로, 0으로 임의 생성해준 뒤 합쳐줌



for col in df_train.columns[24:] :

    df_test[col] = 0

    

df = pd.concat([df_train, df_test], axis=0)
print("train 수 : ", df_train.shape)

print("test 수 : ", df_test.shape)

print("df 수 : ", df.shape)
df.head()
# col name : 고객&은행 간 첫 계약 체결 날짜(fecha_alta)

df['고객&은행 간 첫 계약 체결 날짜(fecha_alta)_month'] = df['고객&은행 간 첫 계약 체결 날짜(fecha_alta)'].map(lambda x : 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)

df['고객&은행 간 첫 계약 체결 날짜(fecha_alta)_year'] = df['고객&은행 간 첫 계약 체결 날짜(fecha_alta)'].map(lambda x : 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)



# col name : 1등급 고객으로서 마지막 날짜(ult_fec_cli_1t)

df['1등급 고객으로서 마지막 날짜(ult_fec_cli_1t)_month'] = df['1등급 고객으로서 마지막 날짜(ult_fec_cli_1t)'].map(lambda x : 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)

df['1등급 고객으로서 마지막 날짜(ult_fec_cli_1t)_year'] = df['1등급 고객으로서 마지막 날짜(ult_fec_cli_1t)'].map(lambda x : 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)



df.head()
#for col in cat_cols :

#    df[col], _ = df[col].factorize(na_sentinel=-99)

# factorize : 0,1,2,3,4,5... 로 변경
df.head()
# 1) 제품 변수를 list 형태로 저장

products = df.columns[24:]



# 2) 날짜를 숫자로 변환하는 함수 선언

def date_to_int(str_date):

    Y, M, D = [int(a) for a in str_date.strip().split("-")]

    int_date = (int(Y) - 2015) * 12 + int(M)

    return int_date



df['int_date'] = df['날짜(fecha_dato)'].map(date_to_int).astype(np.int8)



# 3) int_date 에 1을 더하여 lag를 생성하고, 변수명에 _prev 를 추가



df_lag = df.copy()

df_lag.columns = [col + '_prev' if col not in ['int_date', '고객 고유식별번호(ncodpers)'] else col for col in df.columns]

df_lag['int_date'] += 1



# 4) 원본 데이터와 lag 데이터를 '날짜(fecha_dato)', '고객 고유식별번호(ncodpers)' 기준으로 합친다. 

#    lag 데이터의 int_date 는 1 밀려있기 때문에 저번달의 제품 정보가 삽입된다.



df = df.merge(df_lag, on = ['int_date', '고객 고유식별번호(ncodpers)'], how = 'left')



# 5) 메모리 효율을 위해 불필요 변수 제거

del df_lag



# 6) 저번 달의 제품 정보가 존재 하지 않을 경우를 대비하여 0으로 대체

for prod in products :

    prev = prod + '_prev'

    df[prev].fillna(0, inplace=True)



df.fillna(-99, inplace=True)

df.head()
df['날짜(fecha_dato)'].unique()
df.shape[0]
trn_dates = df['날짜(fecha_dato)'].unique()[:-2]

val_dates = df['날짜(fecha_dato)'].unique()[-1]

print(trn_dates)

print(val_dates)
trn = df[df['날짜(fecha_dato)'].isin(trn_dates)]

val = df[df['날짜(fecha_dato)'] == '2016-06-28']
trn.shape, val.shape
X = []

Y = []



for i, prod in enumerate(products):

    prev = prod + '_prev'

    prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]

    prY = np.zeros(prX.shape[0], dtype=np.int8) + i

    X.append(prX)

    Y.append(prY)
XY = pd.concat(X)

Y = np.hstack(Y)

XY['y'] = Y
features = []

for col in XY.columns :

    XY[col], _ = XY[col].factorize(na_sentinel=-99)

# factorize : 0,1,2,3,4,5... 로 변경



XY['날짜(fecha_dato)'].unique()
XY_trn = XY[XY['날짜(fecha_dato)'] != 15] # 15 = 2016-06-28

XY_val = XY[XY['날짜(fecha_dato)'] == 15]
#from sklearn.preprocessing import OneHotEncoder
#XY_trn_obj = XY_trn.select_dtypes(include='object')

#XY_trn_obj_one_hot = XY_trn.obj.
#qq = OneHotEncoder(XY_trn_obj)
XY.columns
XY.columns
param = {

    'booster': 'gbtree',

    'max_depth': 8,

    'nthread': 4,

    'num_class': len(products),

    'objective': 'multi:softprob',

    'silent': 1,

    'eval_metric': 'mlogloss',

    'eta': 0.1,

    'min_child_weight': 10,

    'colsample_bytree': 0.8,

    'colsample_bylevel': 0.9,

    'seed': 2018,

    }


# trn / val 데이터를 XGBoost 형태로 변환

y_trn = XY_trn.as_matrix(columns=['y'])



features = []

for col in XY.columns:

    if 'y' in col :

        continue

    features.append(col)



X_trn = XY_trn.as_matrix(columns=features)
X_val = XY_val.as_matrix(columns=features)

y_val = XY_val.as_matrix(columns=['y'])
import xgboost as xgb

dtrn = xgb.DMatrix(X_trn, label=y_trn, feature_names = features)

dval = xgb.DMatrix(X_val, label=y_val, feature_names = features)
watch_list = [(dtrn, 'train'), (dval, 'eval')]

model = xgb.train(param, dtrn, num_boost_round=1000, evals=watch_list, early_stopping_rounds=20)
best_ntree_limit=model.best_ntree_limit

print(best_ntree_limit)
preds_val = model.predict(dval, ntree_limit = best_ntree_limit)
preds_val