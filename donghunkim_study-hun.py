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
import seaborn as sns
import matplotlib.pyplot as plt
%pylab inline
pylab.rcParams['figure.figsize'] = (10, 6)

# 데이터 값 실수. 소수점 자리 설정
pd.options.display.float_format = '{:.2f}'.format
%%time 
train_full = pd.read_csv("/kaggle/input/santander-product-recommendation/train_ver2.csv.zip")
train = train_full.sample(n=10000).copy()
train.reset_index(drop=True)

del train_full
%%time 
train.shape
train.head()
train.fecha_dato.dtype
%time train.info()
np.unique(train.ind_nomina_ult1)
np.unique(train.ind_nom_pens_ult1.astype(str))
%time train.describe()
#memory 부족으로 실행 못함....
#import pandas_profiling as pp
#pp.ProfileReport(train)
%%time
num_cols = [col for col in train.columns[:24] if train[col].dtype in ['int64','float64']]
train[num_cols].describe()
%%time
cat_cols = [col for col in train[:24] if train[col].dtype in ['O']]

train[cat_cols].describe()
%%time
for col in cat_cols:
    uniq = np.unique(train[col].astype(str))
    print('-' * 50)
    print('# col = {} , n_uniq = {}, uniq = {}'.format(col, len(uniq), uniq) )
train['ind_nomina_ult1'].value_counts()
%%time
skip_cols = ['ncodpers','renta']
for col in train.columns[:3]:
    if col in skip_cols:
        continue
    print('-'*50)
    print('col : ', col)
    
    f, ax = plt.subplots(figsize=(15,10))
    sns.countplot(x=col, data=train, alpha=0.5)
    plt.show()
%%time
months = train['fecha_dato'].unique().tolist()
label_cols = train.columns[24:].tolist()

label_over_time = []
for i in range(len(label_cols)):
    #매월 각 제품의 총합을 groupby().agg('sum')으로 계산
    label_sum = train.groupby(['fecha_dato'])[label_cols[i]].agg('sum')
    label_over_time.append(label_sum.tolist())

label_sum_over_time = []
for i in range(len(label_cols)):
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0)) #<-- 내용 확인필요

color_list = ['#F5B7B1','#D2B4DE','#AED6F1','#A2D9CE','#ABEBC6','#F9E79F','#F5CBA7','#CCD1D1']
    
f, ax = plt.subplots(figsize=(30,15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_over_time[i], alpha=0.7, color=color_list[i%8])

plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor='none') for i in range(len(label_cols))], label_cols, loc=1, ncol=2, prop={'size':16} )

train.iloc[:5,24:]
label_sum_over_time
%%time
label_sum_percent = (label_sum_over_time / (1. * np.asarray(label_sum_over_time).max(axis=0))) * 100
    
f, ax = plt.subplots(figsize=(30,15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], alpha=0.7, color=color_list[i%8])

plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor='none') for i in range(len(label_cols))], label_cols, loc=1, ncol=2, prop={'size':16} )
plt.show()
#sample 데이터 10000개로 수행
trn = train
prods = train.columns[24:].tolist()
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-") ]
    int_date = (int(Y) - 2015)*12 + int(M)
    return int_date
trn['int_date'] = trn['fecha_dato'].map(date_to_int).astype(np.int8)
trn.head()
%%time
trn_lag = trn.copy()
trn_lag['int_date'] += 1
trn_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in trn.columns]
trn.head()
trn_lag.head()
%%time
df_trn = trn.merge(trn_lag, on=['ncodpers','int_date'], how='left')
#del trn, trn_lag
%%time
#지난달 정보 없는거 0으로 채움
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)

for prod in prods:
    padd = prod + '_add'
    prev = prod + '_prev'
    df_trn[padd] = ((df_trn[prod] == 1) & (df_trn[prev] == 0)).astype(np.int8)

add_cols = [prod + '_add' for prod in prods]    
labels = df_trn[add_cols].copy()
labels.columns = prods
labels.to_csv('./labels.csv',index=False)
labels['date'] = trn.fecha_dato
labels['date'].value_counts()
trn.fecha_dato.value_counts()
labels = pd.read_csv('./labels.csv').astype(int)
labels
fecha_dato = trn.fecha_dato
fecha_dato = fecha_dato.reset_index(drop=True)
fecha_dato
labels['date'] = fecha_dato
labels['date'].value_counts()
fecha_dato.value_counts()
labels
months = np.unique(fecha_dato).tolist()
labels_cols = labels.columns.tolist()[:24]

label_over_time = []
for i in range(len(label_cols)):
    label_over_time.append( labels.groupby(['date'])[label_cols[i]].agg('sum').tolist() )
    
label_sum_over_time = []
for i in range(len(label_cols)):
    label_sum_over_time.append( np.asarray(label_over_time[i:]).sum(axis=0) )
    
f, ax = plt.subplots(figsize=(30,15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_over_time[i], alpha=0.7)


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
