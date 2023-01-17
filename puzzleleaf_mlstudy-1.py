import pandas as pd

import numpy as np

import zipfile



def apk(actual, predicated, k=7, default=0.0):

    # MAP@7 이므로 최대 7개 까지 사용

    if len(predicated) > k:

        predicated = predicated[:k]

    

    score = 0.0

    num_hits = 0.0

    

    for i, p in enumerate(predicated):

        if p in actual and p not in predicated[:i]:

            num_hits += 1.0

            score += num_hits / (i + 1.0)

    

    if not actual:

        return default

    

    return score / min(len(actual), k)

    

def mapk(actual, predicated, k=7, default=0.0):

    return np.mean([apk(a, p, k, default) for a, p in zip(actual, predicated)])





trn = pd.read_csv('../input/s-product-recomendation/train_ver2.csv');



trn.shape

# 13.647.309 명의 고객이 있고, 각 고객당 48개의 컬럼이 존재한다.
trn.head()
for col in trn.columns:

    print('{}\n'.format(trn[col].head()))
# pandas에서 자동으로 정보를 요약

trn.info()
# 첫 24개의 고객 변수 중 ['int64', 'float64'] 의 데이터 타입을 갖는 수치형 변수 분석

num_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['int64', 'float64']]

trn[num_cols].describe()
# 첫 24개의 고객 변수 중 ['object'] 의 데이터 타입을 갖는 범주형 변수 분석

cat_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['O']]

trn[cat_cols].describe()
for col in cat_cols:

    uniq = np.unique(trn[col].astype(str))

    print('-' * 50)

    print('# col {}, n_uniq {}, uniq {}'.format(col, len(uniq), uniq))
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
skip_cols = ['ncodpers', 'renta']

for col in trn.columns:

    if col in skip_cols:

        continue

        

    print('-' * 50)

    print('col : ', col)

    

    f, ax = plt.subplots(figsize=(20, 15))

    sns.countplot(x=col, data=trn, alpha=0.5)

    plt.show()