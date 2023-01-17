# Data Science 하면 빠질 수 없는 모듈들

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# jupyter 내부에서 그래프를 출력

%matplotlib inline



# 아래 2줄은 공부할 때 캐글 전문가(이유한)님의 무료강의에서 보고 참고하게 된 방법이다.

plt.style.use('seaborn') # matplotlib의 기본 scheme이 아닌 seaborn scheme 사용

sns.set(font_scale=1.4) # Graph마다 font 크기를 지정해주지 않아도 됨.



# 누락데이터에 대한 시각화

import missingno as msno



# warnings를 무시해준다.

import warnings

warnings.filterwarnings('ignore')
# csv파일을 dataframe으로 가져옵니다.

# df_train = pd.read_csv('/kaggle/input/santander-product-recommendation/train_ver2.csv')

# 아래에서 내용 확인 후 date column을 indexing 해줍니다.

df_train = pd.read_csv('/kaggle/input/santander-product-recommendation/train_ver2.csv', index_col=0)
df_train.shape
df_train.head(10)
df_train.tail(10)
for col in df_train.columns:

    print('{}\n'.format(df_train[col].head(10)))
df_train.info()
skip_cols = ['ncodpers', 'renta']

for col in df_train.columns:

    # 출력에 너무 시간이 오래 걸려서 ncodpers, renta는 제외

    if col in skip_cols:

        continue

    

    # 보기 편하게 변수명과 함께 출력한다.

    print('col : ', col)

    

    # 그래프

    f, ax = plt.subplots(figsize=(20, 15))

    # seaborn을 사용한 막대 그래프 생성

    sns.countplot(x=col, data=df_train, alpha=0.5)

    # show!

    plt.show()
submission = pd.read_csv('/kaggle/input/santander-product-recommendation/sample_submission.csv')

submission.head()