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
df = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv').iloc[0:, 1:]

print(df.shape)

df.head(3)
# 원본 데이터 세트에 PAY_0다음이 PAY_2이니 순서에 맞게 변경

df.rename(columns = {'PAY_0':'PAY_1',

                     'default.payment.next.month':'default'}, inplace=True)

y_target = df['default']

x_features = df.drop('default', axis=1)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



corr = x_features.corr()

plt.figure(figsize=(14, 14))

sns.heatmap(corr, annot=True, fmt='.1g')
# BILL_AMT 1~6과 PAY_1~6 사이의 상관도가 높음

# 이 중 6개의 속성을 2개의 컴포넌트로 PCA 변환한 뒤 개별 컴포넌트의 변동성 알아보기 

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



cols_bill = ['BILL_AMT' + str(i) for i in range(1, 7)]

print('대상 속성명:', cols_bill)



# PCA 객체 생성하고 변동성 계산을 위해 fit 호출

scaler = StandardScaler()

df_cols_scaled = scaler.fit_transform(x_features[cols_bill])

pca = PCA(n_components=2)

pca.fit(df_cols_scaled)

print('PCA components별 변동성:', pca.explained_variance_ratio_)
# 원본 데이터 세트와 비교

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



rcf = RandomForestClassifier(n_estimators=300,

                            random_state=156)

scores = cross_val_score(rcf, x_features, y_target, scoring='accuracy', cv=3)



print('CV=3인 경우 개별 fold세트 별 정확도:', scores)

print('평균 정확도: {0:.4f}'.format(np.mean(scores)))
# 6개 컴포넌트 PCA 변환 후 분류 예측 비교

scaler = StandardScaler()

df_scaled = scaler.fit_transform(x_features)



pca = PCA(n_components=6)

df_pca = pca.fit_transform(df_scaled)

scores_pca = cross_val_score(rcf, df_pca, y_target, scoring='accuracy', cv=3)



print('CV=3인 경우의 PCA 변환된 개별 fold 세트별 정확도:', scores_pca)

print('PCA 변환 데이터 세트 평균 정확도: {0:.4f}'.format(np.mean(scores_pca)))