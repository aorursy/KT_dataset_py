# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 기본 세팅 값

import os # 디렉토리 설정

import numpy as np # 선형대수, 행렬, 벡터

import math # 각종 수치연산(루트 등)

import pandas as pd # CSV파일 읽기, DataFrame 객체, 평균, 중앙값, 분산, 표준편차, 사분위수, 상관관계

import matplotlib.pyplot as plt # 박스플랏, 산점도

import scipy.stats as stats # 정규분포, t분포, 신뢰구간(z분포, t분포), 가설검정

import statsmodels.api as sm # 비율의 신뢰구간, 비율의 가설검정, two-sample ttest, 평균차의 신뢰구간, 회귀분석

import statsmodels.formula.api as smf

import seaborn as sns



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.head()
train['LotArea'].mean()
train['LotArea'].min()
train['LotArea'].max()
train['LotArea'].median()
train['LotArea'].mode()[0]
train['LotArea'].value_counts()
ex1 = pd.DataFrame([2, 2, 2, 3, 5, 6, 6, 14], columns = ['data'])

ex1
train.head()
train.describe()
train['SalePrice'].var()
train['SalePrice'].std()
sns.distplot(train['OverallQual'], kde=True)

# plt.title('Histogram - LotArea')

plt.show()
train[['OverallQual','SalePrice']]
sns.boxplot(y="SalePrice", data=train)
sns.boxplot(x='OverallQual', y='SalePrice', data=train)

plt.show()
sns.distplot(train['SalePrice'], kde=True)

plt.title('Histogram - SalePrice')

plt.show()
train['SalePrice'].skew()
train['SalePrice'].kurt()
train.plot.scatter(x='LotArea', y='SalePrice')
train.columns
sns.jointplot(x='OverallQual', y='SalePrice', data=train)

plt.title('joint plot')

plt.show()
train[['OverallQual', 'GarageCars', 'YearRemodAdd','SalePrice']]
sns.pairplot(train[['OverallQual', 'GarageCars', 'YearRemodAdd','SalePrice']])

plt.show()
train.corr()['SalePrice'].sort_values(ascending=False)
stats.pearsonr(train['OverallQual'], train['SalePrice'])
train.corr()
# 다중 상관 분석(Heatmap)

import matplotlib as mpl

import matplotlib.pylab as plt

import seaborn as sns



corrmat = train.corr()

f,ax = plt.subplots(figsize = (12,9))

print(sns.heatmap(corrmat, vmax=.8, square=True, cmap="jet"))