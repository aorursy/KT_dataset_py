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
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train.shape, df_test.shape
df_train.head()
df_train.describe()
# numerical data가 아닌 애들은 안 나타난다. 79개 중 36개가 numeric
df_train.columns
# 컬럼 세부사항은 같이 첨부된 txt 파일에서 확인 가능
# 타겟(sales price)에 대한 정보
df_train.SalePrice.describe()
import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(df_train['SalePrice'])
plt.show()
# 지상면적과 집값의 관계

data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice')
plt.show()

# dataframe.plot.scatter(x, y) 형식으로 써줘야하기 때문에 pd.concat 을 먼저 해줬다.
# axis는 디폴트 0이며 0은 index, 1은 columns다. 즉, 0이면 saleprice 표 밑에 grlivarea가 붙는다.
# 지금은 axis=1 이므로 같은 인덱스끼리 해서 옆으로 챡 붙었다.
plt.scatter(x=df_train.GrLivArea, y=df_train.SalePrice)
plt.show()

# 이건 위랑 똑같은 그래프인데 pd.plot.scatter 과 plt.scatter을 비교하기 위해서 쓴 것.
# 크게 다른 점은 보이지 않는데 굳이 집자면 판다스는 x, y축 라벨이 자동으로 되어있고 해상도가 더 좋다(?)
# plt는 비주얼라이제이션 패키지이므로 더 자세하게 조정하고 싶다면 아마 plt가 나을 것 같다.
# 지금처럼 간단하게 확인만 할거라면 pd
# 지하면적과 집값의 관계

data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')
plt.show()
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
data.plot.scatter(x='OverallQual', y='SalePrice')
plt.show()

# 관계성은 보이지만 이거 말고 boxplot으로 보는게 좋겠다.
# numerical data 라기 보다는 categorical data
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
plt.show()

# 판다스 df.boxplot 은 제대로 못다루겠어서 일단 시본으로
data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
plt.figure(figsize=(20, 8))
fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
plt.xticks(rotation=90)
plt.show()
corrmat = df_train.corr()
# df.corr()은 df간의 correlation을 계산해준다.(null값 제외) correlation maxtrix를 데이터프레임 형식으로 return
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
# vmax는 value to anchor the colormap, 즉 기준값인듯? square는 셀을 정사각형으로 할건지 여부
plt.show()
corrmat
# 요 코드에 대한 설명은 길어져서 그래프 밑에 따로 서술

k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
plt.figure(figsize=(8, 8))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
cols
# np.corrcoef의 작동방식을 보자. 

a = np.array([[0.8, 0.7, 0.6, 0.8, 0.5],
              [0.2, 0.3, 0.4, 0.7, 0.1],
              [0.2, 0.4, 0.1, 0.1, 0.8]])
np.corrcoef(a)
# 결과값이 (3, 3)형태로 나오며 대각선을 기준으로 대칭임을 볼 수 있다. 
# 즉 1행1열은 a의 1행과 1행 간의 상관관계, 1행2열은 a의 1행과 2열 간의 상관관계, 이런식이다.
# a의 형태가 3,5가 아니라 3,100이었어도 return되는 어레이는 3,3이다
zoom_corrmat = df_train[cols].corr()
zoom_corrmat
plt.figure(figsize=(8, 8))
sns.set(font_scale=1.25)
hm = sns.heatmap(zoom_corrmat, )
plt.show()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
        'FullBath', 'YearBuilt']
# 분석한대로 GarageArea, 1stFlrSF, TotRmsAbvGrd 는 제외하고 다시 구성
sns.pairplot(df_train[cols], height = 2.5)
plt.show()
# 밑에 코드 설명 따로 기술 

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
total = df_train.isnull()
total
total = df_train.isnull().sum()
total
df_train.isnull().count()
# 바로 위에서 결정한 것들 실행

df_train = df_train.drop(missing_data[missing_data['Total'] > 1].index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() # missing value가 안 남아있는 것 확인
(missing_data[missing_data['Total'] > 1]).index
missing_data[missing_data['Total'] > 1].index
# 코드설명은 밑에
from sklearn.preprocessing import StandardScaler

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
sns.boxplot(saleprice_scaled)
plt.show()
df_train['SalePrice'].shape
saleprice_scaled[:,0].argsort()
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice')
plt.show()
drop_values = df_train.sort_values(by='GrLivArea', ascending=False)[:2]
# GrLivArea 기준으로 내림차순 정렬해서 앞의 두개 선택
df_train = df_train.drop(drop_values.index)
# 이 셀 여러번 실행하면 계속 셀 두개씩 없어지니 유의
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000), xlim=(0, 5000))
plt.show()
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')
plt.show()
