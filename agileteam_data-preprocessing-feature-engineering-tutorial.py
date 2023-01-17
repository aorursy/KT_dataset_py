import numpy as np
import pandas as pd

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='NanumBarunGothic') 

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm_notebook

from sklearn import preprocessing
import category_encoders as ce
path_house = "../input/house-prices-advanced-regression-techniques/train.csv"
df = pd.read_csv(path_house)
print(df.shape)
df.head()
df.describe()
df.describe(include=['O'])
df.isnull().sum()[:10]
df = pd.read_csv(path_house)

cols=['Alley', 'PoolQC']
df = df.drop(cols, axis=1)
df = pd.read_csv(path_house)
col = ["LotFrontage"]

#zero
df[col] = df[col].fillna(0)

# mean
df[col] = df[col].fillna(df[col].mean())

# median
df[col] = df[col].fillna(df[col].median())

# min
df[col] = df[col].fillna(df[col].min())

# max
df[col] = df[col].fillna(df[col].max())

#freq(최빈값)
# df[col] = df[col].fillna(df[col].mode()[0])
df = pd.read_csv(path_house)
col = ["LotFrontage"]

# 평균값
df[col] = df[col].fillna(df.groupby('MSZoning')[col].transform('mean'))

# 중앙
df[col] = df[col].fillna(df.groupby('MSZoning')[col].transform('median'))
df = pd.read_csv(path_house)
col = ["LotFrontage"]

# 앞 값으로 채우기
df[col] = df[col].fillna(method='ffill')

# 뒷 값으로 채우기
df[col] = df[col].fillna(method='bfill')
# 시계열데이터에서 선형으로 비례하는 방식으로 결측값 보간

df = pd.read_csv(path_house)

df = df.interpolate() # method='values
df = df.interpolate(method='time') # 날자기준으로 보간
df = df.interpolate(method='values', limit=1) #사이에 결측치가 여러개 있더라도 하나만 채우기
df =df.interpolate(method='values', limit=1, limit_direction='backward') #보간 방향 설정 뒤에서 앞으로
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp_mean.transform(X))
df = pd.read_csv(path_house)
plt.scatter(x=df['GrLivArea'], y=df['SalePrice'])
plt.xlabel('GrLivArea', fontsize=12)
plt.ylabel('SalePrice', fontsize=12)
outlier = df[(df['GrLivArea']>4000)&(df['SalePrice']<500000)].index
df=df.drop(outlier, axis=0)
df = pd.read_csv(path_house)
df.info()
df = pd.read_csv(path_house)
col = ['MSZoning']
cols = ['MSZoning', 'Neighborhood']


# Object -> Categorical

# 1개 변환
df[col] = df[col].astype('category')

# 여러개 변환
for c in cols : 
    df[c] = df[c].astype('category') 
df.info()
# 라벨 인코딩 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(path_house)
cols = ['SaleType', 'SaleCondition']

display(df[cols].head(1))

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    df[col]=le.fit_transform(df[col])

display(df[cols].head(1))
# 원핫 인코딩 
df = pd.read_csv(path_house)
cols = ['SaleType', 'SaleCondition']

display(df.head(1))

df_oh = pd.get_dummies(df[cols])
df = pd.concat([df, df_oh], axis=1)
df = df.drop(cols, axis=1)

display(df.head(1))
# !pip install category_encoders
# 카운터 인코딩


df = pd.read_csv(path_house)
col =['MSZoning']

display(df[col].head(1))

for col in tqdm_notebook(cols):
    count_enc = ce.CountEncoder()
    df[col]=count_enc.fit_transform(df[col])

display(df[col].head(1))
def labelcount_encode(X, categorical_features, ascending=False):
    print('LabelCount encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = X[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            # for ascending ordering
            value_counts_range = list(
                reversed(range(len(cat_feature_value_counts))))
        else:
            # for descending ordering
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        X_[cat_feature] = X[cat_feature].map(
            labelcount_dict)
    X_ = X_.add_suffix('_labelcount_encoded')
    if ascending:
        X_ = X_.add_suffix('_ascending')
    else:
        X_ = X_.add_suffix('_descending')
    X_ = X_.astype(np.uint32)
    return X_
df = pd.read_csv(path_house)
df['LotArea'] = labelcount_encode(df, ['LotArea'])
df.head(3)
df = pd.read_csv(path_house)
y = df['LotArea']
X = df['MSZoning']
Hashing_encoder = ce.HashingEncoder(cols = ['MSZoning'])
Hashing_encoder.fit_transform(X, y)
df = pd.read_csv(path_house)
y = df['LotArea']
X = df['MSZoning']
Sum_encoder = ce.SumEncoder(cols = ['MSZoning'])
Sum_encoder.fit_transform(X, y)
df = pd.read_csv(path_house)
y = df['LotArea']
X = df['SaleCondition']
ce_target = ce.TargetEncoder(cols = ['SaleCondition'])
ce_target.fit(X, y)
ce_target.transform(X, y)
# Standard Scaling (평균을 0, 분산을 1로 변경)
from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))
# MinMax Scaling 0과 1사이
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.data_max_)
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))
# Nomalization 정규화
from sklearn.preprocessing import Normalizer
X = [[4, 1, 2, 2],
     [1, 3, 9, 3],
     [5, 7, 5, 1]]
transformer = Normalizer().fit(X)  # fit does nothing.
transformer
transformer.transform(X)
# Standard와 유사 하나 평균과 분산 대신, median과 quartile을 사용
from sklearn.preprocessing import RobustScaler
X = [[ 1., -2.,  2.],
     [ -2.,  1.,  3.],
     [ 4.,  1., -2.]]
transformer = RobustScaler().fit(X)
transformer
transformer.transform(X)
# Log Scaling
df = pd.read_csv(path_house)
col =['SalePrice']
display(df[col].head(3))
df[col].plot(kind='kde')
df[col] = np.log1p(df[col]) # 원본 값
df[col].plot(kind='kde')

display(df[col].head(3)) # 로그 스케일

display(np.expm1(df[col]).head(3)) # expm으로 환원

# Quantile binning
df = pd.read_csv(path_house)
col =['LotArea']

q = df[col].quantile([.1,.5,1])
df[col].describe()
q
#이진화 0 또는 1

df = pd.read_csv(path_house)
col =['LotArea']

binarizer = preprocessing.Binarizer(threshold=10000)
b = binarizer.transform(df[col])
b = pd.DataFrame(b)
display(df[col])
display(b)
df = pd.DataFrame({'일시':['2020.7.1 19:00',
                   '2020.8.1 20:10',
                   '2021.9.1 21:20',
                   '2022.10.1 22:30',
                   '2022.11.1 23:30',
                   '2022.12.1 23:40',
                   '2023.1.1 08:30']})
df
df.info()
# 문자열을 datetime 타입으로 변경
df['일시'] = df.일시.apply(pd.to_datetime)
df.info()
# s1
df = df.assign(
               year=df.일시.dt.year,
               month=df.일시.dt.month,
               day=df.일시.dt.day,
               hour=df.일시.dt.hour,
               minute=df.일시.dt.minute,
    
               quarter=df.일시.dt.quarter,
               weekday=df.일시.dt.weekday,
               weekofyear=df.일시.dt.weekofyear,
    
               month_start=df.일시.dt.is_month_start,
               month_end=df.일시.dt.is_month_end,
               quarter_start=df.일시.dt.is_quarter_start,
               quarter_end=df.일시.dt.is_quarter_end,
    
               daysinmonth=df.일시.dt.daysinmonth
               )
df.head(7)
# datetime 타입에서 년, 월, 일, 시간 추출
#2
df['year'] = df.일시.apply(lambda x : x.year)
df['month'] = df.일시.apply(lambda x : x.month)
df['day'] = df.일시.apply(lambda x : x.day)
df['hour'] = df.일시.apply(lambda x: x.hour)
df['minute'] = df.일시.apply(lambda x: x.minute)

#3
df['weekday'] = df['일시'].dt.weekday
df['weekofyear'] = df["일시"].dt.weekofyear
df['quarter'] = df["일시"].dt.quarter
# kaggle 데이터 셋에서 날씨로 된 데이터를 불러옵니다! 
path_house = "../input/austin-weather/austin_weather.csv"
w = pd.read_csv(path_house)
w.head()
w.info()
w['Date'] = w['Date'].apply(pd.to_datetime)
w.info()
w.head(8)
df = pd.DataFrame({'date':['2013.12.22 19:00',
                   '2013.12.23 20:10',
                   '2013.12.24 21:20',
                   '2013.12.25 22:30',
                   '2013.12.26 23:30',
                   '2013.12.27 23:40',
                   '2013.12.28 08:30'], 
                   'name':['A',
                   'B',
                   'C',
                   'D',
                   'E',
                   'F',
                   'G']})
df['date'] = df.date.apply(pd.to_datetime)
df
df = df.assign(
               year=df.date.dt.year,
               month=df.date.dt.month,
               day=df.date.dt.day
               )
df.head()
w = w.assign(
               year=w.Date.dt.year,
               month=w.Date.dt.month,
               day=w.Date.dt.day
               )
w.head()
df = pd.merge(df, w, how='left', on=['year','month','day'])
df.head()
# house-prices-advanced-regression Data-set
df = pd.read_csv(path_house)
df.head()
df.head()
# groupby 작성법1
df.groupby('MSZoning')['LotArea'].max()
df.groupby('MSZoning')['LotArea'].min()
df.groupby('MSZoning')['LotArea'].mean()
df.groupby(['MSZoning','LotShape'])['LotArea'].count()
# groupby 작성법2
df_group = df.groupby('MSZoning')
df_group['LotArea'].max() 
# 피처생성방법1
df['new_max'] = df.groupby('MSZoning')['LotArea'].transform(lambda x: x.max())
df.head()
df_agg = pd.DataFrame()
df_agg =  df.groupby('MSZoning')['LotArea'].agg(['max', 'min', 'mean'])
# 피처생성방법2
df_all = pd.merge(df, df_agg, how='left', on=['MSZoning'])
df_all.head()

# Real or Not? NLP with Disaster Tweets Data-set
df = pd.read_csv('../input/nlp-getting-started/train.csv')
df.head(10)
df["length"] = df["text"].str.len()
df.head()
df[df['text'].str.contains('911')].head()
df['k'] = 0
df.loc[df[df['text'].str.contains('emergency')].index,'k'] = 1
df.loc[df[df['text'].str.contains('help')].index,'k'] = 2
df.loc[df[df['text'].str.contains('accident')].index,'k'] = 3

df = pd.read_csv('../input/sandp500/all_stocks_5yr.csv')
print(df.shape)
df.head()
df = df[:300]
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma30'] = df['close'].rolling(window=30).mean()
df['ma60'] = df['close'].rolling(window=60).mean()
df.tail(6)
plt.plot(df.index, df['ma5'], label = "ma5")
plt.plot(df.index, df['ma30'], label = "ma30")
plt.plot(df.index, df['ma60'], label = "ma60")
plt.plot(df.index, df['close'], label='close')
plt.legend()
plt.grid()
# shift(-1) 밀어서 피처 생성
df['nextClose']= df['close'].shift(-1)

# 주가변동
df['fluctuation'] = df['nextClose'] - df['close']
df['f_rate'] = df['fluctuation'] / df['nextClose']

df.head()
plt.figure(figsize=(10,6))
plt.plot(df.index, df['f_rate'])
plt.axhline(y=0, color='gray', ls = '-')
# 분포 
df['f_rate'].plot.hist()
# df['f_rate'].plot.kde()
df['f_rate'].plot.kde()
