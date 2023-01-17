# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
submit = pd.read_csv("../input/sample_submission.csv")
y_train = train.price

X_train = train.drop(['id', 'price'], axis=1)

X_test = test.drop(['id'], axis=1)
row_idx = X_train.shape[0]
row_idx
df = pd.concat([X_train, X_test],axis=0)
df.columns
df.info()
buy_year = df.date.apply(lambda x:x[0:4]).astype(int)
year_subs = buy_year-df.yr_built
year_subs = year_subs.apply(lambda x:x+2)

#+2를 해주는 이유는 log로 scale 해줄 예정이기 때문에
df['year_subs'] = year_subs
df[df['year_subs']==1]
df = df.drop(['date'],axis=1)
df.lat.head()
# 0 : 시애틀

# 1 : 레드먼드

# 2 : 벨뷰

# 3 : 이사콰

# 4 : 스노퀄미

# 5 : 스노퀄미 패스

# 6 : 렌턴

# 7 : 시택

# 8 : 켄트

# 9 : 페더럴웨이

# 10 : 버클리

cities = [[47.6127, -122.3333],

 [47.6697,-122.1997],

 [47.6133,-122.1944],

 [47.5317,-122.0339],

[47.5294,-121.8297],

[47.4083,-121.4014],

[47.4839,-122.2165],

[47.4427,-122.2883],

[47.3833,-122.2453],

[47.3251,-122.3098],

[47.1694,-122.0208]]
cities_lat = [i[0] for i in cities]

cities_long = [i[1] for i in cities]
print(cities_lat)

print(cities_long)

city_idx = list(range(11))

print(city_idx)

df_cities = pd.DataFrame({'cities_lat':cities_lat, 'cities_long':cities_long, 'city_idx':city_idx})
df_cities
del cities_lat

del cities_long

del city_idx
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(df_cities[['cities_lat','cities_long']],df_cities['city_idx'])
pred_cities = knn.predict(df[['lat','long']])
df['city_idx'] = pred_cities
df.plot(kind='scatter',x='long',y='lat', alpha = 0.2,

       marker='o', c='city_idx', cmap=plt.get_cmap('jet'),figsize=(15,8))
X_train = df.iloc[:row_idx,:]

X_test = df.iloc[row_idx:,:]
fig, ax = plt.subplots(5,4,figsize=(20,20))

cols = X_train.columns

n=0

for r in range(5):

    for c in range(4):

        sns.kdeplot(X_train[cols[n]], ax=ax[r][c])

        ax[r][c].set_title(cols[n], fontsize=20)

        n+=1

        if n==X_train.shape[1]:

            break
def preprocessing(df):

    room_sum = df.bedrooms + df.bathrooms + 1

    df['roomsize'] = df.sqft_living / room_sum

    

    df.sqft_living = np.log(df.sqft_living)

    df.sqft_lot = np.log(df.sqft_lot)

    df.sqft_above = np.log(df.sqft_above)

    df.sqft_basement = np.log(df.sqft_basement)

    df.sqft_lot15 = np.log(df.sqft_lot15)

    df.sqft_living15 = np.log(df.sqft_living15)

    df.year_subs = np.log(df.year_subs)

    

    return df
X_train = preprocessing(X_train)

X_test = preprocessing(X_test)
cols = X_train.columns

fig, ax = plt.subplots(5,5,figsize=(20,20))

n = 0

for r in range(5):

    for c in range(5):

        sns.kdeplot(X_train[cols[n]],ax=ax[r][c])

        ax[r][c].set_title(cols[n],fontsize=20)

        n+=1

        if n==X_train.shape[1]:

            break

xgb_params = {

    'eta': 0.01,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



print('Transform DMatrix...')

dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(X_test)



print('Start Cross Validation...')



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=20,verbose_eval=50, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

print('best num_boost_rounds = ', len(cv_output))

rounds = len(cv_output)
model = xgb.train(xgb_params, dtrain, num_boost_round=rounds)

y_pred = model.predict(dtest)
submit['price'] =y_pred
submit.to_csv("suwon_study_ic.csv",index=False)