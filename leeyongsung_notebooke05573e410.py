# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('dark')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
items=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

shops=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

cats=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

train=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
#특이치 제거
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

flierprops = dict(marker='o', markerfacecolor='purple', markersize=6,

                  linestyle='none', markeredgecolor='black')

sns.boxplot(x=train.item_cnt_day, flierprops=flierprops)



plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price, flierprops=flierprops)
train = train[(train.item_price < 300000 )& (train.item_cnt_day < 1000)] # 1000개이상 팔린 품목 제거, 30만원이상 품목 제거
train = train[train.item_price > 0].reset_index(drop = True) # 가격이 마이너스인 가격 제거, 환불될 가능성이있음

train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0 # 판매갯수가 0개인건 -1로 변경
train.loc[train.shop_id == 0, 'shop_id'] = 57 # 둘이 같은 상점인데 이름에 pah가 껴있음

test.loc[test.shop_id == 0, 'shop_id'] = 57

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"' # 띄어쓰기 되있는 도시 이름 바꾸기

shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] ) # 공백을 기준으로 문자를 나누고 0번째

shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] ) # 1번째

shops.loc[shops.city == "!Якутск", "city"] = "Якутск"  # 도시이름에 !가 들어가면 바꿔줌
category = []

for cat in shops.category.unique(): # category의 문자열 유니크값

    if len(shops[shops.category == cat]) >= 5: # 만약 shops['category'] 개수가 5가 크거나 같을때

        category.append(cat) # category안에 유니크값을 넣는다 

shops.category = shops.category.apply( lambda x: x if (x in category) else "other" ) # 5개가 있는 상점이 카테고리 안에있으니까

                                                                                     # 5개인 상점은 그대로 바꾸고 그 아래것들은 other로 변경
from sklearn.preprocessing import LabelEncoder 

shops["shop_category"] = LabelEncoder().fit_transform( shops.category ) # category를 숫자로

shops["shop_city"] = LabelEncoder().fit_transform( shops.city )# shops.city를 숫자로

shops = shops[["shop_id", "shop_category", "shop_city"]] # shops안에 shops_name빼고 요렇게
cats["type_code"] = cats.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str) #  " " 공백기준 문자나누고, 0번째로 바꾸기

cats.loc[ (cats.type_code == "Игровые")| (cats.type_code == "Аксессуары"), "category" ] = "Игры" # type_code가 저거거나 이거거나 만족하면 category열은 만든 후 요걸로 바꿈
category = []

for cat in cats.type_code.unique(): 

    if len(cats[cats.type_code == cat]) >= 5: # 유니크값의 크기가 5보다 크거나 같을때 

        category.append( cat ) # category에 유니크값 추가

cats.type_code = cats.type_code.apply(lambda x: x if (x in category) else "etc") # cats['type_code']안에 category 값이 있으면 그대로 아니면 etc 반환
cats.type_code = LabelEncoder().fit_transform(cats.type_code)

cats["split"] = cats.item_category_name.apply(lambda x: x.split("-")) # -기준으로 나눠서

cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip()) # 만약 문자열 x의 크기가 1보다크면 1을 기준으로나눔 아니면 0을 기준으로 나눔

cats["subtype_code"] = LabelEncoder().fit_transform( cats["subtype"] ) # 라벨인코딩

cats = cats[["item_category_id", "subtype_code", "type_code"]] # item_category_name 빼고 나머지로 데이터프레임구성
import re

def name_correction(x):

    x = x.lower() # 소문자로 바꿔

    x = x.partition('[')[0] # [을 기준으로 문자열나눠

    x = x.partition('(')[0] # (을 기준으로 문자열나눠

    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) # x의 패턴이외 ' '로 바꾸심

    x = x.replace('  ', ' ') # 스페이스바 두번누른걸 한번으로바꿈

    x = x.strip() # 앞뒤 공백지우기

    return x


items["name1"], items["name2"] = items.item_name.str.split("[", 1).str # 1번에 [을 기준으로 나눔

items["name1"], items["name3"] = items.item_name.str.split("(", 1).str # 1번에 (을 기준으로 나눔





items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower() # 대문자를 소문자로

items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower() # 대문자를 소문자로





items = items.fillna('0') # items에 공백은 0으로 채움



items["item_name"] = items["item_name"].apply(lambda x: name_correction(x)) # item_name 클리닝



items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0") # items['name2']에 0이없으면 그대로 아니면 0
items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] ) # " "을 기준으로 나눠, 0번째가 xbox면 x의 0:8길이 까지만, 아니면 0번째전체

items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360" # items['type']이 x360 or xbox360 or xbox 360 모두 xbox 360으로 변환

items.loc[ items.type == "", "type"] = "mac" # type안에 type이 공백이면 mac으로 

items.type = items.type.apply( lambda x: x.replace(" ", "") ) # " "이면 공백으로

items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc" # 글자가 다름

items.loc[ items.type == 'рs3' , "type"] = "ps3" # 글자가 다름
group_sum = items.groupby(["type"]).agg({"item_id": "count"}) # type을 기준으로 item_id를 정렬후 group_sum에 저장

group_sum = group_sum.reset_index() #인덱스를 원래대로 

drop_cols = []

for cat in group_sum.type.unique():

    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40: # item_id에 type값이 cat이랑 같을때 그 값이 40보다 작은경우

        drop_cols.append(cat) # drop_cols에 추가

items.name2 = items.name2.apply( lambda x: "other" if (x in drop_cols) else x ) # items['name2']에 drop_cols값이 있으면 other 아니면 그대로

items = items.drop(["type"], axis = 1) # type을 제거
group_sum.loc[(group_sum.type == cat), 'item_id']
items.name2 = LabelEncoder().fit_transform(items.name2) # 라벨인코딩

items.name3 = LabelEncoder().fit_transform(items.name3)



items.drop(["item_name", "name1"],axis = 1, inplace= True) # items에 item_name, name1 컬럼제거

items.head()
from itertools import product

import time

ts = time.time() # 현재시간이요

matrix = []

cols  = ["date_block_num", "shop_id", "item_id"] 

for i in range(34):

    sales = train[train.date_block_num == i] # date_block_num(달마다 숫자로 표현)

    matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )

    # i, sale['shop_id'], sale['item_id'] 유니크값을 곱집합한걸 리스트로 나타내어 배열로 만든걸 추가



matrix = pd.DataFrame( np.vstack(matrix), columns = cols ) # matrix를 세로로 결합후 col대로 데이터프레임만듬

matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8) # matrix['date_block_num']을 정수형으로

matrix["shop_id"] = matrix["shop_id"].astype(np.int8) # matrix['shop_id']를 정수형으로

matrix["item_id"] = matrix["item_id"].astype(np.int16) # matrix['item_id']를 정수형으로

matrix.sort_values( cols, inplace = True ) # 열을 오름차순으로 정렬

time.time()- ts # 아까 시간 - 지금 시간 = 실행시간
matrix
# add revenue to train df

train["revenue"] = train["item_cnt_day"] * train["item_price"] # 판갯수 * 판매가격으로 train['revenue'] 컬럼만듬
ts = time.time()

group = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} ) # 3개열 기준으로 item_cnt_day를 

group.columns = ["item_cnt_month"] # group에 item_cnt_month 열 추가

group.reset_index( inplace = True)

matrix = pd.merge( matrix, group, on = cols, how = "left" ) # matrix와 group을 cols기준으로 왼쪽으로 병합

matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).astype(np.float16) # item_cnt_month의 null값은 0이고 소수점형태로 나타낸다

time.time() - ts
test["date_block_num"] = 34 # test['date_block_num']에는 34을 넣음

test["date_block_num"] = test["date_block_num"].astype(np.int8) # 정수형으로 표현

test["shop_id"] = test.shop_id.astype(np.int8) # shop_id를 정수형으로

test["item_id"] = test.item_id.astype(np.int16)# item_id를 정수형으로
ts = time.time()



matrix = pd.concat([matrix, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)

# matrix와 drop되는 ID이외에 컬럼들을 합침? 기존 index를 인덱스를 유지하지않고, 내림차순 계층적 인덱스 사용

matrix.fillna( 0, inplace = True ) # matrix 공백값은 0으로

time.time() - ts
ts = time.time()

matrix = pd.merge( matrix, shops, on = ["shop_id"], how = "left" ) # shop_id를 기준으로 왼쪽 데이터프레임으로 결합

matrix = pd.merge(matrix, items, on = ["item_id"], how = "left") # item_id를 기준으로 왼쪽 데이터프레임으로 결합

matrix = pd.merge( matrix, cats, on = ["item_category_id"], how = "left" ) # item_category_id를 기준으로 왼쪽 데이터프레임으로 결합

matrix["shop_city"] = matrix["shop_city"].astype(np.int8) # shop_city 정수형으로

matrix["shop_category"] = matrix["shop_category"].astype(np.int8) # shop_category를 정수형으로

matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8) # item_category_id를 정수형으로

matrix["subtype_code"] = matrix["subtype_code"].astype(np.int8) # subtype_code를 정수형으로

matrix["name2"] = matrix["name2"].astype(np.int8) # name2를 정수형으로

matrix["name3"] = matrix["name3"].astype(np.int16) # name3를 정수형으로

matrix["type_code"] = matrix["type_code"].astype(np.int8) # type_code를 정수형으로

time.time() - ts
def lag_feature( df,lags, cols ):

    for col in cols:

        print(col)

        tmp = df[["date_block_num", "shop_id","item_id",col ]]  # 데이터프레임안에 'item_cnt_month' 값을넣는다.

        for i in lags:

            shifted = tmp.copy() # tmp 데이터프레임을 복사

            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)] # item_cnt_month + _lag_ + str(i)

            shifted.date_block_num = shifted.date_block_num + i # date_block_num 값에 i값을 더하다

            print(i)

            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left') # df에 3개 열을 기준으로 왼쪽 병합

            # item_cnt_month + _lag_ + str(1) item_cnt_month + _lag_ + str(2) item_cnt_month + _lag_ + str(3) 이런식으로

    return df
ts = time.time()

matrix = lag_feature( matrix, [1,2,3], ["item_cnt_month"] )

time.time() - ts
ts = time.time()

group = matrix.groupby( ["date_block_num"] ).agg({"item_cnt_month" : ["mean"]}) # date_block_num 을 기준으로 ite_cnt_month의 평균

group.columns = ["date_avg_item_cnt"] # date_avg_item_cnt 열 추가

group.reset_index(inplace = True)



matrix = pd.merge(matrix, group, on = ["date_block_num"], how = "left") # date_block_num 기준으로 왼쪽 데이터프레임으로 결합

matrix.date_avg_item_cnt = matrix["date_avg_item_cnt"].astype(np.float16) # date_avg_item_cnt를 소수형으로

matrix = lag_feature( matrix, [1], ["date_avg_item_cnt"] )

# date_avg_item_cnt  + _lag_ + 1, date_avg_item_cnt  + _lag_ + 2, date_avg_item_cnt  + _lag_ + 3 생성 

matrix.drop( ["date_avg_item_cnt"], axis = 1, inplace = True ) # date_avg_item_cnt 제거 (사용했기 때문에)

time.time() - ts
ts = time.time()

group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']}) # 두개 열을 기준으로 item_cnt_month의 평균

group.columns = [ 'date_item_avg_item_cnt' ] # date_item_avg_item_cnt 열 생성

group.reset_index(inplace=True) # index를 원래대로 만든다



matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left') # 두개의 열을 기준으로 왼쪽 데이터프레임으로 결합

matrix.date_item_avg_item_cnt = matrix['date_item_avg_item_cnt'].astype(np.float16) # date_item_avg_item_cnt 소수형으로

matrix = lag_feature(matrix, [1,2,3], ['date_item_avg_item_cnt']) 

# date_item_avg_item_cnt  + _lag_ + 1, date_item_avg_item_cnt  + _lag_ + 2, date_item_avg_item_cnt  + _lag_ + 3 생성

matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
ts = time.time()

group = matrix.groupby( ["date_block_num","shop_id"] ).agg({"item_cnt_month" : ["mean"]})

group.columns = ["date_shop_avg_item_cnt"]

group.reset_index(inplace = True)



matrix = pd.merge(matrix, group, on = ["date_block_num","shop_id"], how = "left")

matrix.date_avg_item_cnt = matrix["date_shop_avg_item_cnt"].astype(np.float16)

matrix = lag_feature( matrix, [1,2,3], ["date_shop_avg_item_cnt"] )

matrix.drop( ["date_shop_avg_item_cnt"], axis = 1, inplace = True )

time.time() - ts
ts = time.time()

group = matrix.groupby( ["date_block_num","shop_id","item_id"] ).agg({"item_cnt_month" : ["mean"]})

group.columns = ["date_shop_item_avg_item_cnt"]

group.reset_index(inplace = True)



matrix = pd.merge(matrix, group, on = ["date_block_num","shop_id","item_id"], how = "left")

matrix.date_avg_item_cnt = matrix["date_shop_item_avg_item_cnt"].astype(np.float16)

matrix = lag_feature( matrix, [1,2,3], ["date_shop_item_avg_item_cnt"] )

matrix.drop( ["date_shop_item_avg_item_cnt"], axis = 1, inplace = True )

time.time() - ts
ts = time.time()

group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_shop_subtype_avg_item_cnt']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')

matrix.date_shop_subtype_avg_item_cnt = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1], ['date_shop_subtype_avg_item_cnt'])

matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
ts = time.time()

group = matrix.groupby(['date_block_num', 'shop_city']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_city_avg_item_cnt']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', "shop_city"], how='left')

matrix.date_city_avg_item_cnt = matrix['date_city_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1], ['date_city_avg_item_cnt'])

matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
ts = time.time()

group = matrix.groupby(['date_block_num', 'item_id', 'shop_city']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_item_city_avg_item_cnt' ]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'shop_city'], how='left')

matrix.date_item_city_avg_item_cnt = matrix['date_item_city_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1], ['date_item_city_avg_item_cnt'])

matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
ts = time.time()

group = train.groupby( ["item_id"] ).agg({"item_price": ["mean"]}) # item_id 그룹에 item_price 평균을 group에 저장

group.columns = ["item_avg_item_price"] # item_avg_item_price 열에 추가

group.reset_index(inplace = True)



matrix = matrix.merge( group, on = ["item_id"], how = "left" ) # matrix에 item_id를 기준으로 group을 결합

matrix["item_avg_item_price"] = matrix.item_avg_item_price.astype(np.float16)





group = train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )

# date_block_num을 기준으로 item_id를 정렬하고 item_id에 대한 item_price의 평균을 정렬



group.columns = ["date_item_avg_item_price"] # date_item_avg_item_price 열 추가

group.reset_index(inplace = True)



matrix = matrix.merge(group, on = ["date_block_num","item_id"], how = "left") 

# matrix에 date_block_num, item_id 기준으로 group병합



matrix["date_item_avg_item_price"] = matrix.date_item_avg_item_price.astype(np.float16)





lags = [1, 2, 3]

matrix = lag_feature( matrix, lags, ["date_item_avg_item_price"] )

# date_item_avg_item_price_lag_1, date_item_avg_item_price_lag_2, date_item_avg_item_price_lag_3 생성 



for i in lags:

    matrix["delta_price_lag_" + str(i) ] = (matrix["date_item_avg_item_price_lag_" + str(i)]- matrix["item_avg_item_price"] )/ matrix["item_avg_item_price"]

# delta값 구하기 #  기초자산의 가격변화에 대한 옵션가격의 변화량, delta_price_lag_1, delta_price_lag_2, delta_price_lag3 열 추가

    

def select_trends(row) :

    for i in lags:

        if row["delta_price_lag_" + str(i)]: 

            return row["delta_price_lag_" + str(i)]

    return 0



matrix["delta_price_lag"] = matrix.apply(select_trends, axis = 1) 

matrix["delta_price_lag"] = matrix.delta_price_lag.astype( np.float16 )

matrix["delta_price_lag"].fillna( 0 ,inplace = True) # null값을 0으로 반환



features_to_drop = ["item_avg_item_price", "date_item_avg_item_price"] 



for i in lags:

    features_to_drop.append("date_item_avg_item_price_lag_" + str(i) ) 

    features_to_drop.append("delta_price_lag_" + str(i) ) # 아까 만들었던것들 제거

matrix.drop(features_to_drop, axis = 1, inplace = True)

time.time() - ts
ts = time.time()

group = train.groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] }) # 달별 shopid의 revenue 전체값

group.columns = ["date_shop_revenue"] # date_shop_revenue 컬럼생성

group.reset_index(inplace = True)



matrix = matrix.merge( group , on = ["date_block_num", "shop_id"], how = "left" ) # 두개의 열 기준으로 group을 왼쪽으로 병합

matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32) # 소수형으로



group = group.groupby(["shop_id"]).agg({ "date_block_num":["mean"] }) # shop_id에 date_block_num 평균 group 변수에 입력

group.columns = ["shop_avg_revenue"] # shop_avg_revenue 열 추가

group.reset_index(inplace = True )



matrix = matrix.merge( group, on = ["shop_id"], how = "left" ) # shop_id를 기준으로 group 병합

matrix["shop_avg_revenue"] = matrix.shop_avg_revenue.astype(np.float32)

matrix["delta_revenue"] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']

#revenue의 delta값 구하기



matrix["delta_revenue"] = matrix["delta_revenue"]. astype(np.float32)



matrix = lag_feature(matrix, [1], ["delta_revenue"]) # delta_revenue_lag_1 열 생성

matrix["delta_revenue_lag_1"] = matrix["delta_revenue_lag_1"].astype(np.float32) # 소수형으로

matrix.drop( ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"] ,axis = 1, inplace = True) # delta_revenue_lag_1 남기고 다 지우는듯

time.time() - ts
matrix["month"] = matrix["date_block_num"] % 12 # 12로 나눈 몫구하기

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31]) # 각 월의 끝나는 일

matrix["days"] = matrix["month"].map(days).astype(np.int8)
ts = time.time()

matrix["item_shop_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id","shop_id"])["date_block_num"].transform('min')

# date_block_num - item_id, shop_id로 묶은 date_block_num의 최소값

matrix["item_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id"])["date_block_num"].transform('min')

time.time() - ts
ts = time.time()

matrix = matrix[matrix["date_block_num"] > 3] # 3을 넘는 date_block_num값들만 추출

time.time() - ts
matrix.head().T
import gc

import pickle

from xgboost import XGBRegressor

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
data = matrix.copy() # data 변수에 matrix를 복사

del matrix

gc.collect() # 가비지 콜렉션
data[data["date_block_num"]==34].shape
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1) 

Y_train = data[data.date_block_num < 33]['item_cnt_month'] # 33 아래까지는 train set

X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = data[data.date_block_num == 33]['item_cnt_month'] # 33은 vaildation set

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1) # 34는 test set
Y_train = Y_train.clip(0, 20)

Y_valid = Y_valid.clip(0, 20)
del data

gc.collect();
ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_submission.csv', index=False)
from xgboost import plot_importance



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



plot_features(model, (10,14))