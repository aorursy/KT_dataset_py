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
df = pd.read_csv('../input/2018_death_data.csv',encoding='euc-kr')
df.head()
df.info()
df.isnull().sum()
df = df[['발생년월일시','사망자수','주야','요일','발생지시도','발생지시군구','경도','위도']]
df.info()
t_map = {

    '주' : 0,

    '야' : 1

}



d_map = {

    '월' : 1,

    '화' : 2,

    '수' : 3,

    '목' : 4,

    '금' : 5,

    '토' : 6,

    '일' : 7

}



r_map = {

    '강원' : 'Gangwon',

    '경기' : 'Gyeonggi',

    '경남' : 'Gyeongsangnam',

    '경북' : 'Gyeongsanbuk',

    '광주' : 'Gwangju',

    '대구' : 'Daegu',

    '대전' : 'Daejeon',

    '부산' : 'Busan',

    '서울' : 'Seoul',

    '세종' : 'Sejong',

    '울산' : 'Ulsan',

    '인천' : 'Incheon',

    '전남' : 'Jeollanam',

    '전북' : 'Jeollabuk',

    '제주' : 'Jeju',

    '충남' : 'Chungcheongnam',

    '충북' : 'Chungcheongbuk'

}
df['주야'] = df['주야'].map(t_map)

df['요일'] = df['요일'].map(d_map)

df['발생지시도'] = df['발생지시도'].map(r_map)

df.head()
df['발생년월일시'] = df['발생년월일시'].apply(lambda x: pd.to_datetime(str(x),format='%Y%m%d%H'))
df.info()
df['발생년월'] = df['발생년월일시'].dt.strftime('%Y%m')

df.drop(['발생년월일시'],axis=1, inplace=True)
df.info()
df.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.set_style('darkgrid')
df_time = df.groupby(['발생년월','주야'],as_index=False).sum()

df_time.head()
plt.figure(figsize=(13,6))

plt.title('timezone graph')

sns.lineplot(x='발생년월', y='사망자수',data=df_time, hue='주야')

sns.scatterplot(x='발생년월', y='사망자수',data=df_time, hue='주야')

plt.xlabel('month')

plt.ylabel('count')
df_day = df.groupby(['요일'],as_index=False).sum()

df_day.head()
plt.figure(figsize=(13,6))

plt.title('timezone graph')

sns.barplot(x='요일', y='사망자수',data=df_day)

plt.xlabel('day')

plt.ylabel('count')
df_region = df.groupby(['발생지시도','주야'],as_index=False).sum()

df_region.head()
plt.figure(figsize=(25,6))

plt.title('region graph')

sns.barplot(x='발생지시도', y='사망자수',data=df_region, hue='주야')

plt.xlabel('region')

plt.ylabel('count')
import folium
folium.Map(location=[36.29,127.16],zoom_start=6)
folium.Map(location=[36.4,127.16],zoom_start=6.5)
df.경도.head()
accident_circle = folium.Map(location=[36.4,127.16],zoom_start=6.5)

for lat,lng,num in zip(df.위도, df.경도,range(3000)):

    folium.CircleMarker([lat,lng],radius=1,fill=True).add_to(accident_circle)

accident_circle