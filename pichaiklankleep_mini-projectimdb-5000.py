import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

df_movie = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

df_credit = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
df_movie
df_movie.columns
df_movie.head()
#จากนั้นเราจะใช้คำสั้ง isnull เพื่อเช็คการเคลียร์ของขอมูลว่ามีข้อมูลใดบ้างที่ต้องทำการเคลียร์

df_movie.isnull().any()
#ทำการเช็คข้อมูลว่าถ้าเกิดว่ามีค่า null ขึ้นมาจะทำการเปลี่ยนให้เป็น False

df_movie = df_movie.fillna(value=0, axis=1)
#ทำการเช็คข้อมูลอีกที่ว่าทำการเคลียร์ของมูลครบถ้วนรึยัง

#ถ้าเกิดชุดข้อมูลขึ้นFalse แปลว่าถูกเคลียร์แล้ว

df_movie.isnull().any()
#คำสั่งเรียกข้อมูลประเภทหนังทั้งหมดเพราะเราต้องการดูประเภทของหนังทั้งหมด

df_movie.genres
# เนื่องจากมีการกำหนดประเภทในรูปแบบต่างๆมากเราจึงต้องแยกค่าแรกจากแต่ละประเภทโดยการแยกแยกประเภทของหนังออกมาเป็นรูปแบบ(Str)

df_movie.genres = df_movie.genres.str.split(",",n=2,expand=True)[1]

df_movie.genres
#ตรวจสอบการจำนวนข้อมูลการผิดพลาด หลังจากแยก

df_movie.genres.isnull().sum()
#1.ทำการทิ้ง (Drop) ข้อมูลที่ไม่เกี่ยวข้องออก Columns

df_movie.genres.dropna(inplace=True)
# ยืนยันว่าค่า Null ทั้งหมดถูกเติมด้วย 0

df_movie.genres.isnull().sum()
df_movie['genres'] = df_movie['genres'].apply(lambda x: x.split('"name"')[1])
df_credit
# ทำความสะอาดชุดข้อมูลก่อนโดยเฉพาะคอลัมน์นักแสดงแยกข้อมูลนักแสดงมาในรูปแบบ (Str) 

df_credit['cast'] = df_credit['cast'].str.split(":",n=6,expand=True)[6]

df_credit['cast'] = df_credit['cast'].str.split(",",n=0,expand=True)[0]

df_credit['cast']


#เมื่อเราทำการเคลียร์ข้อมูลเสร็จขั้นตอนนี้เราจะหาว่าจากปัญหาที่เราได้ทำการเขียนไว้เราต้องการชุดข้อมูลอะไรบ้างและชุดข้อมูลที่ใช้ก็จะมีอะไรบ้าง

# 1.   ชุดข้อมูลที่สรุปประเภทหนังว่า3อันดับแรกหนังประเภทไหนมาแรง

# 2.   นักแสดงที่ได้รับความนิยมมากสุด
# ลบอักขระพิเศษ ทั้งหมดออกจากข้อมูลประเภทหนัง

df_movie['genres'] =  df_movie['genres'].str.replace('"', '')

df_movie['genres'] =  df_movie['genres'].str.replace(':', '')

df_movie['genres'] =  df_movie['genres'].str.replace(']', '')

df_movie['genres'] =  df_movie['genres'].str.replace('}', '')
#ลบ white space ออกประเภทหนัง

df_movie['genres'] = df_movie['genres'].str.strip()
# ยืนยันค่าทั้งหมดว่าเหมาะสมในการใช้ข้อมูลแล้ว

df_movie['genres']
# แยกปีจากวันที่เผยแพร่

df_movie['release_date'] = pd.to_datetime(df_movie['release_date'])

df_movie['year'] = df_movie['release_date'].dt.year

df_movie.year = df_movie.year.fillna(0).astype(int)
# ยืนยันว่าคอลัมน์ ปี มีค่าที่เหมาะสม

df_movie.year
#เราใช้คำสั่งนี้เพื่อหาแค่อันดับของประเภทหนังที่ดีที่สุดมา3อันดับ

# 1.Action  2.Adventure 3.Drama

df_movie.groupby('genres').revenue.sum().nlargest(3)
# ทำการเปลียนข้อมูลเคดิดตัวเลขเป็นข้อมูลตัอักษร

df_credit['cast'] =  df_credit['cast'].str.replace('"', '')

df_credit['cast'] = df_credit['cast'].str.strip()
# รวมรายชื่อนักแสดงกับidหนังเพื่อดูว่าใครแสดงหนังเรื่องอะไร

df_combined = df_credit.merge(df_movie, left_on='movie_id', right_on='id', how='inner')
# ทำการรวมผลข้อมูลนักแสดงปีและผลคะแนนเข้าด้วยกัน

df_cast = df_combined[['cast','year','revenue']]
# โชว์ตารางผลคะแนนของนักแสดงทั้งหมด

df_combined
# ทำการเรียกดูอันดับของหนังแอคชันในทุกๆ 10 ปี ว่าติดอันดับใดบ้างในแต่ละปี

df_action = df_movie.query('genres == "Action"').groupby('year').revenue.sum()

df_action = df_action.rolling(window=10).mean()

df_action 
#  ทำการเรียกดูอันดับของหนังผจญภัยชันในทุกๆ 10 ปี ว่าติดอันดับใดบ้างในแต่ละปี

df_adventure = df_movie.query('genres == "Adventure"').groupby('year').revenue.sum()

df_adventure = df_adventure.rolling(window=10).mean()

df_adventure 
 #ทำการเรียกดูอันดับของหนังเศร้าในทุกๆ 10 ปี ว่าติดอันดับใดบ้างในแต่ละปี

df_drama = df_movie.query('genres == "Drama"').groupby('year').revenue.sum()

df_drama = df_drama.rolling(window=10).mean()

df_drama
#ทำการรวมข้อมูลของหนังแสดง กับ ปีของข้อมูลเพื่อดูว่าปีไหนนักแสดงคนไหนเป็นที่นิยม

df_cast = df_cast.ix[df_cast.groupby(['year']).revenue.idxmax()]
#เรียกดูรายชื่อนักแสดง ที่ทำการคลีนข้อมูลแล้ว

df_cast
# เรียกข้อมูลของนักแสดงทีได้รับความนิยมมากที่สุด10 อันดับแรก โดยการร่วมค่านักแสดงกับ ปี

df_cast = df_cast.groupby('cast').year.count()

df_cast  = df_cast [df_cast >1]

df_cast
# โชว์กราฟโชว์ค่าเฉลี่ยของประเภทหนัง Top3 ในทุกๆ ปี ก็จะมี Action , Adventure , Drama

df_action.dropna(inplace=True)

plt.subplots(figsize=(10, 6))

plt.bar(df_adventure .index, df_adventure )

plt.title('10 Years moving average for category - Action')

plt.xlabel('Year')

plt.ylabel('Moving average');

df_adventure .dropna(inplace=True)

plt.subplots(figsize=(10, 6))

plt.bar(df_adventure .index, df_adventure )

plt.title('10 Years moving average for category Adventure')

plt.xlabel('Year')

plt.ylabel('Moving average');

df_drama.dropna(inplace=True)

plt.subplots(figsize=(10, 6))

plt.bar(df_drama.index, df_drama)

plt.title('10 Years moving average for category Drama')

plt.xlabel('Year')

plt.ylabel('Moving average');

plt.style.use('ggplot');
# โชว์กราฟค่าความนิยมของนักแสดงที่นิยมมากสุด10อันดับ

df_cast.plot(kind='bar', stacked=True, colormap='PiYG')