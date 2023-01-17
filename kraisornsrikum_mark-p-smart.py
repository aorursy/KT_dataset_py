import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df_anime=pd.read_csv('../input/anime.csv') #นำข้อมูล anime.csv ไปเก็บ df_anime
df_rate=pd.read_csv('../input/rating.csv') #นำข้อมูล anime.csv ไปเก็บ df_rate
#แสดงตารางข้อมูล df_anime
df_anime.head()
#แสดง 5 อันดับท้ายสุดของข้อมูล
df_anime.tail()
#แสดงข้อมูลตัวเลขทั้งหมด
df_anime.describe()
#ตรวจสอบชุดข้อมูลว่ามีความสมบูรณ์รึป่าว
df_anime.isna().any()
#แสดงประเภทชุดข้อมูล
df_anime.dtypes
#แสดงข้อมูลประเภท INT64 ว่ามีชุดข้อมูลเท่าไหร่
df_anime.isnull().sum()
#ต้องการที่จะลบข้อมูลในคอลัมน์ rating genre type 
df_anime.dropna(subset=['rating','genre','type'],inplace=True)
#หลังจากที่ลบข้อมูล ก็เช็คข้อมูลอีกครั้ง
df_anime.isnull().sum()
#แสดงชุดประเภทข้อมูล
df_anime.dtypes
df_anime=df_anime[df_anime["episodes"]!='Unknown']
df_anime['episodes']=df_anime['episodes'].astype('int64')
#แสดงข้อมูลตาราง
df_anime.head()
#แสดงประเภทข้อมูลของอนิเมะ
df_anime.type.unique()
#คำสั่งเลือกข้อมูลประเภทหนัง
df_movies=df_anime[df_anime["type"]=='Movie']
#แสดงข้อมูลประภทหนัง
df_movies.head()
#แสดงค่าประเภท members,rating ของประเภท moive
df_movies.sort_values(['members','rating'],ascending=False)
plt.figure(figsize = (16,5))
df_check=df_movies.corr()
sns.heatmap(df_check, annot=True, fmt="g", cmap='viridis')
plt.show()
#ลบข้อมูลประเภท หนัง ลบ ตอน
del df_movies['episodes']
#ลบข้อมูลประเภท หนัง ลบ ไอดีของหนัง
del df_movies['anime_id']
#ลบข้อมูลประเภท หนัง ลบ ประเภท
del df_movies['type']
#แสดงข้อมูลประเภทหนัง
df_movies
plt.figure(figsize = (16,5))
df_check=df_movies.corr()
sns.heatmap(df_check, annot=True, fmt="g", cmap='viridis')
plt.show()
#แสดงข้อมูลจากประเภทหนัง ตัวเลือก สมาชิก
#แสดงข้อมูลจากประเภทหนัง ตัวเลือก คะแนนน่าสนใจ
print(df_movies['members'].mean())
print(df_movies['rating'].mean())
def f(row):
    val=0
    if row['rating']>=9.5:
        val=1
    elif row['rating']>=9:
        val=2
    elif row['rating']>=8.5:
        val=3
    elif row['rating']>=8:
        val=4
    elif row['rating']>=7.5:
        val=5
    elif row['rating']>=7:
        val=6
    elif row['rating']>=6.5:
        val=7
    return val

df_movies['hexcode'] = df_movies.apply(f, axis=1)
#แสดงตารางข้อมูลประเภทหนัง
df_movies.head()
df_considerable_movies=df_movies[(df_movies['members']>df_movies['members'].mean()) & (df_movies['rating']>df_movies['rating'].mean())]
df_considerable_movies.head()
#จะแสดงกราฟแบบจุด โดยแกน x คือ Rating แกน y คือ Audience
colordict={1:'#30E6B1',2:'#A1F5AD',3:'#E4B06E',4:'#EA4052',5:'#42C6CF',6:'#C6D359',7:'#45B39C'} 
plt.figure(figsize=(16, 8))
plt.scatter(df_considerable_movies['rating'],df_considerable_movies['members'],s=df_considerable_movies['rating']**2,alpha=0.8,c=df_considerable_movies['hexcode'])
plt.xlabel('Ratings')
plt.ylabel('Audience')
plt.show()
#จะแสดงกราฟแบบจุด โดยแกน x คือ Rating แกน y คือ Members และมีการเปลี่ยนสีจุด
plt.figure(figsize=(20, 8))
sns.lmplot('rating','members', data=df_considerable_movies, hue='hexcode',fit_reg=False,size=8)
plt.show()
#แสดงข้อมูลแบบตัวเลข ของ user_id anime_id rating
df_rate.describe()
#ดึงข้อมูลประเภทหนัง
df_movies=df_anime[df_anime["type"]=='Movie']
df_movies_rec=pd.merge(df_movies,df_rate,on='anime_id')
#หาค่าเฉลี่ยของrating_x
mainmean=df_movies_rec['rating_x'].mean()
print(mainmean)
#แสดงข้อมูลตารางของอนิเมะ
df_movies_rec.head()
#แสดงข้อมูลตัวเลข ของ anime_id episodes rating_x members user_id rating_y
df_movies_rec.describe()
#แสดงข้อมูลเรทติ้งในรูปแบบแกน x,y โดยเป็นประเภท float64
df_movies_rec.dtypes
df_movies_rec['rating_x']=df_movies_rec['rating_x'].astype('float64')
df_movies_rec['rating_y']=df_movies_rec['rating_y'].astype('float64')
#แสดงความยาวข้อมูล
df_movies_rec.name.unique()
print(len(df_movies_rec))
#แสดงจำนวนของ rating ที่เป็นแกรน Y 
count_node=pd.DataFrame(df_movies_rec.groupby('name')['rating_y'].mean())
count_node['num of ratings'] = pd.DataFrame(df_movies_rec.groupby('name')['rating_y'].count())
#แสดงค่าของ rating_y และ จำนวนการโหวดrating_y และ แสดงผลของ rating_y ทั้งหมด
count_node.sort_values(['rating_y'],ascending=False)
count_node.head()
#เพิ่มจำนวนการใส่อันดับหน้าเรื่อง
count_node.reset_index(inplace=True)
#แสดงข้อมูลตาราง
count_node.head()
#แสดงข้อมูลตัวเลข rating บอกเป็นช่วง
count_node.describe()
#แสดงจำนวนของเรทติ้ง 18,5
plt.figure(figsize=(18,5))
count_node['num of ratings'].hist(bins=300,color='g',alpha=0.6)
plt.title('NUMBER OF RATING DISTRIBUTION OVER THE COUNT')
plt.xlim(0,1600)
plt.show()
#แสดงจำนวนของเรทติ้ง 18,5
plt.figure(figsize=(18,5))
count_node['rating_y'].hist(bins=30,color='y',alpha=0.4)
plt.title('RATING DISTRIBUTION OVER THE COUNT')
plt.show()
# 
genre_sets=set()
for i in df_movies['genre']:
    k=i.split(',')
    for i in k:
        genre_sets.add(str(i))
#แสดงจำนวนของเรทติ้ง 20,5
plt.figure(figsize=(20,5))
sns.jointplot(x='rating_y',y='num of ratings',data=count_node,color='#2ecc71',size=10,kind='reg')
plt.show()
#แสดง ชื่อเรื่องและแนว
df_genre_mov =pd.DataFrame(df_movies[['name','genre']])
#แสดงจำนวนลำดับข้อมูล
df_genre_mov.reset_index(inplace=True)
df_genre_mov.head()
df_genre_mov.set_index('name',inplace=True)
#merge 2 table โดยใช้ anime_id เป็น key หลัก
df_main=pd.merge(df_rate,df_movies,on='anime_id')
df_main.head(1)
#คำสั่งลบ anime_id
del df_main['anime_id']
#คำสั่งลบ rating_x
del df_main['rating_x']
#คำสั่งลบ genre
del df_main['genre']
#คำสั่งลบ members
del df_main['members']
#คำสั่งลบ type
del df_main['type']
#คำสั่งลบ episodes
del df_main['episodes']
#แสดงข้อมูลตัวแรก
df_main.head(1)
def rec_mov(name_):
    rating_name_=count_node[count_node['name']==name_]
    #print(rating_name_)
    
    #เอาประเภทเรื่องมาแยกเล้วใส่ในlist
    k_genre_list=df_genre_mov.loc[name_]['genre'].split(',')
    for i in range(len(k_genre_list)):
        k_genre_list[i]=k_genre_list[i].strip()
        
    #สร้างtable ของเรื่องที่จะแนะนำ โดยเรื่องที่แนะนำจะเป็นประเภทเดียวที่อยู่ในlist
    df_suggest=df_genre_mov[df_genre_mov.isin(k_genre_list)]
    df_suggest.dropna(subset=['genre'],inplace=True)
    df_suggest.reset_index(inplace=True)
    df_joint=pd.merge(df_suggest,count_node,on='name')
    df_joint['rating_y']=df_joint['rating_y'].astype('float64')
    check=float(rating_name_['rating_y'])
    df_joint_1=df_joint.copy()
    del df_joint_1['genre']
    del rating_name_['name']
    del df_joint_1['index']
   #print(df_joint_1)
    df=df_main.pivot_table(index='user_id',columns='name',values='rating_y')
    
    #หาเรื่องที่คล้ายกับตัวแปร name_
    similar_df=df.corrwith(df[name_])
    #สร้าง data frame ที่มีคอลั่มเพิ่มมาจาก similar_df คือ Correlation  โดย Correlation เป็นค่าความเหมือน
    corr_df = pd.DataFrame(similar_df, columns=['Correlation'])
    corr_df.dropna(inplace=True)
    corr_df.reset_index(inplace=True)
    corr_df=corr_df.merge(count_node,on='name')
   #print(corr_df.head(3))

    #ให้ table แสดงเรื่องที่แนะนำโดยเรียงจะratingมากสุดไปน้อยสุด ลบ 3 คอลั่ม Correlation, num of ratings, rating_y return 10 ตัวแรก
    show_recommendation=corr_df[corr_df['num of ratings'] > 100].sort_values(by=['Correlation','num of ratings'], ascending=False).head(10)
    del show_recommendation['Correlation']
    del show_recommendation['num of ratings']
    del show_recommendation['rating_y']
    return show_recommendation.head(10)
    
# set library pandas 
def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None
    # pd.options.display.precision = 2  # set as needed

set_pandas_options()
#แนะนำเรื่องที่ใกล้เคียงกับเรื่อง Kimi no Na wa.
rec_mov('Kimi no Na wa.')
#แนะนำเรื่องที่ใกล้เคียงกับเรื่อง Koe no Katachi
rec_mov('Koe no Katachi')
#แนะนำเรื่องที่ใกล้เคียงกับเรื่อง .hack//G.U. Trilogy
rec_mov('.hack//G.U. Trilogy')
#แนะนำเรื่องที่ใกล้เคียงกับเรื่อง Ookami Kodomo no Ame to Yuki
rec_mov('Ookami Kodomo no Ame to Yuki')

