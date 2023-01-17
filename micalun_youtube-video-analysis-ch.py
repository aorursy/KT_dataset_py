import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/USvideos.csv')
df.info()  # 查看数据基本信息：description中有缺失，不使用可忽略。category_id为int，要转换成str。trending_date, publish_date时间格式需要标准化。
df['title'].duplicated().any() # title有重复值，需要去重复
# 根据title去重复（有些视频是rewind版，保留最后一天trending的记录）
df = df[~df['title'].duplicated(keep='last')] 
df['title'].duplicated().any()
# 修改时间为标准格式
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['publish_time'] = pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ') 

df['publish_date'] = df['publish_time'].dt.date # 获得publish日期
df['publish_times']=df['publish_time'].dt.time # 获得publish时间
df['trending_days'] = df['trending_date'].dt.date # 获得trending日期
# 提取年月日
df["publish_Year"]=df["publish_date"].apply(lambda time : time.year) 
df["publish_Month"]=df["publish_date"].apply(lambda time : time.month) 
df["publish_Day"]=df["publish_date"].apply(lambda time : time.day)
df["publish_Hour"]=df["publish_time"].apply(lambda time : time.hour) 

# 定位发布日期的周数
df["trending_day_of_week"]=df["trending_date"].apply(lambda time:time.dayofweek)
df['publish_day_of_week'] = df['publish_time'].apply(lambda time:time.dayofweek)

# 获取星期
day_map= {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df["publish_day_of_week2"]=df["publish_day_of_week"].map(day_map)
df["trending_day_of_week2"]=df["trending_day_of_week"].map(day_map)
# 获取视频上架时间跨度
df['date_gap'] =df['trending_days'].values - df['publish_date'].values  # 从publish到trending时间间隔
df['date_gap']= df['date_gap'].dt.days  # 将timedelta转换成int

# 新特征构造
df['views_perday'] = df['views']/df['date_gap']  #构造特征日观看量：总观看量/ date_gap
df['likes_perday']= df['likes']/df['date_gap']   # 日点赞数
df['dislikes_perday']= df['dislikes']/df['date_gap']  #日点踩数
df['comment_perday'] = df['comment_count'] / df['date_gap'] # 日评论数

df['likes_to_views']= df['likes']/df['views']   # 已观看量中点赞率
df['dislikes_to_views']= df['dislikes']/df['views']  # 已观看量中点踩率
df['comment_to_views'] = df['comment_count'] / df['views']  # 已观看中评论率

df['likes_to_dislikes'] = df['likes']/df['dislikes'] # likes与dislikes比率
df['likes_to_comment'] = df['likes']/ df['comment_count'] # likes与comment比率
df['dislikes_to_comment'] = df['dislikes'] / df['comment_count'] # dislikes与comment比率

# 从json文件获取对应category_id的类别名
id_to_category = {} 
with open('../input/US_category_id.json', 'r') as f:    
    data = json.load(f)   
    for category in data['items']:       
        id_to_category[category['id']] = category['snippet']['title'] 

df['category_id']= df['category_id'].astype(str) # 将int类型转换成str
df['category']= df['category_id'].map(id_to_category) # 增加一列，存放对应分类号的类别名
df.head(3)
# 获得最大观看量、点赞数、点踩数的视频排名
def visualize_max(df, column, num=10): 
    max_df = df.sort_values(column, ascending=False).iloc[:num]
  
    fig,ax = plt.subplots()
    ax.barh(max_df['title'],max_df[column])

    ax.set( xlabel=column,title ='Top %s %s of video' %(num,column))
   
  
visualize_max(df,'likes',20)
visualize_max(df,'views',20)
visualize_max(df,'dislikes',20)
visualize_max(df,'comment_count',20)
# 各频道视频上传量排名
df_channel= df.groupby('channel_title').size().sort_values(ascending=True)  
plt.figure(figsize=(8,8))
df_channel[-20:].plot(kind='barh')
# 各频道观看量排名
df_channel_views = df.groupby('channel_title')['views'].sum().sort_values(ascending=True)
plt.figure(figsize=(8,8))
df_channel_views[-20:].plot(kind='barh')
# 类别与上传日期对观看量的影响
category_publishday_hm=df.groupby(["category","publish_Day"])['views'].sum()
cat_pub_hm = category_publishday_hm.unstack().fillna(0)

plt.figure(figsize=(12,10))
sns.heatmap(cat_pub_hm)
plt.title("category vs publish date",color='red', fontsize='large')
# 类别与上传星期对观看量的影响
category_weekday_hm=df.groupby(["category","publish_day_of_week2"])['views'].sum()
cat_pubweek_hm = category_weekday_hm.unstack().fillna(0)

plt.figure(figsize=(12,10))
sns.heatmap(cat_pubweek_hm,annot=True)
plt.title("category vs publish_day_of_week",color='red', fontsize='large')
# 月份与星期对视频上传量的影响
month_vs_dayofweek = df.groupby(['publish_Month','publish_day_of_week2']).count()
month_vs_dayofweek_hm=month_vs_dayofweek['video_id'].unstack().fillna(0)
plt.figure(figsize=(8,8))
sns.heatmap(month_vs_dayofweek_hm)
plt.title("publish_month vs publish_day_of_week",color='red', fontsize='large')

month_vs_dayofweek_view = df.groupby(['publish_Month','publish_day_of_week2'])['views'].sum()
month_vs_dayofweek_view_hm=month_vs_dayofweek_view.unstack().fillna(0)
plt.figure(figsize=(8,8))
sns.heatmap(month_vs_dayofweek_view_hm)
plt.title("publish_month vs publish_day_of_week",color='red', fontsize='large')
# 点赞、点踩和评论相关性分析
corr = df[['likes_to_views','dislikes_to_views','comment_to_views','likes_to_dislikes','likes_to_comment','dislikes_to_comment']].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot = True)