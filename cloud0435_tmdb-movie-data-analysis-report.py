# 导入包
import numpy as np
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.misc import imread
movies = pd.read_csv('../input/tmdb_5000_movies.csv')
credits = pd.read_csv('../input/tmdb_5000_credits.csv')
print('电影数据信息集：', movies.shape, '\n演员数据信息集：', credits.shape, '\n')
print((movies['id'] == credits['movie_id']).shape[0])
print()
print((movies['title'] == credits['title']).shape[0])
print()
print(movies.columns)
print()
print(credits.columns)
del credits['movie_id']
del credits['title']
del movies['homepage']
del movies['spoken_languages']
del movies['original_language']
del movies['original_title']
del movies['tagline']
del movies['overview']
del movies['status']
full = pd.concat([movies, credits], axis=1)
full.tail(20)
full.isnull().sum()
full.loc[full['runtime'].isnull(), :]
full.loc[2056, 'runtime'] = 94
full.loc[4140, 'runtime'] = 81
full.loc[[2056, 4140], :]
full.loc[full['release_date'].isnull(), :]
full.loc[4553, 'release_date'] = '2014-06-01'
full.loc[4553, :].values.reshape(-1)
year = full.release_date.map(lambda x: re.compile('^\d+').search(x).group())
full['release_date'] = year
full
full.head(5)
# 提取有用的信息(‘name’字段)
cols = ['genres', 'keywords', 'production_companies', 'production_countries', 'cast', 'crew']    
    
for i in cols:
    full[i] = full[i].apply(json.loads)
    
def get_names(x):
    return ','.join([i['name'] for i in x])

for i in cols:
    full[i] = full[i].apply(get_names)

full.head()
genre_set = set()
for x in full['genres']:
    genre_set.update(x.split(','))
genre_set.discard('')
genre_set
# 对各种电影风格genre，进行one-hot编码。
genre_df = pd.DataFrame()

for genre in genre_set:
    #如果一个值中包含特定内容，则编码为1，否则编码为0
    genre_df[genre] = full['genres'].str.contains(genre).map(lambda x:1 if x else 0)

#将原数据集中的year列，添加至genre_df
genre_df['year']=full['release_date']

genre_df.head()
#将genre_df按year分组，计算每组之和。
genre_by_year = genre_df.groupby('year').sum()  #groupby之后，year列通过默认参数as_index=True自动转化为df.index

genre_by_year.head()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.figure(figsize=(18,12))
plt.plot(genre_by_year)  

plt.xlabel('Year', fontsize=10)
plt.xticks(rotation = '45')
plt.ylabel('Film count', fontsize=12)
plt.title('genres change over years',fontsize=18)
plt.legend(genre_by_year)

plt.show()
genresum_by_year = genre_by_year.sum().sort_values(ascending=False)
genresum_by_year
plt.figure(figsize = (15,10))
plt.subplot(111)
genresum_by_year.plot(kind = 'barh')
plt.xlabel('数量')
plt.title('电影类型的排名')
plt.show()
#增加收益列
full['profit'] = full['revenue']-full['budget']

profit_df = pd.DataFrame()
profit_df = pd.concat([genre_df.iloc[:,:-1],full['profit']], axis=1)

#创建一个Series，其index为各个genre，值为按genre分类计算的profit之和
profit_by_genre = pd.Series(index=genre_set)
for genre in genre_set:
    profit_by_genre.loc[genre]=profit_df.loc[:,[genre,'profit']].groupby(genre, as_index=False).sum().loc[1,'profit']
profit_by_genre

#创建一个Series，其index为各个genre，值为按genre分类计算的budget之和
budget_df = pd.concat([genre_df.iloc[:,:-1],full['budget']],axis=1)
budget_df.head(5)
budget_by_genre = pd.Series(index=genre_set)
for genre in genre_set:
    budget_by_genre.loc[genre]=budget_df.loc[:,[genre,'budget']].groupby(genre,as_index=False).sum().loc[1,'budget']
budget_by_genre

#横向合并数据框
profit_rate = pd.concat([profit_by_genre, budget_by_genre],axis=1)
profit_rate.columns=['profit','budget']
profit_rate

#添加收益率列。乘以100是为了方便后续可视化以百分比显示坐标轴刻度标签
profit_rate['profit_rate'] = (profit_rate['profit']/profit_rate['budget'])*100
profit_rate.sort_values(by=['profit','profit_rate'], ascending=False, inplace=True)  
profit_rate
#可视化不同风格电影的收益（柱状图）和收益率（折线图）
fig = plt.figure(figsize=(18,13))
profit_rate['profit'].plot(kind = 'barh', label = 'profit', alpha = 0.7)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('Profit by genres', fontsize=20)
plt.xlabel('Film Profit',fontsize=18)
plt.ylabel('Genre',fontsize=18)
plt.xlim(0,1.2e11)
plt.legend(loc='best', fontsize=20)
#创建受欢迎程度数据框
popu_df = pd.DataFrame()
popu_df = pd.concat([genre_df.iloc[:,:-1], full['popularity']], axis=1)
popu_df.head()

#计算每个风格电影的受欢迎程度的均值
popu_mean_list=[]
for genre in genre_set:
    popu_mean_list.append(popu_df.loc[:,[genre,'popularity']].groupby(genre, as_index=False).mean().loc[1,'popularity'])

popu_by_genre = pd.DataFrame(index=genre_set)
popu_by_genre['popu_mean'] = popu_mean_list
popu_by_genre.sort_values('popu_mean',inplace=True)
popu_by_genre

#可视化不同风格电影的平均受欢迎程度
fig = plt.figure(figsize=(15,12))
ax = plt.subplot(111)
popu_by_genre.plot(kind='barh', alpha=0.7, color= 'r', ax =ax)
plt.title('Popularity by genre', fontsize=20)
plt.ylabel('Film genres',fontsize=15)
plt.xlabel('Mean of popularity', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()
#创建平均评分数据框
vote_avg_df = pd.concat([genre_df.iloc[:,:-1], full['vote_average']],axis=1)
vote_avg_df.head(2)

#计算不同风格电影的平均评分
voteavg_mean_list=[]
for genre in genre_set:
    voteavg_mean_list.append(vote_avg_df.groupby(genre,as_index=False).mean().loc[1,'vote_average'])
    
#形成目标数据框
voteavg_mean_by_genre = pd.DataFrame(index=genre_set)
voteavg_mean_by_genre['voteavg_mean']=voteavg_mean_list

#排序
voteavg_mean_by_genre.sort_values('voteavg_mean',ascending=False,inplace=True)

#可视化不同风格电影的平均评分
fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111)
voteavg_mean_by_genre.plot(kind='bar', ax=ax)
plt.title('vote_average by genre',fontsize=15)
plt.xlabel('genre',fontsize=13)
plt.ylabel('vote_average',fontsize=13)
plt.xticks(rotation=45)
plt.axhline(y=6.75, color='r')
plt.axhline(y=5.75, color='r')
plt.legend(fontsize=13)
plt.ylim(5,7,0.5)
plt.show()
# 不同风格电影的平均评分相差并不悬殊，最高与最低只差了1分有余。
# #计算相关系数矩阵
corr_df = pd.concat([genre_df.iloc[:,:-1], full[['popularity','vote_average','vote_count','budget','revenue']]],axis=1)
corrDf = corr_df.corr()
corrDf['revenue'].sort_values(ascending = False)

# 可以看到相关性最高的就是vote_count, budget预算（成本）， popularity 。
#创建票房收入数据框
revenue = full[['popularity','vote_count','budget','revenue']]

#可视化票房收入分别与受欢迎度（蓝）、评价次数（绿）、电影预算（红）的相关性散点图，并配线性回归线。
fig = plt.figure(figsize=(17,5))
ax1 = plt.subplot(1,3,1)
ax1 = sns.regplot(x='popularity', y='revenue', data=revenue, x_jitter=.1)  # x_jitter随机噪声
ax1.text(400,3e9,'r=0.64',fontsize=15)
plt.title('revenue by popularity',fontsize=15)
plt.xlabel('popularity',fontsize=13)
plt.ylabel('revenue',fontsize=13)

ax2 = plt.subplot(1,3,2)
ax2 = sns.regplot(x='vote_count', y='revenue', data=revenue, x_jitter=.1,color='g',marker='+')
ax2.text(5800,2.2e9,'r=0.78',fontsize=15)
plt.title('revenue by vote_count',fontsize=15)
plt.xlabel('vote_count',fontsize=13)
plt.ylabel('revenue',fontsize=13)

ax3 = plt.subplot(1,3,3)
ax3 = sns.regplot(x='budget', y='revenue', data=revenue, x_jitter=.1,color='r',marker='^')
ax3.text(1.6e8,2.2e9,'r=0.73',fontsize=15)
plt.title('revenue by budget',fontsize=15)
plt.xlabel('budget',fontsize=13)
plt.ylabel('revenue',fontsize=13)
plt.show()
#创建关于原创性的数据框
orginal_novel = pd.DataFrame()
orginal_novel['keywords'] = full['keywords'].str.contains('based on').map(lambda x:1 if x else 0)
orginal_novel[['revenue','budget']]=full[['revenue','budget']]
orginal_novel['profit']=full['revenue']-full['budget']
orginal_novel1 = orginal_novel.groupby('keywords',as_index=False).mean()
orginal_novel

#创建原创与改编对比数据框
org_vs_novel = pd.DataFrame()
org_vs_novel['count'] = [full.shape[0]-full['keywords'].str.contains('based on').sum(),
                        full['keywords'].str.contains('based on').sum()]
org_vs_novel['profit']=orginal_novel1['profit']
org_vs_novel.index=['orginal works','based on novel']
org_vs_novel

#可视化原创与改编电影的数量占比（饼图），和片均收益（柱状图）
fig = plt.figure(figsize=(20,6))
ax1 = plt.subplot(131)
# autopct，圆里面的文本格式, startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看.
# pctdistance，百分比的text离圆心的距离  labeldistance，文本的位置离原点有多远，1.1指1.1倍半径的位置
ax1 = plt.pie(org_vs_novel['count'], labels=org_vs_novel.index, autopct='%.2f%%', startangle=90, pctdistance=0.6)
plt.title('Film quantities Comparison\nOriginal works VS based on novel',fontsize=13)

ax2 = plt.subplot(132)
ax2 = org_vs_novel['profit'].plot.bar()
plt.xticks(rotation=0)
plt.ylabel('Profit',fontsize=12)
plt.title('Profit Comparison\nOriginal works VS based on novel',fontsize=13)


ax3 = plt.subplot(133)
c = pd.concat([orginal_novel['keywords'], full['release_date']], axis=1)
c.rename(columns = {'release_date':'year'}, inplace=True)
b = pd.get_dummies(c, columns=['keywords'])
b.rename(columns={'keywords_0':'orginal works', 'keywords_1':'based on novel'}, inplace=True)
b.groupby('year').sum().plot(kind='line', alpha=0.7, ax=ax3)
plt.title('org_vs_novel by year', fontsize=15)

plt.show()
# 可以看出，虽然改编电影的数量相对较少，但是平均每部改编电影的收益却很高，且改编电影是近年来才有上升的。
#创建公司数据框
company_list = ['Universal Pictures', 'Paramount Pictures']
company_df = pd.DataFrame()
for company in company_list:
    company_df[company]=full['production_companies'].str.contains(company).map(lambda x:1 if x else 0)
company_df = pd.concat([company_df,genre_df.iloc[:,:-1],full['revenue']],axis=1)


#创建巨头对比数据框
Uni_vs_Para = pd.DataFrame(index=['Universal Pictures', 'Paramount Pictures'],columns=company_df.columns[2:])

#计算二公司收益总额
Uni_vs_Para.loc['Universal Pictures']=company_df.groupby('Universal Pictures',as_index=False).sum().iloc[1,2:]
Uni_vs_Para.loc['Paramount Pictures']=company_df.groupby('Paramount Pictures',as_index=False).sum().iloc[1,2:]
Uni_vs_Para

#可视化二公司票房收入对比
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
Uni_vs_Para['revenue'].plot(ax=ax,kind='bar')
plt.xticks(rotation=0)
plt.title('Universal VS. Paramount')
plt.ylabel('Revenue')
plt.grid(True)
# Universal Pictrues总票房收入高于Paramount Pictures
#转置
Uni_vs_Para = Uni_vs_Para.T

#拆分出二公司数据框
universal = Uni_vs_Para['Universal Pictures'].iloc[:-1]
paramount = Uni_vs_Para['Paramount Pictures'].iloc[:-1]

#将数量排名9之后的加和，命名为others
universal['others']=universal.sort_values(ascending=False).iloc[8:].sum()
universal = universal.sort_values(ascending=True).iloc[-9:]


#将数量排名9之后的加和，命名为others
paramount['others']=paramount.sort_values(ascending=False).iloc[8:].sum()
paramount = paramount.sort_values(ascending=True).iloc[-9:]


#可视化二公司电影风格数量占比
fig = plt.figure(figsize=(13,6))
ax1 = plt.subplot(1,2,1)
ax1 = plt.pie(universal, labels=universal.index, autopct='%.2f%%', startangle=90, pctdistance=0.75)
plt.title('Universal Pictures',fontsize=15)

ax2 = plt.subplot(1,2,2)
ax2 = plt.pie(paramount, labels=paramount.index, autopct='%.2f%%', startangle=90, pctdistance=0.75)
plt.title('Paramount Pictures',fontsize=15)

# 两家公司的主要电影类型几乎一致：喜剧类（Comedy）、戏剧类（Drama）、惊悚类（Thriller）、动作类（Action）