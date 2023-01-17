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
        
        
os.listdir("../input/font-list")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

plt.rcParams['axes.unicode_minus']=False
fontpath="../input/font-list/NanumBarunGothic.ttf"
fontprop=font_manager.FontProperties(fname=fontpath,size=12)

data=pd.read_csv("../input/youtube-new/KRvideos.csv", engine="python") # KR 데이터 불러오기
data=data.copy()
data.info()
data.head()
cat_data=pd.read_json("../input/youtube-new/KR_category_id.json") # 카테고리 정보 불러오기
cat_items=cat_data['items']
cat_items.count()  # 카테고리 정보 정리 (ID, 카테고리 명 매핑)
for idx in range(0, cat_items.count()):
    cat_data.loc[idx,'id'] = cat_items[idx]['id']
    cat_data.loc[idx,'category'] = cat_items[idx]['snippet']['title']
cat_data=cat_data.drop(columns=['kind','etag','items'])
cat_data.info()
cat_data.head()
cat_data['id']=cat_data['id'].astype('int64')
data=pd.merge(data, cat_data, left_on='category_id', right_on='id', how='left') # data와 카테고리 정보 매핑
data.info()
data['category_id'].loc[data['id'].isnull()==True].value_counts()
data['id'].fillna(29, inplace=True)
data['category'].fillna('Nonprofits & Activism', inplace=True)
data.info()
idx=(data['video_error_or_removed']==False) & (data['ratings_disabled']==False) & (data['comments_disabled']==False)
data=data.loc[idx,:]
#data[['comments_disabled','ratings_disabled','video_error_or_removed']].describe()
data=data.drop(columns=['comments_disabled','ratings_disabled','video_error_or_removed'])  # 불필요 옵션 제거
data['video_id'].describe()
idx=(data['video_id']!='#NAME?') # 오류 video_id 제거
data=data.loc[idx,:]
data['video_id'].describe()
data['trending_date'].head()
data['trending_date']=pd.to_datetime(data['trending_date'], format='%y.%d.%m').dt.date
data['publish_time'].head()
data[['publish_date','publish_time']]=data['publish_time'].str.split('T', expand=True)
data[['publish_date','publish_time']].head()
data['publish_date']=pd.to_datetime(data['publish_date']).dt.date
data['to_trending_days']=(data['trending_date']-data['publish_date']).dt.days # 동영상이 공개되어 인기 동영상이 되기까지 기간
data['to_trending_days'].head()
data.info()
data['tags'].head()
data['tag_count']=data['tags'].apply(lambda x: len(x.split("|")) if x != '[none]' else 0) # 동영상별 테그 수 도출
data['tag_count'].head()
data['tags'].head()
data['tags']=data['tags'].str.replace(pat=r'|', repl=r' ', regex=True)
data['tags']=data['tags'].str.replace(pat=r'[^\w\s]', repl=r'', regex=True) # tag 내 특수문자 제거
data['tags'].head()
data['tag_list']=data['tags'].str.split(" ") # tag 분리
data['tag_list'].head()
data['tag_list']=data['tag_list'].apply(lambda x: list(set(x))) # tag 중복 제거
data['tag_list'].head()
video_count=data.groupby("video_id").size() # 비디오 별 인기동영상 등록 횟수
video_count=video_count.reset_index()
video_count.head()
data=pd.merge(data, video_count, on='video_id', how='left')
data.rename(columns={0:'trending_count'}, inplace=True)
data.info()
data['trending_count'].describe()
data['title_length']=data['title'].apply(lambda x: len(str(x)) if pd.isnull(x) == False else 0) # title 길이
data['title_length'].describe()
data.info()
data['desc_length']=data['description'].apply(lambda x: len(str(x)) if pd.isnull(x) == False else 0) # description 길이
data['desc_length'].describe()
data.info()
data.sort_values(by='trending_date', inplace=True)
data_dr=data.drop_duplicates(['video_id','trending_date'], keep='first') # video_id와 trending_date가 중복되는 데이터 제거
data_dr['video_id'].describe(include='all')
data_dr=data_dr.copy()
data_dr.info()
data_dr['views'].describe()
import seaborn as sns
sns.set_style('whitegrid')
sns.set()

plt.figure(figsize = (20,14))


plt.subplot(2,1,1)
views_boxplot=sns.boxplot(x="views", data=data_dr)

plt.subplot(2,1,2)
views_distplot=sns.distplot(data_dr['views'], hist=True, rug=True)

#plt.subplots_adjust(hspace = 0.8)

plt.show()
data_dr=data_dr.reset_index(drop=True)
#tag_sum=[]
#for i in range(0, data_dr['tag_list'].count()):
#    tag_sum = tag_sum + data_dr.loc[i, 'tag_list']
#count={}
#for i in tag_sum:
#    try: count[i] += 1
#    except: count[i]=1
        
count={}
    
for tags in data_dr['tag_list']:
    for tag in tags:
        if tag in count:
            count[tag] += 1
        else:
            count[tag] = 1
df_count = pd.DataFrame(count, index=['tag_count'])
df_count = df_count.T
df_count.head()
df_count=df_count.drop('none') # none 데이터 제거
tag_sample=df_count.sort_values(by=['tag_count'], ascending=False, axis=0).head(2000)
tag_sample.describe()
tag_sample.head()
tag_list=list(tag_sample.index)
tag_list
for tag in tag_list:
    tag_sample.loc[tag, 'tag_views']=data_dr.loc[data_dr['tags'].str.contains(tag),'views'].mean()
    tag_sample.loc[tag, 'tag_like']=data_dr.loc[data_dr['tags'].str.contains(tag),'likes'].mean()
    tag_sample.loc[tag, 'tag_dislike']=data_dr.loc[data_dr['tags'].str.contains(tag),'dislikes'].mean()
    tag_sample.loc[tag, 'tag_comment']=data_dr.loc[data_dr['tags'].str.contains(tag),'comment_count'].mean()
tag_sample.describe()
tag_sample.head()
from wordcloud import WordCloud

text = tag_sample['tag_count'].T.to_dict()

wordcloud = WordCloud(font_path='../input/font-list/NanumBarunGothic.ttf', \
                      background_color="white", max_words=100).generate_from_frequencies(text)

plt.figure(figsize = (10,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
text = tag_sample['tag_views'].T.to_dict()

wordcloud = WordCloud(font_path='../input/font-list/NanumBarunGothic.ttf', \
                      background_color="white", max_words=100).generate_from_frequencies(text)

plt.figure(figsize = (10,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
text = tag_sample['tag_like'].T.to_dict()

wordcloud = WordCloud(font_path='../input/font-list/NanumBarunGothic.ttf', \
                      background_color="white", max_words=100).generate_from_frequencies(text)

plt.figure(figsize = (10,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
text = tag_sample['tag_dislike'].T.to_dict()

wordcloud = WordCloud(font_path='../input/font-list/NanumBarunGothic.ttf', \
                      background_color="white", max_words=100).generate_from_frequencies(text)

plt.figure(figsize = (10,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
text = tag_sample['tag_comment'].T.to_dict()

wordcloud = WordCloud(font_path='../input/font-list/NanumBarunGothic.ttf', \
                      background_color="white", max_words=100).generate_from_frequencies(text)

plt.figure(figsize = (10,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
cat_list=cat_data['category'].tolist()

count={}
    
for cat in data_dr['category']:
    if cat in count:
        count[cat] += 1
    else:
        count[cat] = 1

cat_sample = pd.DataFrame(count, index=['cat_count'])
cat_sample = cat_sample.T
tag_list=list(cat_sample.index)
tag_list

for cat in tag_list:
    cat_sample.loc[cat, 'cat_views']=data_dr.loc[data_dr['category']==cat,'views'].mean()
    cat_sample.loc[cat, 'cat_like']=data_dr.loc[data_dr['category']==cat,'likes'].mean()
    cat_sample.loc[cat, 'cat_dislike']=data_dr.loc[data_dr['category']==cat,'dislikes'].mean()
    cat_sample.loc[cat, 'cat_comment']=data_dr.loc[data_dr['category']==cat,'comment_count'].mean()
cat_sample.head()
import seaborn as sns
sns.set_style('whitegrid')
sns.set()
cat_sample=cat_sample.sort_values(by=['cat_count'], ascending=False, axis=0)

plt.figure(figsize = (20,7))
#plt.title('카테고리 별 동영상 수',fontproperties=fontprop, size=15)
cat_bar=sns.barplot(cat_sample.index, cat_sample['cat_count'])
cat_bar.set_title("카테고리별 인기동영상 수", fontsize=100, fontproperties=fontprop)
cat_bar.set_xticklabels(cat_sample.index, rotation='vertical')
cat_bar.set_ylabel("인기동영상 수", fontproperties=fontprop)

plt.show()
cat_sample=cat_sample.sort_values(by=['cat_views'], ascending=False, axis=0)

plt.figure(figsize = (20,7))
cat_bar=sns.barplot(cat_sample.index, cat_sample['cat_views'])
cat_bar.set_title("카테고리 별 조회수 평균", fontsize=100, fontproperties=fontprop)
cat_bar.set_xticklabels(cat_sample.index, rotation='vertical')
cat_bar.set_ylabel("조회수", fontproperties=fontprop)

plt.show()
cat_sample=cat_sample.sort_values(by=['cat_like'], ascending=False, axis=0)

plt.figure(figsize = (20,7))
cat_bar=sns.barplot(cat_sample.index, cat_sample['cat_like'])
cat_bar.set_title("카테고리 별 좋아요 갯수 평균", fontsize=100, fontproperties=fontprop)
cat_bar.set_xticklabels(cat_sample.index, rotation='vertical')
cat_bar.set_ylabel("좋아요 갯수", fontproperties=fontprop)

plt.show()
cat_sample=cat_sample.sort_values(by=['cat_dislike'], ascending=False, axis=0)

plt.figure(figsize = (20,7))
cat_bar=sns.barplot(cat_sample.index, cat_sample['cat_dislike'])
cat_bar.set_title("카테고리 별 싫어요 갯수 평균", fontsize=100, fontproperties=fontprop)
cat_bar.set_xticklabels(cat_sample.index, rotation='vertical')
cat_bar.set_ylabel("싫어요 갯수", fontproperties=fontprop)

plt.show()
cat_sample=cat_sample.sort_values(by=['cat_comment'], ascending=False, axis=0)

plt.figure(figsize = (20,7))
cat_bar=sns.barplot(cat_sample.index, cat_sample['cat_comment'])
cat_bar.set_title("카테고리 별 댓글 갯수 평균", fontsize=1000, fontproperties=fontprop)
cat_bar.set_xticklabels(cat_sample.index, rotation='vertical')
cat_bar.set_ylabel("댓글 갯수", fontproperties=fontprop)

plt.show()
data_dr.info()
data_dr=data_dr.drop(columns=['publish_time','id','thumbnail_link','description'])
data_dr.info()
data_dr['category_id']=data_dr['category_id'].astype('object')
data_dr.info()
import seaborn as sns
sns.set_style('whitegrid')
sns.set()
data_dr_corr=data_dr.corr()
plt.figure(figsize = (10,7))
plt.title('상관분석',fontproperties=fontprop, size=15)
sns.heatmap(data_dr_corr,
            cmap='coolwarm', cbar=True, annot=True, square=True, fmt='.2f')
data_dr_corr['views'].sort_values(ascending=False)