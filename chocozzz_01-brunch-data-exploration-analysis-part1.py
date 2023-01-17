import pickle
import pandas as pd
import numpy as np
import os, sys, gc 
from plotnine import *
import plotnine

from tqdm import tqdm_notebook
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib import rc
import re
from matplotlib.ticker import PercentFormatter
import datetime
from math import log # IDF 계산을 위해
%config InlineBackend.figure_format = 'retina'
mpl.font_manager._rebuild()

fontpath = '../input/t-academy-recommendation/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9).get_name()

mpl.pyplot.rc('font', family=font)
plt.rc('font', family=font)
plt.rcParams['font.family'] = font
warnings.filterwarnings(action='ignore')
path = "../input/t-academy-recommendation/"
print(os.listdir(path))
# pd.read_json : json 형태의 파일을 dataframe 형태로 불러오는 코드 
magazine = pd.read_json(path + 'magazine.json', lines=True) # lines = True : Read the file as a json object per line.
metadata = pd.read_json(path + 'metadata.json', lines=True)
users = pd.read_json(path + 'users.json', lines=True)
import itertools
from itertools import chain
import glob
import os 

input_read_path = path + '/read/'
# os.listdir : 해당 경로에 있는 모든 파일들을 불러오는 명령어 
file_list = os.listdir(input_read_path)
print(file_list[0:2])
%%time 
read_df_list = []
exclude_file_lst = ['read.tar', '.2019010120_2019010121.un~']
for file in tqdm_notebook(file_list):
    # 예외처리 
    if file in exclude_file_lst:
        continue 
    else:
        file_path = input_read_path + file
        df_temp = pd.read_csv(file_path, header=None, names=['raw'])
        # file명을 통해서 읽은 시간을 추출(from, to)
        df_temp['from'] = file.split('_')[0]
        df_temp['to'] = file.split('_')[1]
        read_df_list.append(df_temp)
    
read_df = pd.concat(read_df_list)
read_df.head()
read_df['user_id'] = read_df['raw'].apply(lambda x: x.split(' ')[0])
read_df['article_id'] = read_df['raw'].apply(lambda x: x.split(' ')[1:])
read_df.head()
# 하나의 리스트로 반환하는 코드 
def chainer(s):
    return list(itertools.chain.from_iterable(s))

# article_id의 리스트가 풀어지면서 길어지는 것을 맞추기 위해서 np.repeat을 통해 같은 정보를 반복
read_cnt_by_user = read_df['article_id'].map(len)
read_rowwise = pd.DataFrame({'from': np.repeat(read_df['from'], read_cnt_by_user),
                             'to': np.repeat(read_df['to'], read_cnt_by_user),
                             'user_id': np.repeat(read_df['user_id'], read_cnt_by_user),
                             'article_id': chainer(read_df['article_id'])})

read_rowwise.reset_index(drop=True, inplace=True)
del read_cnt_by_user
read_rowwise.head()
users.head()
print("사용자의 수: ", users.shape[0])
print("작가의 수: ", users[users['keyword_list'].apply(lambda x: len(x)) != 0].shape[0])
print("구독하는 작가가 있는 사용자의 수: ", users[users['following_list'].apply(lambda x: len(x)) != 0].shape[0])
print("{}가 구독하는 작가가 있을 정도로 많은 비율을 차지".format('97.6%'))
users[users['keyword_list'].apply(lambda x: len(x)) != 0].head(1)['keyword_list'].values[0][0:10]
users['following_count'] = users['following_list'].apply(lambda x: len(x))
following_cnt_by_id = pd.DataFrame(users.groupby('following_count')['id'].count()).reset_index()

(ggplot(data=following_cnt_by_id)
    + geom_point(aes(x='following_count', y='id'), colour = '#49beb7')
    + theme_minimal()
    + ggtitle("구독하는 작가의 수별 사용자의 수")
    + labs(x="구독하는 작가의 수", y="사용자 수") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_text_y = element_text(color='black'),
         figure_size=(8,4))
 )
pd.DataFrame(users['following_count'].describe()).T
following_cnt_by_user = users['following_list'].map(len)
following_rowwise = pd.DataFrame({'user_id': np.repeat(users['id'], following_cnt_by_user),
                             'author_id': chainer(users['following_list'])})

following_rowwise.reset_index(drop=True, inplace=True)
following_rowwise.head()
following_cnt_by_id = following_rowwise.groupby('author_id')['user_id'].agg({'count'}).reset_index().sort_values(by='count', ascending=False)
following_cnt_by_id.head(10).T
following_cnt_by_id_ = following_cnt_by_id[following_cnt_by_id['author_id'] != '@brunch']
(ggplot(data=following_cnt_by_id_)
    + geom_histogram(aes(x='count', y='stat(count)'), fill = '#49beb7', binwidth=10)
    + theme_minimal()
    + ggtitle("작가별로 평균 구독자의 수")
    + labs(x="평균 구독자의 수", y="빈도") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_text_y = element_text(color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(8,4))
 )
pd.DataFrame(following_cnt_by_id['count'].describe()).T
del following_cnt_by_id_
gc.collect()
keyword_dict = {}
for i in tqdm_notebook(users[users['keyword_list'].apply(lambda x: len(x)) != 0]['keyword_list'].values):
    for j in range(0, len(i)):
        word = i[j]['keyword']
        cnt = i[j]['cnt']
        try:
            keyword_dict[word] += cnt
        except:
            keyword_dict[word] = cnt
# wordcloud에 대한 자세한 정보는 lovit님의 블로그 https://lovit.github.io/nlp/2018/04/17/word_cloud/를 참고하시기 바랍니다. 
from wordcloud import WordCloud
from PIL import Image

wordcloud = WordCloud(
    font_path = fontpath,
    width = 800,
    height = 800,
    background_color="white",
    mask= np.array(Image.open(path + "/figure/RS-KR.png"))

)
wordcloud = wordcloud.generate_from_frequencies(keyword_dict)
fig = plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()
fig.savefig('wordcloud.png')
del users
del wordcloud
del keyword_dict
gc.collect()
read_rowwise.head()
# article_id가 없는 경우 삭제 
read_rowwise = read_rowwise[read_rowwise['article_id'] != ''].reset_index(drop=True)
# 읽은날짜와 시간 추출 
read_rowwise['dt'] = read_rowwise['from'].astype(str).apply(lambda x: x[0:8]).astype(int)
read_rowwise['hr'] = read_rowwise['from'].astype(str).apply(lambda x: x[8:10]).astype(int)
read_rowwise['read_dt'] = pd.to_datetime(read_rowwise['dt'].astype(str).apply(lambda x: x[0:4] + '-' + x[4:6] + '-' + x[6:8]))
read_rowwise['article_id'].value_counts()[0:5]
following_cnt_by_id.head(10).T
del following_cnt_by_id
gc.collect()
read_rowwise['author_id'] = read_rowwise['article_id'].apply(lambda x: str(x).split('_')[0])
pd.DataFrame(read_rowwise['author_id'].value_counts()).head(10).T
# 구독하는 작가의 글을 읽는 비율 vs 그렇지 않은 작가의 글을 읽는 비율 
following_rowwise['is_following'] = 1
read_rowwise = pd.merge(read_rowwise, following_rowwise, how='left', on=['user_id', 'author_id'])
del following_rowwise
gc.collect()
read_rowwise['is_following'] = read_rowwise['is_following'].fillna(0)
read_rowwise['is_following'].value_counts(normalize=True)
read_following_author = read_rowwise.groupby(['user_id'])['is_following'].agg({'mean'}).reset_index()

(ggplot(data=read_following_author)
    + geom_histogram(aes(x='mean', y='stat(count)'), fill = '#49beb7', binwidth=0.005)
    + theme_minimal()
    + ggtitle("사용자별 구독하는 작가의 글을 읽는 비율")
    + labs(x="글 소비 비율", y="빈도") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_text_y = element_text(color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(8,4))
 )
del read_following_author
gc.collect()