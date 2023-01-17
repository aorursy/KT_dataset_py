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
# 하나의 리스트로 반환하는 코드 
def chainer(s):
    return list(itertools.chain.from_iterable(s))
# pd.read_json : json 형태의 파일을 dataframe 형태로 불러오는 코드 
magazine = pd.read_json(path + 'magazine.json', lines=True) # lines = True : Read the file as a json object per line.
metadata = pd.read_json(path + 'metadata.json', lines=True)
users = pd.read_json(path + 'users.json', lines=True)
import itertools
from itertools import chain
import glob
import os 
read_rowwise = pd.read_csv(path + "read_rowwise.csv")
%%time 
# article_id가 없는 경우 삭제 
read_rowwise = read_rowwise[read_rowwise['article_id'] != ''].reset_index(drop=True)

# 읽은날짜와 시간 추출 
read_rowwise['dt'] = read_rowwise['from'].astype(str).apply(lambda x: x[0:8]).astype(int)
read_rowwise['hr'] = read_rowwise['from'].astype(str).apply(lambda x: x[8:10]).astype(int)
read_rowwise['read_dt'] = pd.to_datetime(read_rowwise['dt'].astype(str).apply(lambda x: x[0:4] + '-' + x[4:6] + '-' + x[6:8]))

read_rowwise['author_id'] = read_rowwise['article_id'].apply(lambda x: str(x).split('_')[0])
read_rowwise[read_rowwise['user_id'] == '#0000d1188f75d0b0ea7a8e23a2b760e5']
read_rowwise[read_rowwise['user_id'] == '#0000eea6d339abfd02ed590bc451fc63']
following_cnt_by_user = users['following_list'].map(len)
following_rowwise = pd.DataFrame({'user_id': np.repeat(users['id'], following_cnt_by_user),
                             'author_id': chainer(users['following_list'])})

following_rowwise.reset_index(drop=True, inplace=True)

# 구독하는 작가의 글을 읽는 비율 vs 그렇지 않은 작가의 글을 읽는 비율 
following_rowwise['is_following'] = 1
read_rowwise = pd.merge(read_rowwise, following_rowwise, how='left', on=['user_id', 'author_id'])

del following_rowwise
gc.collect()
user_id = '#a87e970972364bb14a542f57b0933db9'
read_user = read_rowwise[read_rowwise['user_id'] == user_id]
read_user = read_user.groupby(['read_dt', 'is_following', 'author_id'])['author_id'].agg({'count'}).reset_index()
from mizani.breaks import date_breaks
from mizani.formatters import date_format

(ggplot(data=read_user)
    + geom_point(aes(x='read_dt', y='count', color='author_id', size='count'), alpha = 0.5, show_legend=False)
    + geom_text(aes(x='read_dt', y='count', label='author_id'), color='grey', 
               data = read_user[read_user['is_following'] == 1], size=7)
    + scale_x_datetime(breaks=date_breaks('1 month'), labels=date_format('%Y%m'))
    + theme_minimal()
    + ggtitle("일별 읽은 작가의 글의 빈도")
    + labs(x="년도", y="읽은 작가의 글의 빈도") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_text_y = element_text(color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(12,6))
 )
metadata.head()
from datetime import datetime 
metadata['reg_datetime'] = metadata['reg_ts'].apply(lambda x : datetime.fromtimestamp(x/1000.0))
metadata.loc[metadata['reg_datetime'] == metadata['reg_datetime'].min(), 'reg_datetime'] = datetime(2090, 12, 31)
metadata['reg_dt'] = metadata['reg_datetime'].dt.date
metadata['type'] = metadata['magazine_id'].apply(lambda x : '개인' if x == 0.0 else '매거진')
metadata['reg_dt'] = pd.to_datetime(metadata['reg_dt'])
read_cnt_by_reg_dt = pd.DataFrame(metadata.groupby('reg_dt')['article_id'].count()).reset_index()
read_cnt_by_reg_dt = read_cnt_by_reg_dt.iloc[:-1]

(ggplot(data=read_cnt_by_reg_dt)
    + geom_line(aes(x='reg_dt', y='article_id'), colour = '#49beb7')
    + theme_minimal()
    + ggtitle("등록일자별 글 수")
    + labs(x="등록일자", y="글 수") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(12,6))
 )
read_cnt_by_reg_dt_ = read_cnt_by_reg_dt[read_cnt_by_reg_dt['reg_dt'] >= '2019-03-01']

(ggplot(data=read_cnt_by_reg_dt_)
    + geom_line(aes(x='reg_dt', y='article_id'), colour = '#49beb7')
    + theme_minimal()
    + ggtitle("3월의 등록일자별 글 수")
    + labs(x="등록일자", y="글 수") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(12,6))
 )
# metadata[metadata['magazine_id'] == 34075]
read_cnt_by_reg_dt = read_rowwise[read_rowwise['author_id'] == '@merryseo'].groupby(['read_dt','article_id'])['article_id'].agg({'count'}).reset_index()
read_cnt_by_reg_dt = read_cnt_by_reg_dt.sort_values(by='read_dt', ascending=False)

(ggplot(data=read_cnt_by_reg_dt)
    + geom_line(aes(x='read_dt', y='count',group='article_id', colour='article_id'), show_legend=False)
    + theme_minimal()
    + ggtitle("위클리 매거진 Magazine(34075)")
    + labs(x="등록일자", y="글 소비수") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(12,6))
    + scale_color_hue(l=0.5)
 )
# metadata[metadata['magazine_id'] == 34075]
read_cnt_by_reg_dt = read_rowwise[read_rowwise['author_id'] == '@basenell'].groupby(['read_dt','article_id'])['article_id'].agg({'count'}).reset_index()
metadata_ = metadata[['id', 'reg_dt']].rename(columns={'id':'article_id'})
read_cnt_by_reg_dt = pd.merge(read_cnt_by_reg_dt, metadata_[['article_id', 'reg_dt']], how='left', on='article_id')
read_cnt_by_reg_dt = read_cnt_by_reg_dt[read_cnt_by_reg_dt['reg_dt'] >= '2019-01-15']

(ggplot(data=read_cnt_by_reg_dt)
    + geom_line(aes(x='read_dt', y='count',group='article_id', colour='article_id'), show_legend=False)
    + theme_minimal()
    + ggtitle("위클리 매거진 Magazine(40511)")
    + labs(x="등록일자", y="글 소비수") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(12,6))
    + scale_color_hue(l=0.5)
 )
# 등록일에 따른 글 소비수의 변화 
read_rowwise = pd.merge(read_rowwise, metadata_, how='left', on='article_id')
read_rowwise['diff_dt'] = (read_rowwise['read_dt'] - read_rowwise['reg_dt']).dt.days
off_day = read_rowwise.groupby(['diff_dt'])['diff_dt'].agg({'count'}).reset_index()

# 메타데이터에 날짜가 잘못 매핑되어서 음수값이 나오는 값 제거 
# 200이하로 뽑은 이유는 날짜가 너무 큰 데이터가 있어서 제거했음 
off_day = off_day[(off_day['diff_dt'] >= 0) & (off_day['diff_dt'] <= 200)]
(ggplot(data=off_day)
    + geom_line(aes(x='diff_dt', y='count'), color='#49beb7')
    + theme_minimal()
    + ggtitle("경과일에 따른 글 소비수 변화")
    + labs(x="경과일", y="평균 글 소비수") 
    + theme(text = element_text(fontproperties=fm.FontProperties(fname=fontpath, size=9)),
         axis_text_x = element_text(angle=60, color='black'),
         axis_line=element_line(color="black"),
         axis_ticks=element_line(color = "grey"),
         figure_size=(12,6))
    + scale_color_hue(l=0.5)
 )
metadata['keyword_list'].values[0]
keyword_dict = {}
for i in tqdm_notebook(metadata[metadata['keyword_list'].apply(lambda x: len(x)) != 0]['keyword_list'].values):
    for j in range(0, len(i)):
        word = i[j]
        cnt = 1
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
fig.savefig('wordcloud2.png')