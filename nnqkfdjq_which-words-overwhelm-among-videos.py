import pandas as pd
import numpy as np
import holoviews as hv
from datetime import datetime, timedelta
import json
import sqlite3
from sqlalchemy import create_engine
from urllib.parse   import quote
from urllib.request import urlopen
import time
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
hv.extension('bokeh')

fm = pd.read_csv("../input/statistics-observation-of-random-youtube-video/count_observation_upload.csv")
fm2 = pd.read_csv("../input/statistics-observation-of-random-youtube-video/video_characteristics_upload.csv")
fm2 = fm2.drop('Unnamed: 0', axis = 1)
fm = fm.drop('Unnamed: 0', axis = 1)
fm = fm.set_index('index')
datetime_tran2 = lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
fm.loc[:,['commentCount', 'dislikeCount', 'favoriteCount', 'likeCount', 'viewCount']] =  fm.loc[:,['commentCount', 'dislikeCount', 'favoriteCount', 'likeCount', 'viewCount']].astype(np.float)
fm['Time'] = fm['Time'].map(datetime_tran2)
videoId_list = list(fm.videoId.unique())
vi_cat_dict = fm2.loc[:,['videoId','categoryId']].set_index('videoId').to_dict()['categoryId']
fm['categoryId'] = fm['videoId'].map(vi_cat_dict)
with open('../input/youtube-new/US_category_id.json') as fb:
    fb = fb.read()
    categoryId_to_name = json.loads(fb)
categoryId_to_name2 = {}
for item in categoryId_to_name['items']:
    categoryId_to_name2[np.float(item['id'])] = item['snippet']['title'] 
fm['categoryId'] = fm['categoryId'].map(categoryId_to_name2)
%opts Bars [stack_index=1 xrotation=0 width=800 height=500 show_legend=False tools=['hover']]
%opts Bars (color=Cycle('Category20'))
wordnet_lemmatizer = WordNetLemmatizer()
def treebank_tag_to_wordnet_pos2(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
vi_tit_dict = fm2.loc[:,['videoId','title']].set_index('videoId').to_dict()['title'] #create word to viewcount, title, and tag dictionary 
fm['title'] = fm['videoId'].map(vi_tit_dict)
kingking = re.compile(r"official|music|lyric|ft|mv") #트레일러는 나중에 추가할까. 
net_text = {} # lemmatization할때 대문자면 제대로 안됨. 
word2_title_dict = {}
tag_checker = {}
net_text3 = {} 
target_fm = fm[fm['categoryId'] != 'Music'] #카테고리 music 인거 빼고 
target_fm = target_fm.loc[:,['title','Time','viewCount_diff']]  #사이즈 줄이고. 
for numnum, row in enumerate(target_fm.iterrows()):
    title = row[1]['title']
    viewC = row[1]['viewCount_diff']
    infofo = row[1].iloc[1:]
    if type(title) != str: #title이 str 아니면 빼고 
        continue
    title = title.lower()
    if len(re.findall(kingking, title)) > 0:
        continue  #music official lyric ft 이런말 들어가는 거 다 빼고. 정규표현식으로 따내야할 듯. 
    text = nltk.word_tokenize(title)
    #text = list(map(lambda x:x.lower(),text))
    pos_text = nltk.pos_tag(text)
    dict__KEYS = list(net_text.keys())
    for word, tag in pos_text:
        changed_tag = treebank_tag_to_wordnet_pos2(tag)
        if changed_tag != wordnet.NOUN: #명사만 헀고. pos tag 했을때 명사로 뜬 애들만. 
            continue
        word2 = (wordnet_lemmatizer.lemmatize(word, pos = changed_tag))
        if not word2 in dict__KEYS:
            net_text[word2]  = []
            word2_title_dict[word2] = []
            tag_checker[word2] = []
            net_text3[word2]  = []
        net_text[word2].append(viewC)
        word2_title_dict[word2].append(title)
        tag_checker[word2].append(changed_tag)
        net_text3[word2].append(infofo)
    #print("{0} done".format(numnum))
mean_net_dict = {} #sum of view counts of each word related videos during the entire period 
for net in net_text:
    dirt = net_text[net]
    dirt2 = [x for x in dirt if x >= 0] #sorting out error 
    smm = sum(dirt2)
    mean_net_dict[net] = smm #구함. 
final_series = pd.Series(mean_net_dict).sort_values(ascending = False )
final_series = final_series.reset_index()
final_series.columns = ['word', 'viewCount_sum']
final_series_ds = hv.Dataset(final_series, kdims = 'word', vdims = 'viewCount_sum')
hv.Bars(final_series_ds, label = "Sort words by their entire viewcounts they obtained")
mean_net_dict2 = {}
for word in word2_title_dict:
    title = set(word2_title_dict[word])
    if len(title) > 2:
        mean_net_dict2[word] = mean_net_dict[word]
final_series2 = pd.Series(mean_net_dict2).sort_values(ascending = False)
final_series2 = final_series2.reset_index()
final_series2.columns = ['word', 'viewCount_sum']
final_series2_ds = hv.Dataset(final_series2, kdims = 'word', vdims = 'viewCount_sum')
hv.Bars(final_series2_ds, label = "Words that have more than three related videos")
set(word2_title_dict['royale'])


word_title_len = {}
for word in word2_title_dict:
    word_title_len[word] = len(set(word2_title_dict[word]))
word_title_len_df = pd.Series(word_title_len).sort_values(ascending= False).reset_index()
word_title_len_df.columns = ['word', 'num_of_title']
word_title_len_ds = hv.Dataset(word_title_len_df, kdims = 'word', vdims = 'num_of_title')
hv.Bars(word_title_len_ds, label="Sort words by the number of related videos.")



def top10(x):
    return x.sort_values(by = 'viewCount_diff', ascending = False)[0:10]
df_listlist = []
for key in net_text3:
    keydf = net_text3[key]
    keydf = pd.concat(keydf, axis = 1).T.groupby('Time').sum()
    keydf['word'] = key
    df_listlist.append(keydf) #밑에랑 합쳤으면 좋겠는데 이게 단어별로 모여있어서. 밑에 과정을 거쳐서 
word_tit = pd.concat(df_listlist, axis = 0).reset_index().groupby(['Time','word']).sum().reset_index()
word_tit2 = word_tit.groupby('Time').apply(top10)
word_tit2 = word_tit2.drop('Time',axis = 1).reset_index()
key_dimensions3   = [('Time', 'Time'), ('word', 'word')]
value_dimensions3 = [('viewCount_diff', 'viewCount_diff')]
macro4 = hv.Table(word_tit2, key_dimensions3, value_dimensions3)
macro4.to.bars(['Time','word'], 'viewCount_diff', [], label = "How about Top 10 words of each Time.")

def title_more(x):
    return x in list(mean_net_dict2.keys())
word_tit3 = word_tit[word_tit.word.map(title_more)]
word_tit3 = word_tit3.groupby('Time').apply(top10).drop('Time', axis = 1).reset_index()
key_dimensions32   = [('Time', 'Time'), ('word', 'word')]
value_dimensions32 = [('viewCount_diff', 'viewCount_diff')]
macro42 = hv.Table(word_tit3, key_dimensions32, value_dimensions32)
macro42.to.bars(['Time','word'], 'viewCount_diff', [], label="Words that have more than three related videos") 



