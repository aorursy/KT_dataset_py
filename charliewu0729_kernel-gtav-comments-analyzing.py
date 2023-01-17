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
import re
import logging
import pandas as pd
import nltk, jieba
import codecs,csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
in_file = "../input/gtav_schinese.csv"
with codecs.open(in_file, 'r','utf-8') as f:
    comment_ls = []
    reader = csv.reader(f)
    for i in reader:
        comment_ls.append(i)
language = 'schinese'
def acq_chinese(x):
    #list1 = re.findall('[\u4e00-\u9fff,\u3000-\u302f]+',x)
    if not x:
        return ''
    else:
        list1 = re.findall('[\u4e00-\u9fff]+',x)
        new_sentence = ''.join(str(e) for e in list1)
        return new_sentence

comment_df = pd.DataFrame(comment_ls,columns=['Nickname','Title','Date','Hours','Link','Comment'])
if language == 'schinese' or language == 'tchinese':
    comment_df['Comment'] = comment_df['Comment'].apply(acq_chinese)
    comment_df = comment_df[~comment_df['Comment'].str.contains('本群欢迎各位萌新和老玩家')]
comment_df = comment_df.drop_duplicates().reset_index(drop=True)
all_comments_text = ''.join(list(comment_df['Comment']))
from jieba import posseg
pos_comments = jieba.posseg.lcut(all_comments_text)
stopwords_ls = stopwords = [line.strip() for line in open('../input/stopwords.txt', 'r',encoding="utf-8").readlines()]
outstr = ''
for word,pos in pos_comments:
    if word not in stopwords_ls and pos == 'n':
        outstr += word
        outstr += " "
wordcloud = WordCloud(width=800, height=400,background_color="white", max_words=2000, 
                      max_font_size=200,font_path="../input/simhei.ttf", random_state=42,collocations=False).generate(outstr)
plt.rcParams['figure.figsize'] = (8.0, 4.0) # set figure_size
plt.rcParams['image.interpolation'] = 'nearest' # set interpolation style
plt.rcParams['savefig.dpi'] = 500 #pixels
plt.rcParams['figure.dpi'] = 500 #resolution
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
top_num = 50
#nltk.download('punkt')
tokenstr = nltk.word_tokenize(outstr)
fdist1 = nltk.FreqDist(tokenstr)    
listkey = []
listval = []        
print(u".........Plot the frequency distribution of top n words ...............")
for key, val in sorted(fdist1.items(), key=lambda x: (x[1], x[0]), reverse=True)[:top_num]:
    if len(key) > 1:
        listkey.append(key)
        listval.append(val)

df = pd.DataFrame(listval, columns=['Frequency'])
df.index = listkey

if language == 'schinese' or language == 'tchinese':
    myfont = FontProperties(fname='../input/simhei.ttf')
    #plt.rcParams['font.family'] = 'SimHei'
ax = df.plot(kind='bar')
labels = [label for label in df.index.values] 
ax.set_xticklabels(labels, fontproperties=myfont) 
plt.rc('xtick', labelsize=6)
plt.title(u'Frequency distribution of Steam Comments',fontproperties=myfont)
plt.show()
