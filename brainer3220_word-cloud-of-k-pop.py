from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print(os.listdir('../input/kpopdb'))
nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/kpopdb/kpop_music_videos.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'IN_youtube_trending_data.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(3)
!pip3 install konlpy
from konlpy.tag import Twitter
from collections import Counter
SongName = df1['Song Name'].dropna()
KoreanName = df1['Korean Name'].dropna()
KoreanName
twitter = Twitter()
morphs = []

for sentence in SongName: 
    morphs.append(twitter.pos(sentence))
    
for sentence in KoreanName: 
    morphs.append(twitter.pos(sentence))
    
morphs
noun_adj_adv_list=[] 
for sentence in morphs : 
    for word, tag in sentence : 
        if tag in ['Noun'] and ("것" not in word) and ("내" not in word)and ("나" not in word)and ("수"not in word) and("게"not in word)and("말"not in word): 
            noun_adj_adv_list.append(word) 
noun_adj_adv_list[:10]
count = Counter(noun_adj_adv_list)
words = dict(count.most_common())
words
from wordcloud import WordCloud 

import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
%matplotlib inline

import matplotlib
from IPython.display import set_matplotlib_formats
matplotlib.rc('font',family = 'Malgun Gothic')

set_matplotlib_formats('retina')

matplotlib.rc('axes',unicode_minus = False)
# 그래프에서 한글표현을 위해 폰트를 설치합니다.
%config InlineBackend.figure_format = 'retina'

!apt -qq -y install fonts-nanum > /dev/null
import matplotlib.font_manager as fm
fontpath = 'fonts/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
# 기본 글꼴 변경
import matplotlib as mpl
mpl.font_manager._rebuild()
mpl.pyplot.rc('font', family='NanumBarunGothic')
wordcloud = WordCloud(font_path = fontpath, background_color='white',colormap = "Accent_r", width=2500, height=2000).generate_from_frequencies(words) 
plt.imshow(wordcloud)
plt.axis('off') 
plt.show()
