%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import sys
import time
import tqdm
import pickle
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
text = pd.read_csv('../input/chinese-official-daily-news-since-2016/chinese_news.csv', usecols=['tag', 'headline', 'content'])
dates = pd.read_csv('../input/chinese-official-daily-news-since-2016/chinese_news.csv', usecols=['date'])
text.shape
text.head()
dates['datetime'] = dates['date'].apply(lambda x: pd.to_datetime(x))
dates['year'] = dates['datetime'].dt.year
dates['month'] = dates['datetime'].dt.month
dates['dow'] = dates['datetime'].dt.dayofweek
plt.figure(figsize=[8, 4])
sns.countplot(x='year', data=dates)
plt.title('News count by day of week')
plt.ylabel('News count');
plt.xlabel('Year(where year 2018 is up to 10/09)');
plt.figure(figsize=[10, 5])
sns.countplot(x='dow', data=dates)
plt.title('News count by day of week')
plt.ylabel('News count');
plt.xlabel('Day of week');
plt.figure(figsize=[12, 6])
sns.countplot(x='month', data=dates)
plt.title('News count by month')
plt.ylabel('News count');
plt.xlabel('Month');
!wget https://github.com/adobe-fonts/source-han-sans/raw/release/SubsetOTF/SourceHanSansCN.zip
!unzip -j "SourceHanSansCN.zip" "SourceHanSansCN/SourceHanSansCN-Regular.otf" -d "."
!rm SourceHanSansCN.zip
!ls
import matplotlib.font_manager as fm
font_path = './SourceHanSansCN-Regular.otf'
prop = fm.FontProperties(fname=font_path)
plt.figure(figsize=[8, 4])
ax = sns.countplot(x='tag', data=text)
plt.title('News count by tag')
plt.ylabel('News count')
plt.xlabel('Tag')
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=prop);
!pip install jieba
import jieba

def jieba_cut(x, sep=' '):
    return sep.join(jieba.cut(x, cut_all=False))

print('raw', text['headline'][0])
print('cut', jieba_cut(text['headline'][0], ', '))
from joblib import Parallel, delayed
%%time
text['headline_cut'] = Parallel(n_jobs=4)(
    delayed(jieba_cut)(x) for x in tqdm.tqdm_notebook(text['headline'].values)
)
%%time
text['content_cut'] = Parallel(n_jobs=4)(
    delayed(jieba_cut)(x) for x in tqdm.tqdm_notebook(text['content'].fillna('').values)
)
from wordcloud import WordCloud, ImageColorGenerator

def get_wc(
    text_li, 
    background_color='white',
    max_words = 1500,
    font_path=font_path,
    width=800,
    height=600,
    max_font_size=64,
    mask=None,
    margin= 1,
):
    return WordCloud(
        background_color=background_color,
        max_words=max_words,        
        font_path=font_path,
        width=width,
        height=height,
        max_font_size=max_font_size,
        mask=mask,
        margin=margin,
        contour_color='steelblue'
    ).generate(" ".join(text_li))
mask = np.load('../input/jiangmask/jiang_mask.npy')
text_li = text['headline_cut'].values.tolist()
wc = get_wc(text_li, mask=mask)
plt.figure(figsize=[12, 8])
plt.imshow(wc)
plt.title('All headlines')
plt.axis('off');
text_li = text['content_cut'].values.tolist()
wc = get_wc(text_li, mask=mask)
plt.figure(figsize=[12, 8])
plt.imshow(wc)
plt.title('All contents')
plt.axis('off');
plt.figure(figsize=[8*3, 6])
tags = text['tag'].unique()
for i,op in enumerate([('tag', tags[0]), ('tag', tags[1]), ('tag', tags[2])]):
    plt.subplot(1, 3, i+1)
    text_li = text.loc[text[op[0]]==op[1], 'headline_cut'].values.tolist()
    wc = get_wc(text_li, mask=mask)
    plt.imshow(wc)
    plt.title(f'Headlines of {op[0]} {op[1]}', fontproperties=prop)
    plt.axis('off');
plt.tight_layout();
plt.figure(figsize=[8*3, 6])
for i,op in enumerate([('year', 2016), ('year', 2017), ('year', 2018)]):
    plt.subplot(1, 3, i+1)
    text_li = text.loc[dates[op[0]]==op[1], 'headline_cut'].values.tolist()
    wc = get_wc(text_li, mask=mask)
    plt.imshow(wc)
    plt.title(f'Headlines of {op[0]} {op[1]}')
    plt.axis('off');
plt.tight_layout();
plt.figure(figsize=[8*3, 6*4])
for i,m in enumerate(range(12)):
    op = ('month', m+1)
    plt.subplot(4, 3, i+1)
    text_li = text.loc[dates[op[0]]==op[1], 'headline_cut'].values.tolist()
    wc = get_wc(text_li, mask=mask)
    plt.imshow(wc)
    plt.title(f'Headlines of {op[0]} {op[1]}')
    plt.axis('off');
plt.tight_layout();
!wget -O 'mask.jpg' -q http://p3.pstatp.com/large/37cb00016104f26b1608 
print(os.listdir('../input/jiangmask'))
from PIL import Image
mask = np.array(Image.open('../input/jiangmask/jiang.png'))
maskb = mask.copy()
maskb[maskb<90]=0
maskb[maskb>=90]=255
maskb = 255 - maskb
mask = 0.8 * mask
mask[maskb==255] = 255

text_li = text['headline_cut'].values.tolist()
wc = get_wc(text_li, mask=mask)
plt.figure(figsize=[12, 12])
plt.imshow(wc.recolor(color_func=ImageColorGenerator(mask)))
plt.imshow(np.array(Image.open('../input/jiangmask/jiang.png')), alpha=0.2)
plt.title('Too young too simple, simetimes naive!')
plt.axis('off');
