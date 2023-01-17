import requests

import pandas as pd

from tqdm import tqdm_notebook

import os

import json

PAGE_SUM = 50

FILE_NAME = 'top'



if not os.path.exists(FILE_NAME):

    os.mkdir(FILE_NAME)



# 访问搜索详情信息的api地址， 返回json格式;

#默认页面weibo_url = "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D%E5%8F%8C%E5%8D%81%E4%B8%80&page_type=searchall"

# 获取网页内容;

for page in tqdm_notebook(range(PAGE_SUM)):

    #不同页面后面加上page+1

    url = "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D%E5%8F%8C%E5%8D%81%E4%B8%80%26t%3D0&page_type=searchall&page="+str(page+1)

    json_file_path = os.path.join(FILE_NAME, '{}.json'.format(page+1))

    if os.path.exists(json_file_path):  # 如果已经爬取

        continue

    while True:  # 一直尝试到成功

        try:

            response = requests.get(url, timeout=5)

        except requests.exceptions.Timeout:

            time.sleep(1)

        if response.status_code == 200:

            break

    with open(json_file_path, 'w') as f:  # 写入本地文件

        # indent 表明需要对齐，ensure_ascii=False 编码中文

        f.write(json.dumps(json.loads(response.content.decode('utf-8')),

                           indent=4, ensure_ascii=False))

weibos = []

for page in tqdm_notebook(range(PAGE_SUM)):

    json_file_path = os.path.join(FILE_NAME, '{}.json'.format(page+1))

    with open(json_file_path) as f:

        #print(json.load(f)['data']['cards'])

        weibos += json.load(f)['data']['cards'] #存储微博数据



with open('top.json', 'w') as f:  # 写入文件

    f.write(json.dumps({'weibos': weibos}, indent=4, ensure_ascii=False))

    

created_at = [] #发布时间

blog_id = [] #微博id

text = [] #微博内容

source = [] #微博客户端

user_id = [] #用户id

user_screen_name = [] #用户名

user_statuses_count = [] #用户微博数

user_gender = [] #用户性别

user_followers_count = [] #用户分析数

user_follow_count = [] #用户关注数

reposts_count = [] #微博转发数

comments_count = [] #微博评论数

attitudes_count = [] #微博点赞数

for blog in weibos:

    if blog['card_type'] == 9:

        created_at.append(blog['mblog']['created_at'])

        blog_id.append(blog['mblog']['id'])

        text.append(blog['mblog']['text'])

        source.append(blog['mblog']['source'])

        user_id.append(blog['mblog']['user']['id'])

        user_screen_name.append(blog['mblog']['user']['screen_name'])

        user_statuses_count.append(blog['mblog']['user']['statuses_count'])

        user_gender.append(blog['mblog']['user']['gender'])

        user_followers_count.append(blog['mblog']['user']['followers_count'])

        user_follow_count.append(blog['mblog']['user']['follow_count'])

        reposts_count.append(blog['mblog']['reposts_count'])

        comments_count.append(blog['mblog']['comments_count'])

        attitudes_count.append(blog['mblog']['attitudes_count'])

df = pd.DataFrame({'created_at':created_at, 'blog_id': blog_id, 'text':text, 'source':source,

               'user_id': user_id, 'user_screen_name':user_screen_name, 'user_statuses_count': user_statuses_count, 'user_gender': user_gender,

               'user_followers_count': user_followers_count, 'user_follow_count':user_follow_count,

               'reposts_count': reposts_count, 'comments_count': comments_count, 'attitudes_count':attitudes_count})



df.to_csv('top.csv')
df.head()
df = pd.read_csv('top.csv')
!wget -nc "http://labfile.oss.aliyuncs.com/courses/1176/stopwords.txt"

def load_stopwords(file_path):

    # 加载停用词函数

    with open(file_path, 'r') as f:

        stopwords = [line.strip('\n') for line in f.readlines()]

    return stopwords
import re

import jieba

stopwords = load_stopwords('stopwords.txt')



def text_clean(string):

    # 对一个微博中文内容进行分词

    result = []

    #print(string)

    para = string.split(' ')

    #print(para)

    result = []

    for p in para:

        #print(p)

        p = ''.join(re.findall('[\u4e00-\u9fa5]', p))

        #print(p)

        # 对每一个分句进行分词

        seg_list = list(jieba.cut(p, cut_all=False))

        for x in seg_list:

            if len(x) <= 1:

                continue

            if x in stopwords:

                continue

            result.append(x)

    return result

text_all = []

text_list = []

for i in range(len(df)):

    text_all.extend(text_clean(df.iloc[i]['text']))

    text_list.append(text_clean(df.iloc[i]['text']))

df['text'] = text_list
!wget -nc "http://labfile.oss.aliyuncs.com/courses/1176/fonts.zip"

!unzip -o fonts.zip
from wordcloud import WordCloud

from matplotlib import pyplot as plt

%matplotlib inline



font_path = 'fonts/SourceHanSerifK-Light.otf'

wc = WordCloud(font_path=font_path, background_color="white", max_words=1000,

               max_font_size=100, random_state=42, width=800, height=600, margin=2)

word_dict= {}

for word in text_all:

    if word not in word_dict:

        word_dict[word] = 1

    else:

        word_dict[word] += 1

wc.generate_from_frequencies(word_dict)





plt.figure(figsize=(8, 6))

plt.imshow(wc, interpolation="bilinear")  # 显示图片

plt.axis("off")
word_dict = sorted(word_dict.items(), key = lambda x:x[1], reverse=True)

word_dict[:20]
created_at_list = []

for created_at in df['created_at']:

    created_at = ''.join(re.findall('[\u4e00-\u9fa5]', created_at))

    if len(created_at) < 1:

        created_at_list.append('很久前')

    else:

        created_at_list.append(created_at)

df['created_at'] = created_at_list
source_list = []

system = []

for source in df['source']:

    source = str(source)

    if 'nova' in source or 'HUAWEI' in source or '华为' in source or '荣耀' in source:

        source_list.append('HUAWEI')

        system.append('Android')

    elif 'iPhone' in source or 'iPad' in source:

        source_list.append('iPhone')

        system.append('IOS')

    elif 'OPPO' in source:

        source_list.append('OPPO')

        system.append('Android')

    elif 'vivo' in source:

        source_list.append('vivo')

        system.append('Android')

    elif 'OnePlus' in source:

        source_list.append('OnePlus')

        system.append('Android')

    elif 'Redmi' in source or '红米' in source or '小米' in source:

        source_list.append('XIAOMI')

        system.append('Android')

    elif '魅族' in source:

        source_list.append('MEIZU')

        system.append('Android')

    elif '联想' in source:

        source_list.append('Lenovo')

        system.append('Android')

    elif 'Samsung' in source or '三星' in source:

        source_list.append('Samsung')

        system.append('Android')

    elif '浏览器' in source or '微博' in source or '网页' in source:

        source_list.append('Web')

        system.append('Web')

    elif 'Android' in source:

        source_list.append('Not Known')

        system.append('Android')

    else:

        source_list.append('Not Known')

        system.append('Not Known')

df['source'] = source_list

df['system'] = system
df.drop(['blog_id', 'user_id', 'user_screen_name'], axis=1, inplace=True)
df.head()
import string

i = 0

hot_word_list = [w for w, n in word_dict[1:20]]

hot_word_count_all = []

for text in df['text']:

    hot_word_count = 0

    cold_word_count = 0

    for word in text:

        if word in hot_word_list:

            hot_word_count += 1

    hot_word_count_all.append(hot_word_count)

#print(hot_word_count_all)

word_columns = []

df['hot_word'] = hot_word_count_all
df.head()
from collections import Counter

hot_word_c = Counter(df['hot_word'])

hot_word_count = []

for i in range(1, df['hot_word'].max()):

    hot_word_count.append(hot_word_c[i])

print(hot_word_count)

plt.pie(x=hot_word_count, labels=[str(i) for i in range(1, df['hot_word'].max())])

plt.show()
import matplotlib

df_text = df.groupby(by='created_at').sum()['hot_word']

df_text.reset_index()

#print(df_text)

df_text = df_text.reindex(index=['很久前', '昨天', '小时前', '分钟前', '刚刚'])

#print(df_text)

fig = plt.figure(figsize=(16, 9))

myfont = matplotlib.font_manager.FontProperties(fname="fonts/SourceHanSerifK-Light.otf")

plt.plot(df_text)

plt.xticks(fontproperties=myfont)

plt.legend('hot_word', prop=myfont)

#plt.xlabel("")

plt.show()
fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(111)

ax.set_xscale("log")

ax.set_yscale("log")

df1 = df[['user_followers_count','reposts_count']]

df1 = df1.sort_values(by='user_followers_count',ascending= False)  

ax.plot(df1['user_followers_count'], df1['reposts_count'])

plt.xlabel('user_followers_count')

plt.ylabel('reposts_count')
df_text_source = df.groupby(by='source').sum()['hot_word']

plt.pie(df_text_source, labels=df_text_source.keys(),autopct='%1.2f%%')

plt.show()