from lxml import html

from matplotlib import pyplot as plt





from lxml import html

from matplotlib import pyplot as plt

from matplotlib.font_manager import FontProperties



#下方所有图例使用中文字体

font = FontProperties(fname=r"../input/fontresource/simhei.ttf")



def readDataList(xmlPath,isMultiHead=False):

    html_content = open(xmlPath, 'r', encoding="utf-8").read()

    root = html.fromstring(html_content)

    timeList = []

    priceList = []

    if isMultiHead:

        a=2

    else:

        a=1

    timeList.extend(root.xpath('//td[1]/span/text()')[a::][::-1])

    priceList.extend(root.xpath('//td[2]/span/text()')[1::][::-1])

    for i, v in enumerate(priceList): priceList[i] = float(v)

    for i, v in enumerate(timeList): timeList[i] = str(v)

    return timeList,priceList







pigTimeList,pigPriceList=readDataList('../input/dataset/homeporkprice.html')



#设置画布大小

fig, axes = plt.subplots(figsize=(16, 9), dpi=200)

axes.plot(pigTimeList, pigPriceList, label='价格')

axes.set_xlabel("2018年01年-2019年12月", fontproperties=font)

axes.set_ylabel("猪肉价格 均价(元/公斤)", fontproperties=font)

axes.set_title('猪肉价格趋势', fontproperties=font)



plt.xticks(rotation=90, fontproperties=font)

#设置刻度标记的大小

plt.tick_params(axis='both',labelsize=10)

#绘制每个价格点,显示价格坐标

axes.scatter(pigTimeList, pigPriceList)

for a, b in zip(pigTimeList, pigPriceList):  

    plt.text(a, b, (b),ha='center', va='bottom', fontsize=10)

plt.legend(prop=font)

plt.show()



import  pandas as pd



def plotXY(xList,yList):

    for a, b in zip(xList, yList):  

        plt.text(a, b, (b),ha='center', va='bottom', fontsize=10)

    axes.scatter(xList, yList)

    

cowTimeList,cowPriceList=readDataList('../input/dataset/cow_price.html')

goatTimeList,goatPriceList=readDataList('../input/dataset/goat_price.html')

eggTimeList,eggPriceList=readDataList('../input/dataset/egg_price.html')

data = pd.read_csv("../input/dataset/province_data.csv", encoding='GBK')

data.set_index('地区', drop=True, inplace=True)

df2=data.stack()

df3=df2.unstack(0)

df3=df3.iloc[::-1]

fig, axes = plt.subplots(figsize=(16, 9), dpi=300)



axes.plot(pigTimeList, pigPriceList, label='猪肉')

axes.plot(cowTimeList, cowPriceList, label='牛肉')

axes.plot(goatTimeList, goatPriceList, label='羊肉')

axes.plot(eggTimeList, eggPriceList, label='鸡蛋')

axes.set_xlabel("2018年01年-2019年12月",fontproperties=font)

axes.set_ylabel("价格 均价(元/公斤)",fontproperties=font)

axes.set_title('价格趋势',fontproperties=font)

plt.xticks(rotation=90,fontproperties=font)



#绘制点坐标



plotXY(pigTimeList, pigPriceList)

plotXY(cowTimeList, cowPriceList)

plotXY(goatTimeList, goatPriceList)

plotXY(eggTimeList, eggPriceList)

plt.legend(prop=font)

plt.show()



#生猪当月进口数量

timeList2,numList=readDataList('../input/dataset/pig_input.html',True)

x = range(len(timeList2))

# print(numList)



#生猪当月出口数量 吨

timeList3,outputList=readDataList('../input/dataset/pig_output.html',True)



"""

绘制条形图

left:长条形中点横坐标

height:长条形高度

width:长条形宽度，默认值0.8

label:为后面设置legend准备

"""

fig, axes = plt.subplots(figsize=(16, 9), dpi=300)

rects1 = plt.bar(x, height=numList, width=0.4, alpha=0.4, color='red', label="进口")

rects2 = plt.bar([i + 0.4 for i in x], height=outputList, width=0.4, color='green', label="出口")

plt.ylabel("数量", fontproperties=font)

"""

设置x轴刻度显示值

参数一：中点坐标

参数二：显示值

"""

plt.xticks([index + 0.2 for index in x], timeList2, fontproperties=font)

plt.xlabel("2018年1月-2019年10月", fontproperties=font)

plt.title("猪肉进出口数量对比 单位(吨)", fontproperties=font)

plt.legend(prop=font)     # 设置题注

# 编辑文本

for rect in rects1:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom", fontproperties=font)

for rect in rects2:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom", fontproperties=font)

plt.xticks(rotation=90, fontproperties=font)

plt.show()
#%%



import  pandas as pd



data = pd.read_csv("../input/dataset/province_data.csv", encoding='GBK')

data.set_index('地区', drop=True, inplace=True)

df2=data.stack()

df3=df2.unstack(0)

df3=df3.iloc[::-1]



axes=df3.plot(figsize=(16,9))

plt.legend(prop=font,loc=2)

axes.set_xlabel("时间轴",fontproperties=font)

axes.set_ylabel("消费价格指数",fontproperties=font)

axes.set_title('畜肉类居民消费价格指数(上年同月=100)',fontproperties=font)

plt.xticks(rotation=90,fontproperties=font)



plt.show()

import numpy as np



html_content = open('../input/dataset/pork_pricelist.html', 'r', encoding="utf-8").read()

root = html.fromstring(html_content)

porkNameList = []

lowPriceList = []

avgPriceList=[]

highPriceList=[]

porkNameList.extend(root.xpath('//td[1]/text()')[1:-1])

lowPriceList.extend(root.xpath('//td[2]/text()')[1:-1])

avgPriceList.extend(root.xpath('//td[3]/text()')[1:-1])

highPriceList.extend(root.xpath('//td[4]/text()')[1:-1])

for i, v in enumerate(lowPriceList): lowPriceList[i] = float(v)

for i, v in enumerate(highPriceList): highPriceList[i] = float(v)

for i, v in enumerate(avgPriceList): avgPriceList[i] = float(v)

    

a=pd.DataFrame({'name':porkNameList,

                'lowPrice':lowPriceList,

                'avgPrice':avgPriceList,

                'highPrice':highPriceList})

#数据

total_width, n = 0.8, 3

width = total_width / n

y_pos = np.arange(len(porkNameList))

y_pos=y_pos - (total_width - width) / 2

#图像绘制

fig,ax=plt.subplots(figsize=(16, 14), dpi=300)







b=ax.barh(y_pos,lowPriceList,align='center',color='blue', ecolor='black',height=0.1,label='最低价')

#添加数据标签

for rect in b:

    w=rect.get_width()

    ax.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%float(w),ha='left',va='center')

    

    

b=ax.barh(y_pos+width, highPriceList, align='center',color='red', ecolor='black',height=0.1,label='最高价')

#添加数据标签

for rect in b:

    w=rect.get_width()

    ax.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%float(w),ha='left',va='center')



    

b=ax.barh(y_pos+width*2, avgPriceList, align='center',color='green', ecolor='black',height=0.1,label='平均价')

#添加数据标签

for rect in b:

    w=rect.get_width()

    ax.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%float(w),ha='left',va='center')

    

#设置Y轴刻度线标签

ax.set_yticks(y_pos+width/1.5)

ax.set_xlabel('猪肉各部位品名 价格表:单位(斤)',fontproperties=font)

ax.set_title("数据采集为2019年12月21日",fontproperties=font)

ax.set_yticklabels(porkNameList,fontproperties=font)

plt.legend(prop=font)



plt.show()
# import requests

# import pandas as pd

# from tqdm import tqdm_notebook

# import os

# import json

# PAGE_SUM = 1

# FILE_NAME = 'top'



# if not os.path.exists(FILE_NAME):

#     os.mkdir(FILE_NAME)



# # 访问搜索详情信息的api地址， 返回json格式;

# #默认页面weibo_url = "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3d1%26q%3d%e7%8c%aa%e8%82%89&page_type=searchall"

# # 获取网页内容;

# for page in tqdm_notebook(range(PAGE_SUM)):

#     #不同页面后面加上page+1

#     url = "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3d1%26q%3d%e7%8c%aa%e8%82%89&page_type=searchall&page="+str(page+1)

#     json_file_path = os.path.join(FILE_NAME, '{}.json'.format(page+1))

#     if os.path.exists(json_file_path):  # 如果已经爬取

#         continue

#     while True:  # 一直尝试到成功

#         try:

#             response = requests.get(url, timeout=5)

#         except requests.exceptions.Timeout:

#             time.sleep(1)

#         if response.status_code == 200:

#             break

#     with open(json_file_path, 'w') as f:  # 写入本地文件

#         # indent 表明需要对齐，ensure_ascii=False 编码中文

#         f.write(json.dumps(json.loads(response.content.decode('utf-8')),

#                            indent=4, ensure_ascii=False))



import pandas as pd

import tqdm

import os

import json

PAGE_SUM = 10

FILE_NAME = 'top'

weibos = []

# for page in tqdm.notebook.tqdm(range(PAGE_SUM)):

#     json_file_path = os.path.join(FILE_NAME, '{}.json'.format(page+1))

#     with open(json_file_path) as f:

#         #print(json.load(f)['data']['cards'])

#         weibos += json.load(f)['data']['cards'] #存储微博数据



# with open('top.json', 'w') as f:  # 写入文件

#     f.write(json.dumps({'weibos': weibos}, indent=4, ensure_ascii=False))
with open('../input/dataset1/top.json', 'r') as f:

    data = json.load(f)

# print(data)

weibos=data['weibos']
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
from wordcloud import WordCloud

from matplotlib import pyplot as plt

%matplotlib inline



font_path = '../input/fontresource/simhei.ttf'

wc = WordCloud(font_path=font_path, background_color="white", max_words=1000,

               max_font_size=100, random_state=42, width=800, height=600, margin=2)

word_dict= {}

for word in text_all:

    if word not in word_dict:

        word_dict[word] = 1

    else:

        word_dict[word] += 1

wc.generate_from_frequencies(word_dict)





plt.figure(figsize=(10, 15),dpi=300)

plt.imshow(wc, interpolation="bilinear")  # 显示图片

plt.axis("off")