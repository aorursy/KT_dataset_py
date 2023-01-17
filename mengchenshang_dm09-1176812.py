#!/usr/bin/env python

# encoding: utf-8

from __future__ import unicode_literals

import pandas as pd

from bs4 import BeautifulSoup

import requests

import time

import json

#  伪装配置

headers = {

    "Accept-Encoding":"gzip",

    "Cache-Control": "max-age=0",

    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36',

    "Accept-Language":  "zh-CN,zh;q=0.8,en;q=0.6,en-US;q=0.4,zh-TW;q=0.2",

    "Connection" :  "keep-alive",

    "Accept-Encoding" :  "gzip, deflate",

    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"

}



# 输出文件路径

file_output = 'cars.csv'



# 主域名,用于拼接完整URL

domain = "http://car.autohome.com.cn"



# 最开始的品牌入口

start_url = 'http://car.autohome.com.cn/AsLeftMenu/As_LeftListNew.ashx?typeId=1%20&brandId=0%20&fctId=0%20&seriesId=0'



# 每一页下载等待时间

wait_sec = 6
def get_cars(brand_name, start_url):

    print ('start_url', start_url)

    cars = []

    # 设置referer

    headers['referer'] = start_url



    # 设置起始抓取页面

#     start_url = 'http://car.autohome.com.cn/price/brand-34.html'

    now_url = start_url

    # next_url 为空是结束抓取，返回数据的条件

    next_url = ''

    while True:

        result = requests.get(now_url, headers=headers)

#         print(result.request.headers)

        html_content = result.content

        html_content = result.content.decode('gbk',errors='ignore').encode('utf-8')

        html_content_soup = BeautifulSoup(html_content, 'html.parser')

        cars_tag = html_content_soup.find_all(class_='main-lever')#list-cont-main

        infos_tag = html_content_soup.find_all(class_='list-cont-bg')#list-cont-main

        # 结束逻辑

        # 1. 一开始就没有翻页

        # 2. 唯一获取 page-item-next

        # 3. 循环

        if html_content_soup.find(class_="price-page") is None:

            next_url = ''

        else:

            next_url_tag = html_content_soup.find(class_="price-page").find(class_="page-item-next")

            # 结束翻页

            if next_url_tag['href'] == 'javascript:void(0)':

                next_url = ''

            else:

                next_url = domain + next_url_tag['href']

        print('next_url is ', next_url)

        for car_tag in cars_tag:

            for info_tag in infos_tag:

                car = {}

                car['brand'] = brand_name

                car['url'] = now_url

                car['name'] = info_tag.find(class_='font-bold').get_text(strip=True)

                car['price'] = car_tag.find(class_='font-arial').get_text(strip=True)

                car['score'] = info_tag.find(class_='score-cont').get_text(strip=True)

                # @TODO 颜色还有问题

                for car_attr_tag  in car_tag.find('ul', class_='lever-ul').find_all('li'):

                    car_attr = car_attr_tag.get_text(',',strip=True)

                    if len(car_attr.split(u'：')) < 2 :

                        continue

                    car_attr_key = car_attr.split(u'：')[0]

                    car_attr_value = car_attr.split(u'：')[1]

                    # 直接空格无效，因为gbk无法转换'\xa0'字符(http://www.educity.cn/wenda/350839.html)

                    car_attr_key = car_attr_key.replace(u'\xa0', '')

                    car[car_attr_key] = car_attr_value.strip(',')

                cars.append(car)

        time.sleep(wait_sec)

        # 抓取结束，返回数据

        if next_url == '':

            return cars



        # 更换页面

        now_url = next_url

df = pd.DataFrame({'brand': [],

          'url': [],

          'name': [],

          'price': [],

          'score': [],

          '级别': [],

          '车身结构': [],

          '发 动 机': [],

          '变 速 箱': [],

          '外观颜色': []

        })

df.to_csv('cars.csv',encoding='utf_8_sig',index=False)



# df2 = pd.DataFrame({'brand': [],'name': [],'score': []})

# df2.to_csv('cars_score.csv',encoding='utf_8_sig',index=False)

# 第一步提取 品牌列表

# 第二部通过品牌列表提取 车辆详细列表(下一页)





#print url

# 设置隐藏爬虫痕迹

result = requests.get(start_url, headers=headers)

# result.encoding = 'gbk'  ## @TODO 这一行干嘛用的

# @TODO 验证是否下载成功

html_content = result.content

html_content = html_content.decode('gbk','ignore').encode('utf-8')

# @TODO 保存原始文档

# beautifulsoup 设置解析器（不然迁移可能出错）

html_content_soup = BeautifulSoup(html_content,'html.parser')

brands_tag =  html_content_soup.find_all('li')



###爬虫时间较长  数据已上传

# for brand_tag in brands_tag:

    

#     cars = []

#     brand_name  = brand_tag.get_text(',').split(',')[0]

#     brand_href = domain + brand_tag.a['href']

#     cars =  get_cars(brand_name, brand_href)

#     # 输出中文问题

#     for car in cars:

#         df1 = pd.DataFrame.from_dict(car,orient='index').T

#         df1.to_csv(file_output, index=False, mode='a+', header=False,encoding='utf_8_sig')
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('../input/autohomedataset/cars_data.csv', index_col=0)

df = df.reset_index()

df.isnull().sum()
df['外观颜色'].fillna('暂无',inplace= True)

df['price'].replace('暂无报价','0.00万',inplace= True) #统一标签格式，方便之后的价格提取

df.drop('url',axis=1, inplace=True) #不需要地址数据
df['mean'] = float(0.00)

df['low']  = float(0.00)

df['high'] = float(0.00)

for i in range (len(df['score'])):

    df['外观颜色'][i] = ",".join(df['外观颜色'][i].split(',')[0:3])#保留前三个颜色

    #清洗用户评分数据

    if df['score'][i][-2:] == '暂无':

        df['score'][i] = df['score'][i][-2:]

    else:

        df['score'][i] = float(df['score'][i][-4:])

    #清洗价格，转换成float类型，价格区间取均值

    if len(df['price'][i])==5:

        df['mean'][i] =float(df['price'][i][:-1])

        df['low'][i] = float(df['price'][i][:-1])

        df['high'][i] = float(df['price'][i][:-1])

    elif len(df['price'][i])==6:

        df['mean'][i] =float(df['price'][i][:-1])

        df['low'][i] = float(df['price'][i][:-1])

        df['high'][i] = float(df['price'][i][:-1])

    elif len(df['price'][i])==7:

        df['mean'][i] =float(df['price'][i][:-1])

        df['low'][i] = float(df['price'][i][:-1])

        df['high'][i] = float(df['price'][i][:-1])

    elif len(df['price'][i])==8:

        df['mean'][i] =float(df['price'][i][:-1])

        df['low'][i] = float(df['price'][i][:-1])

        df['high'][i] = float(df['price'][i][:-1])

    elif len(df['price'][i])==10:

        df['mean'][i] =(float(df['price'][i][0:4])+float(df['price'][i][5:-1]))/2

        df['low'][i] = float(df['price'][i][0:4])

        df['high'][i] = float(df['price'][i][5:-1])

    elif len(df['price'][i])==11:

        df['mean'][i] =(float(df['price'][i][0:4])+float(df['price'][i][5:-1]))/2

        df['low'][i] = float(df['price'][i][0:4])

        df['high'][i] = float(df['price'][i][5:-1])

    elif len(df['price'][i])==12:

        if df['price'][i][4]=='-':

            df['mean'][i] =(float(df['price'][i][0:4])+float(df['price'][i][5:-2]))/2

            df['low'][i] = float(df['price'][i][0:4])

            df['high'][i] = float(df['price'][i][5:-2])

        elif df['price'][i][6]=='-':

            df['mean'][i] =(float(df['price'][i][0:6])+float(df['price'][i][7:-1]))/2

            df['low'][i] = float(df['price'][i][0:6])

            df['high'][i] = float(df['price'][i][7:-1])

        else:

            df['mean'][i] =(float(df['price'][i][0:5])+float(df['price'][i][6:-1]))/2

            df['low'][i] = float(df['price'][i][0:5])

            df['high'][i] = float(df['price'][i][6:-1])

    elif len(df['price'][i])==13:

        df['mean'][i] =(float(df['price'][i][0:5])+float(df['price'][i][6:-1]))/2

        df['low'][i] = float(df['price'][i][0:5])

        df['high'][i] = float(df['price'][i][6:-1])

    elif len(df['price'][i])==14:

        if df['price'][i][5]=='-':

            df['mean'][i] =(float(df['price'][i][0:5])+float(df['price'][i][6:-3]))/2

            df['low'][i] = float(df['price'][i][0:5])

            df['high'][i] = float(df['price'][i][6:-3])

        else:

            df['mean'][i] =(float(df['price'][i][0:6])+float(df['price'][i][7:-1]))/2

            df['low'][i] = float(df['price'][i][0:6])

            df['high'][i] = float(df['price'][i][7:-1])

    elif len(df['price'][i])==15:

        df['mean'][i] =(float(df['price'][i][0:4])+float(df['price'][i][7:-3]))/2

        df['low'][i] = float(df['price'][i][0:4])

        df['high'][i] = float(df['price'][i][7:-3])

    elif len(df['price'][i])==16:

        df['mean'][i] =(float(df['price'][i][0:6])+float(df['price'][i][7:-3]))/2

        df['low'][i] = float(df['price'][i][0:6])

        df['high'][i] = float(df['price'][i][7:-3])

    else:

        print(df['price'][i])

        continue
#重新处理score为数字格式

df['score'].replace('暂无',float(0.00),inplace= True)

#统计下得分

df['score'].value_counts(normalize=True)
df.describe()
#价格和评分的相关度

print(df['score'].corr(df['high']))

print(df['score'].corr(df['low']))

print(df['score'].corr(df['mean']))
import matplotlib

from matplotlib import pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline
# fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 6))

# sns.relplot(x="high", y="low", kind="line", data=df)

plt.figure(figsize=(18, 8))

sns.regplot(x="low", y="score", data=df)
sns.pairplot(data=df)
#中文字体修复

myfont = matplotlib.font_manager.FontProperties(fname="../input/character/msyh.ttc")

# matplotlib.rcParams['axes.unicode_minus'] = False
# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

import plotly.offline as py                    #保存图表，相当于plotly.plotly as py，同时增加了离线功能

py.init_notebook_mode(connected=True)          #离线绘图时，需要额外进行初始化

import plotly.graph_objs as go                 #创建各类图表

import plotly.figure_factory as ff             #创建table

car_type= df['级别'].value_counts()

labels = ['紧凑型车','紧凑型SUV','小型SUV','中型SUV','中型车',

          'MPV','微面','微卡','轻客','皮卡','中大型SUV','中大型车',

          '跑车','小型车','微型车','大型车','大型SUV','紧凑型MPV']

trace = go.Bar(

    x = labels,

    y = car_type

)

data = [trace]

layout = go.Layout(title = '汽车之家-车型数量')

fig = go.Figure(data = trace, layout = layout)

py.iplot(fig, filename='汽车之家-车型数量')
car_type = df['级别'].value_counts(normalize=True)

labels = ['紧凑型车','紧凑型SUV','小型SUV','中型SUV','中型车',

          'MPV','微面','微卡','轻客','皮卡','中大型SUV','中大型车',

          '跑车','小型车','微型车','大型车','大型SUV','紧凑型MPV']

values = car_type

trace = go.Pie(labels=labels, values=values)

layout = go.Layout(title = '汽车之家-车型分布')

fig = go.Figure(data = trace, layout = layout)

py.iplot(fig, filename='汽车之家-车型分布')
#删除无得分车型

df_drop = df.drop(index=(df.loc[(df['score']==0)].index)).reset_index(drop=True)

#评分前10车型

top_ten = df_drop.sort_values(by='score',ascending=False).head(10)

top_ten
#评分后10的车型

tail_ten=df_drop.sort_values(by='score',ascending=False).tail(10)

tail_ten
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(2,1,1)

                 

sns.set(style="ticks")

# sns.set(font='SimHei')#sns.set(font='Droid Sans Fallback') #中文字体

sns.scatterplot(x='low',y='high',data=top_ten,hue='name')

plt.xticks(fontproperties=myfont)

plt.yticks(fontproperties=myfont)

plt.legend(prop=myfont)

ax1.set_title("评分前10车型价格分布（万元）",fontproperties=myfont)





ax2 = fig.add_subplot(2,1,2)    

sns.set(style="ticks")

# sns.set(font='SimHei')

sns.scatterplot(x='low',y='high',data=tail_ten,hue='name')

plt.xticks(fontproperties=myfont)

plt.yticks(fontproperties=myfont)

plt.legend(prop=myfont)

ax2.set_title("评分后10车型价格分布（万元）",fontproperties=myfont)

x = pd.DataFrame([df_drop.index])

x_val = x.T

labels = ['', '', '', '', '', '']

labels_position1 = [1000,2000,4000,6000,8000,9000]

labels[0]=df_drop['brand'][1000]

labels[1]=df_drop['brand'][2000]  

labels[2]=df_drop['brand'][4000]  

labels[3]=df_drop['brand'][6000]  

labels[4]=df_drop['brand'][8000]  

labels[5]=df_drop['brand'][9000]  

fig, axes = plt.subplots(figsize=(15,8))

# plt.rcParams['font.sans-serif']=['Droid Sans Fallback'] #用来正常显示中文标签

plt.scatter(x_val,df_drop['score'],marker=',')

axes.set_title("厂商整体评分分布",fontproperties=myfont)

plt.xlabel("汽车厂商",fontproperties = myfont)

plt.ylabel("评分",fontproperties = myfont)

plt.xticks(labels_position1, labels, rotation='vertical',fontproperties=myfont)

plt.show()
df_data_price = pd.DataFrame(df['mean']).set_index(df['brand'])

df_data_score = pd.DataFrame(df['score']).set_index(df['brand'])

df_data_full = pd.concat([df_data_price,df_data_score],axis=1)

# 获取 5 个国产车标签及对应的坐标刻度

china_labels = ['哈弗', '比亚迪', '东风', '红旗', '吉利汽车']

# 获取 5 个合资车标签及对应的坐标刻度

foreign_labels = ['奥迪', '奔驰', '宝马', '福特', '大众']

# 初始化坐标

labels_position1 = [1,2,3,4,5]

labels_position2 = [1,2,3,4,5]

for i in range(len(df_data_full)):

    if df_data_full.index[i] in china_labels:

            for x in range(len(china_labels)):

                if df_data_full.index[i] == china_labels[x]:

                    china_labels[x]=df_data_full.index[i]

                    labels_position1[x] = i

                else: 

                    continue

                    

for i in range(len(df_data_full)):

    if df_data_full.index[i] in foreign_labels:

            for x in range(len(foreign_labels)):

                if df_data_full.index[i] == foreign_labels[x]:

                    foreign_labels[x]=df_data_full.index[i]

                    labels_position2[x] = i

                else: 

                    continue



# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签



fig, axes = plt.subplots(figsize=(15,8))

df_data_full['mean'].plot(

    kind='line',

    title='国产厂商价格分布',

    ax=axes

)

plt.xlabel("汽车厂商",fontproperties=myfont)

plt.ylabel("价格",fontproperties=myfont)

plt.xticks(labels_position1, china_labels, rotation='vertical',fontproperties=myfont)

plt.show()



fig, axes = plt.subplots(figsize=(15,8))

df_data_full['mean'].plot(

    kind='line',

    title='合资厂商价格分布',

    ax=axes

)

plt.xlabel("汽车厂商",fontproperties=myfont)

plt.ylabel("价格",fontproperties=myfont)

plt.xticks(labels_position2, foreign_labels, rotation='vertical',fontproperties=myfont)

plt.show()
df_data_price1 = pd.DataFrame(df['mean']).set_index(df['name'])

df_data_score1 = pd.DataFrame(df['score']).set_index(df['name'])

df_data_full1 = pd.concat([df_data_price1,df_data_score1],axis=1)

trace = go.Bar(

    x = df['name'][:100],

    y = df_data_full1['mean'][:100]

)

data = [trace]



py.iplot(data, filename='basic-bar')
#每个车型保留一种颜色

for i in range (len(df['score'])):

    df['外观颜色'][i] = "".join(df['外观颜色'][i].split(',')[0])
df_color = df['外观颜色'].drop(index=(df.loc[(df['外观颜色']=='暂无')].index)).reset_index(drop=True)

car_color = df_color.value_counts().sort_values(ascending=False)[:20]

data = [go.Bar(

            x=car_color,

            y=['珍珠白','珠光白','极地白','白色','皓月白','北极白' ,'水晶白',

               '矿石白','雪域白','雪山白','典雅白','朱鹭白','象牙白','冰川白',

               '黑色','冰晶白','曜岩黑','月光白','极光白','塔夫绸白'],

            orientation = 'h')]

layout = go.Layout(

            title = '汽车颜色Top20')

figure = go.Figure(data = data, layout = layout)

py.iplot(figure, filename='汽车颜色Top20')