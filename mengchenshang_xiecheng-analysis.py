import os

import json

import sqlite3

import pandas as pd

from tqdm import tqdm_notebook



# # 读取 JSON 文件，并插入数据库

# conn = sqlite3.connect('xiecheng.db')

# cursor = conn.cursor()

# # 创建一个最低价格的表，包含出发城市、到达城市、日期、最低价格

# CREATE_COMMAND2 = '''

# CREATE table LOWEST_PRICE (

#     start_city varchar(1000),

#     arrival_city varchar(1000),

#     date varchar(1000),

#     price float,

#     primary key(start_city, arrival_city, date)

# );

# '''

# # 插入数据

# INSERT_COMMAND2 = '''

# insert into LOWEST_PRICE values(?,?,?,?);

# '''

# try:

#     cursor.execute(CREATE_COMMAND2)

# except:

#     pass  # 表已创建

# json_path = 'lowest_price'

# # 遍历所有 JSON 文件

# for filename in tqdm_notebook(os.listdir(json_path)):

#     json_file_path = os.path.join(json_path, filename)

#     with open(json_file_path, 'r') as f:

#         start, arrival = filename.strip('.json').split('-')

#         oneday_data = json.loads(f.read())['data']['oneWayPrice']

#         if oneday_data == None:# 并不一定会有航班

#             continue

#         oneday_data = oneday_data[0]

#         insert_data = [(start, arrival, date, oneday_data[date])

#                        for date in oneday_data]

#         # 为提高速度，不查重插入数据，如果重复插入会报错，请只执行一次

#         try:

#             cursor.executemany(INSERT_COMMAND2, insert_data)

#         except:

#             pass

#         conn.commit()

# cursor.close()

# conn.close()
# conn = sqlite3.connect('xiecheng.db')



# #  读取数据库并导出 csv 文件

# def load_data(cursor):

#     # 导出最低价格

#     cursor.execute('SELECT * FROM LOWEST_PRICE')

#     result = cursor.fetchall()

#     # 构建 dataframe

#     lowest_price_df = pd.DataFrame(

#         data=result, columns=['start', 'arrival', 'date', 'price'])

#     # 导出机票信息

#     cursor.execute('SELECT * FROM AIRPLANE')

#     result = cursor.fetchall()

#     # 构建机票价格的 dataframe

#     airplane_df = pd.DataFrame(data=result,

#                                columns=['company_name', 'start_time', 'arrival_time', 

#                                         'start_airport', 'arrival_airport', 'airplane_type', 

#                                         'onetime_rate', 'airplane_number', 'price', 'date'])

#     return lowest_price_df, airplane_df



# # 保存到 CSV 数据文件

# lowest_price_df, airplane_df = load_data(conn.cursor())

# lowest_price_df.to_csv('xiecheng_lowest_price.csv', encoding='utf_8_sig')

# airplane_df.to_csv('xiecheng_airplane.csv', encoding='utf_8_sig')

# "保存完成."
# 加载数据

lowest_price_df = pd.read_csv(

    "../input/xiechengairplane/xiecheng_lowest_price.csv", index_col=0).drop_duplicates()  # 去重

airplane_df = pd.read_csv("../input/xiechengairplane/xiecheng_airplane.csv", index_col=0)

len(lowest_price_df), len(airplane_df)
import datetime

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# 从北京出发的最低价格

chengdu_lowest_price_df = lowest_price_df[lowest_price_df['start'] == '成都']

# 从 2018-10-23开始

BASE_DATE = datetime.datetime(2018, 10, 23, 0, 0)

'''

在此生成一个二维矩阵，X 轴是日期，Y 轴是城市。

因此需要对应日期到 X 轴的坐标，城市对应到 Y 轴的坐标，以字典保存。

具体的点则为价格

'''

# 日期对应下标，原始数据以 20190101 的格式，按照 %Y%m%d 解析成 Python 数据，获取与最早的时间之间的天数差

date_dict = {date: (datetime.datetime.strptime(date, "%Y%m%d")-BASE_DATE).days

             for date in chengdu_lowest_price_df['date'].drop_duplicates().values.astype(str)}

# 城市对应下标，排序是为了防止去重时顺序发生变化

city_name_dict = {city_name: index

                  for index,

                  city_name in enumerate(sorted(chengdu_lowest_price_df['arrival'].drop_duplicates()))}

# 使用一个矩阵表明城市和时间的对应，相应位置为价格

data = np.zeros((len(city_name_dict), len(date_dict)))

for _, row in chengdu_lowest_price_df.iterrows():  # 遍历最低价格

    arrival_index = city_name_dict[row['arrival']]  # 城市的坐标

    date_index = date_dict[str(row['date'])]  # 日期的坐标

    price = row['price']

    # 设置该坐标下的价格

    data[arrival_index, date_index] = price



# 最后绘制出热力图

plt.figure(figsize=(20, 30))

sns.heatmap(data)
list(city_name_dict.keys())[145]
list(city_name_dict.keys())[184]
# 成都 → 上海

condition1 = (lowest_price_df['start'] == '成都') & (

    lowest_price_df['arrival'] == '上海')

# 上海 → 成都

condition2 = (lowest_price_df['start'] == '上海') & (

    lowest_price_df['arrival'] == '成都')

chengdu_to_shanghai_lowest = lowest_price_df[condition1]['price'].values

shanghai_to_chengdu_lowest = lowest_price_df[condition2]['price'].values

X = np.arange(chengdu_to_shanghai_lowest.shape[0])  # X 轴是天数

# 绘制折线图

plt.plot(X, chengdu_to_shanghai_lowest, label='Chengdu to Shanghai')

plt.plot(X, shanghai_to_chengdu_lowest, label='Shanghai to Chengdu')

plt.legend()
# 选出从成都双流机场出发

from_chengdu_df = airplane_df[airplane_df.start_airport.str.contains(r'双流*')]

# 选出从上海出发，非双流机场

from_shanghai_df = airplane_df[~airplane_df.start_airport.str.contains(r'双流*')]



def trans(start_time, time_period=8):

    # 将起飞时间分为三个阶段，每八个小时一个阶段

    curr_hour = datetime.datetime.strptime(start_time, "%H:%M").hour

    return curr_hour//time_period



# 作用于 DataFrame 的 start_time 这一列，获取时间分段之后的结果

from_chengdu_period = from_chengdu_df['start_time'].apply(trans)

from_shanghai_period = from_shanghai_df['start_time'].apply(trans)

# 更新原始的 DataFrame

from_chengdu_df['start_time'] = from_chengdu_period

from_shanghai_df['start_time'] = from_shanghai_period

from_chengdu_df.head()  # 预览从成都出发
# 定义两个重复使用的绘图参数

width = 0.2

pos = [0, 1, 2]



fig, ax = plt.subplots(figsize=(8, 6))

plt.bar(pos, [dict(from_chengdu_period.value_counts())[i]

              for i in range(3)], width, label='graph 1', color='g')

plt.bar([p + width for p in pos], [dict(from_shanghai_period.value_counts())[i]

                                   for i in range(3)], width, label='graph 2', color='r')

# 设置标签和距离

ax.set_ylabel('Price')

ax.set_title('Airplane Counts By Time Period')

ax.set_xticks([p + 1 * width for p in pos])

ax.set_xticklabels(['00:00-08:00', '08:00-16:00', '16:00-24:00'])

# 设置 x，y 轴限制

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.legend(['Chengdu To Shanghai', 'Shanghai To Chengdu'], loc='upper left')
def plot(value):

    pos = list(range(value.shape[0]//3))

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(pos, [value[p*3] for p in pos], label='00:00-08:00', color='g')

    plt.plot(pos, [value[p*3+1] for p in pos], label='08:00-16:00', color='r')

    plt.plot(pos, [value[p*3+2] for p in pos], label='16:00-24:00', color='b')

    plt.xlabel('date')

    plt.ylabel('price')

    base = datetime.date(2018, 10, 30)

    numdays = 80

    date_list = [base + datetime.timedelta(days=x)

                 for x in range(0, numdays, 15)]

    plt.xticks(list(range(0, numdays, 15)), date_list, rotation=0)

    plt.legend()



# 根据日期和出发时间分组，出发时间为 0, 1, 2, 对应早中晚三阶段。

from_shanghai_mean_value = from_shanghai_df.groupby(['date', 'start_time'])['price'].mean()

# 绘制曲线

plot(from_shanghai_mean_value)
# 获取最低价格

from_shanghai_lowest_value = from_shanghai_df.groupby(['date', 'start_time'])['price'].min()

plot(from_shanghai_lowest_value)
import matplotlib.font_manager as fm



myfont = fm.FontProperties(fname='../input/plotfonts/SourceHanSerifK-Light.otf')

# 统计数量

company_data = dict(from_chengdu_df['company_name'].value_counts())

# 绘图

fig, ax = plt.subplots(figsize=(8, 6))

plt.bar(range(len(company_data)), company_data.values(), 0.4)

ax.set_ylabel('price')

ax.set_title('airplane counts by company')

ax.set_xticks(range(len(company_data)))

ax.set_xticklabels(company_data.keys(), fontproperties=myfont)
# 根据日期和航空公司名分组，获取到该航空公司在当日的最低价格

from_chengdu_min_value_bycompany = from_chengdu_df.groupby(

    ['date', 'company_name'])['price'].min()

class_count = 10

pos = list(range(from_chengdu_min_value_bycompany.shape[0]//class_count))

# 所有航空公司名

labels = ['上海航空', '东方航空', '中国国航', '南方航空', '吉祥航空',

          '四川航空', '成都航空', '春秋航空', '深圳航空', '西藏航空']

fig, ax = plt.subplots(figsize=(16, 12))

for i in range(class_count):

    plt.plot(pos, [from_chengdu_min_value_bycompany[p*class_count+i]

                   for p in pos], label=labels[i])

plt.xlabel('Date')

plt.ylabel('Price')

# 从 2018-10-30 之后的 80 天数据。

base = datetime.date(2018, 10, 30)

numdays = 80

date_list = [base + datetime.timedelta(days=x) for x in range(0, numdays, 15)]

plt.xticks(list(range(0, numdays, 15)), date_list, rotation=0)

plt.legend(prop=myfont)
# 航空公司名使用数字唯一编码

from_chengdu_df['company_name'] = pd.Series(

    pd.factorize(from_chengdu_df['company_name'])[0])

# 出发机场编码

from_chengdu_df['start_airport'] = pd.Series(

    pd.factorize(from_chengdu_df['start_airport'])[0])

# 飞机的类型编码

from_chengdu_df['airplane_type'] = pd.Series(

    pd.factorize(from_chengdu_df['airplane_type'])[0])

# 出发日期唯一编码

from_chengdu_df['date'] = pd.Series(pd.factorize(from_chengdu_df['date'])[0])

from_chengdu_df.head()
plt.figure(figsize=(7, 7))

sns.heatmap(from_chengdu_df.corr(), square=True,

            annot=True)  # corr() 函数计算皮尔逊相关系数