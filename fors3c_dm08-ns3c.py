import requests

from bs4 import BeautifulSoup

from urllib.parse import urljoin

import pandas as pd

from time import sleep

import random



base_url = "http://www.xiachufang.com"



# 家常菜最受欢迎 https://www.xiachufang.com/category/40076/pop/?page=20



headers = {

    'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",

    'Accept-Encoding': "gzip, deflate, br",

    'Accept-Language': "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",

    'Cache-Control': "max-age=0",

    'Connection': "keep-alive",

    'Cookie': "bid=iXTvwEuO; __utmc=177678124; __utmz=177678124.1574647324.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); sajssdk_2015_cross_new_user=1; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2216ea04b5da99ea-00df3f41e15204-2393f61-2073600-16ea04b5daae08%22%2C%22%24device_id%22%3A%2216ea04b5da99ea-00df3f41e15204-2393f61-2073600-16ea04b5daae08%22%2C%22props%22%3A%7B%22%24latest_referrer%22%3A%22%22%2C%22%24latest_referrer_host%22%3A%22%22%2C%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%7D%7D; Hm_lvt_ecd4feb5c351cc02583045a5813b5142=1574647324; __utma=177678124.355896112.1574647324.1574647324.1574663874.2; __gads=Test; Hm_lpvt_ecd4feb5c351cc02583045a5813b5142=1574668188; __utmb=177678124.27.10.1574663874",

    'Host': "www.xiachufang.com",

    'Sec-Fetch-Mode': "navigate",

    'Sec-Fetch-Site': "none",

    'Sec-Fetch-User': "?1",

    'Upgrade-Insecure-Requests': "1",

    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"

}



def get_urls(url, page_num)->set:

    """

    :param page_num: 需要爬取的页数

    :return: 采集到的菜谱的url的集合（去重以后）

    """

    urls = []

    for i in range(1, page_num+1):

        res = requests.get(url.format(i), headers=headers)

        if res.status_code==200:

            soup = BeautifulSoup(res.content, features='lxml')

            for soup_ in soup.select("div.info.pure-u > p.name > a"):

                urls.append(urljoin(base_url, soup_.get('href')))



    return set(urls)



def get_cookbook_info(url):

    res = requests.get(url, headers=headers)

    soup = BeautifulSoup(res.text, features='lxml')

    title = soup.select_one('h1.page-title').get_text().strip()

    print(title)

    rating_value = 0

    rating_value_data = soup.select_one('div.stats.clearfix > div.score.float-left > span.number')

    if rating_value_data:

        rating_value = rating_value_data.get_text()

    cooked_number = 0

    cooked_number_data = soup.select_one('div.stats.clearfix > div.cooked.float-left > span.number')

    if cooked_number_data:

        cooked_number = cooked_number_data.get_text()

    ingredients = []

    raw_ingre_data = soup.select('table > tr > td.name')

    for td in raw_ingre_data:

        ingredients.append(td.get_text().strip())



    return [title, rating_value, cooked_number, ingredients]



# 数据已经采集完成，所以下方代码进行注释处理

# if __name__ == "__main__":

#     # f"https://www.xiachufang.com/category/40076/?page={i}" 最近流行

#     pop_url = "http://www.xiachufang.com/category/40076/pop/?page={}" # 24

#     most_url = "http://www.xiachufang.com/category/40076/?page={}"

#     most_urls = get_urls(most_url, 20) # 爬取20页流行家常菜

#     train_data = []

#     for url in list(most_urls):

#         print("Start to get data!\n")

#         train_data.append(get_cookbook_info(url))

#         print("Sleeping...")

#         sleep(random.randint(5, 10))



#     most_df = pd.DataFrame(data=None,columns=['title', 'rating_value', 'cooked_number', 'ingredients'])

#     for data in train_data:

#         most_df = pd.concat([most_df, pd.DataFrame(data=[data], columns=['title', 'rating_value', 'cooked_number', 'ingredients'])])

#     most_df.to_csv('most.csv', index=False)



#     pop_urls = get_urls(pop_url, 24)

#     test_data = []

#     for url in list(pop_urls):

#         if url not in most_urls:

#             print("Start to get data!\n")

#             test_data.append(get_cookbook_info(url))

#             print("Sleeping...")

#             sleep(random.randint(5, 10))

#     test_df = pd.DataFrame(data=None, columns=['title', 'rating_value', 'cooked_number', 'ingredients'])

#     for data in test_data:

#         test_df = pd.concat([test_df, pd.DataFrame(data=[data], columns=['title', 'rating_value', 'cooked_number', 'ingredients'])])

#     test_df.to_csv('pop.csv', index=False)

import pandas as pd

# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



most_df = pd.read_csv('/kaggle/input/xcf-data/most.csv')

pop_df = pd.read_csv('/kaggle/input/xcf-data/pop.csv')



print(most_df.head())

print(pop_df.head())
# 为了显示完整的用料，设置较大的列显示宽度

pd.set_option('display.max_colwidth', 120)



cook_book_df = pd.concat((most_df, pop_df), ignore_index=True)

print("总共采集到{}个菜谱。".format(cook_book_df.shape[0]))

cook_book_df.reset_index()

# 查看一下整合起来的数据的最后5行。

cook_book_df.tail()
train_data = cook_book_df.loc[cook_book_df['rating_value']>0,'ingredients']

print('总共提取到{}个训练数据!'.format(train_data.shape[0]))

train_targets = cook_book_df.loc[cook_book_df['rating_value']>0, 'rating_value']

print(train_data.head())

print(train_targets.head())
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib



from matplotlib.font_manager import FontProperties



# a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# print(a)



# plt.rcParams['font.sans-serif'] = ['KaiTi']

# plt.rcParams['font.serif'] = ['KaiTi']



font = FontProperties(fname=r"/kaggle/input/wordfonts/SourceHanSerifK-Light.otf", size=14) # 先读取字体



# plt.rcParams['font.family'] = ['Droid Sans Fallback']

# plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号



plt.figure(figsize=(18, 8))



sns.lineplot(data=train_targets)

plt.title("评分分布", fontproperties=font)
predict_data = cook_book_df.loc[cook_book_df['rating_value']==0,['ingredients']]

print("获取到{}个待预测数据!".format(predict_data.shape[0]))

predict_data.head()
import numpy as np

import jieba

import jieba.analyse



def text_clean(ingredients):

    ingre_chars = str(np.array(ingredients).tolist())

    ingredients = jieba.analyse.extract_tags(ingre_chars, topK=10, withWeight=False)

    return ingredients



train_data = train_data.apply(text_clean)

# 展示一下清洗后的数据

train_data.head()
def ingre_queue(cuisine, ingredients={})->dict:

    for char in cuisine:

        if char not in ingredients.keys():

            ingredients[char] = 1

        else:

            ingredients[char] += 1

    return ingredients

            

sum_ingredients = {}

for element in train_data:

    sum_ingredients = ingre_queue(element, sum_ingredients)
# 下载字体，为词云展示做准备

!wget -nc "http://labfile.oss.aliyuncs.com/courses/1176/fonts.zip"

#!unzip -o fonts.zip

# 此处运行一直失败，改为直接上传字体
from wordcloud import WordCloud



from wordcloud import WordCloud

from matplotlib import pyplot as plt

%matplotlib inline



font_path = '/kaggle/input/wordfonts/SourceHanSerifK-Light.otf'

wc = WordCloud(font_path=font_path, background_color="white", max_words=200,

               max_font_size=120, random_state=42, width=1080, height=900, margin=3)



wc.generate_from_frequencies(sum_ingredients)





plt.figure(figsize=(20, 12))

plt.imshow(wc, interpolation="bilinear")  # 显示图片

plt.axis("off")
plt.style.use('ggplot')

plt.rcParams['font.family'] = ['Droid Sans Fallback']

#plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

fig = pd.DataFrame(sum_ingredients, index=[0]).transpose()[0].sort_values(ascending=False, inplace=False)[:30].plot(kind='barh', figsize=(20, 10))

fig.invert_yaxis()

fig = fig.get_figure()

fig.tight_layout()

fig.show()

def text_clean(ingre_list):

    return " ".join(i for i in ingre_list)



X_train_data = train_data.apply(text_clean)
X_train_data
from sklearn.feature_extraction.text import TfidfTransformer  

from sklearn.feature_extraction.text import CountVectorizer  



vectorizer=CountVectorizer()

 

transformer = TfidfTransformer()



train_tfidf = transformer.fit_transform(vectorizer.fit_transform(X_train_data)).todense()
print(train_tfidf[:10])
targets = np.array(train_targets).tolist()

targets[:10]
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(train_tfidf, train_targets, test_size=0.2, random_state=42)



X_train, y_valid
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error





model = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=True))])



model = model.fit(X_train, y_train)



print("参数权重:", model.named_steps['linear'].coef_)

print("截距为:", model.named_steps['linear'].intercept_)

y_valid_pred = model.predict(X_valid)

y_valid_pred
print("MSE为: ", mean_squared_error(y_valid, y_valid_pred))
# 对0分的菜谱进行评分预测



# step1 单词清洗

pred_data = predict_data['ingredients'].apply(text_clean)
pred_data
pred_rating = model.predict(transformer.transform(vectorizer.transform(X_train_data)).todense())

pred_rating