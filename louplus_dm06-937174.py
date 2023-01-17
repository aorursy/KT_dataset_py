import pandas as pd

import jieba

from tqdm import tqdm_notebook

from wordcloud import WordCloud

import numpy as np

from gensim.models import Word2Vec

import warnings



warnings.filterwarnings('ignore')
df = pd.read_csv('https://s3.huhuhang.com/temporary/b1vzDs.csv')
df.shape
#获取的数据会有重复的情况，首先根据酒店的名字将一项，将名称完全相同的项从数据表中删除

df = df.drop_duplicates(['HotelName'])

df.info()
df_new_hotel = df[df["HotelCommentValue"]==0].drop(['Unnamed: 0'], axis=1).set_index(['index'])

df_new_hotel.head()
df_in_ana = df[df["HotelCommentValue"]!=0].drop(["Unnamed: 0", "index"], axis=1)

df_in_ana.shape
import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

%matplotlib inline



plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.distplot(df_in_ana['HotelPrice'].values)
df_in_ana['HotelLabel'] = df_in_ana["HotelPrice"].apply(lambda x: '奢华' if x > 1000 else \

                                                        ('高端' if x > 500 else \

                                                        ('舒适' if x > 300 else \

                                                        ('经济' if x > 100 else '廉价')))) 
hotel_label = df_in_ana.groupby('HotelLabel')['HotelName'].count()

plt.pie(hotel_label.values, labels=hotel_label.index, autopct='%.1f%%', explode=[0, 0.1, 0.1, 0.1, 0.1], shadow=True)
hotel_distribution = df_in_ana.groupby('HotelLocation')['HotelName'].count().sort_values(ascending=False)

hotel_distribution = hotel_distribution[:8]

hotel_label_distr = df_in_ana.groupby([ 'HotelLocation','HotelLabel'])['HotelName'].count().sort_values(ascending=False).reset_index()

in_use_district = list(hotel_distribution.index)

hotel_label_distr = hotel_label_distr[hotel_label_distr['HotelLocation'].isin(in_use_district)]



fig, axes = plt.subplots(1, 5, figsize=(17,8))

hotel_label_list = ['高端', '舒适', '经济', '奢华', '廉价']

for i in range(len(hotel_label_list)):

    current_df = hotel_label_distr[hotel_label_distr['HotelLabel']==hotel_label_list[i]]

    axes[i].set_title('{}型酒店的区域分布情况'.format(hotel_label_list[i]))

    axes[i].pie(current_df.HotelName, labels=current_df.HotelLocation, autopct='%.1f%%', shadow=True)
df_in_ana['HotelCommentLevel'] = df_in_ana["HotelCommentValue"].apply(lambda x: '超棒' if x > 4.6 \

                                                                      else ('还不错' if x > 4.0 \

                                                                      else ('一般般' if x > 3.0 else '差评' )))
hotel_label_level = df_in_ana.groupby(['HotelCommentLevel','HotelLabel'])['HotelName'].count().sort_values(ascending=False).reset_index()

fig, axes = plt.subplots(1, 5, figsize=(17,8))

for i in range(len(hotel_label_list)):

    current_df = hotel_label_level[hotel_label_level['HotelLabel'] == hotel_label_list[i]]

    axes[i].set_title('{}型酒店的评分情况'.format(hotel_label_list[i]))

    axes[i].pie(current_df.HotelName, labels=current_df.HotelCommentLevel, autopct='%.1f%%', shadow=True)
# 廉价酒店

df_pos_cheap = df_in_ana[(df_in_ana['HotelLabel']=='廉价') \

                         & (df_in_ana['HotelCommentValue']> 4.6) \

                         & (df_in_ana['HotelCommentAmount']> 500)].sort_values(by=['HotelPrice'], ascending=False)

df_pos_cheap
# 经济型酒店

df_pos_economy = df_in_ana[(df_in_ana['HotelLabel']=='经济') \

                         & (df_in_ana['HotelCommentValue']> 4.6) \

                         & (df_in_ana['HotelCommentAmount']> 2000)].sort_values(by=['HotelPrice'])

df_pos_economy
# 舒适型酒店

df_pos_comfortable = df_in_ana[(df_in_ana['HotelLabel']=='舒适') \

                         & (df_in_ana['HotelCommentValue']> 4.6) \

                         & (df_in_ana['HotelCommentAmount']> 1000)].sort_values(by=['HotelPrice'])

df_pos_comfortable
# 高端酒店

df_pos_hs = df_in_ana[(df_in_ana['HotelLabel']=='高端') \

                         & (df_in_ana['HotelCommentValue']> 4.6) \

                         & (df_in_ana['HotelCommentAmount']> 1000)].sort_values(by=['HotelPrice'])

df_pos_hs
# 奢华酒店

df_pos_luxury = df_in_ana[(df_in_ana['HotelLabel']=='奢华') \

                         & (df_in_ana['HotelCommentValue']> 4.6) \

                         & (df_in_ana['HotelCommentAmount']> 500)].sort_values(by=['HotelPrice'])

df_pos_luxury
df_neg = df_in_ana[(df_in_ana['HotelCommentValue'] < 3.0) \

                         & (df_in_ana['HotelCommentAmount'] > 50)].sort_values(by=['HotelPrice'], ascending=False)

df_neg
!wget -nc "http://labfile.oss.aliyuncs.com/courses/1176/fonts.zip"

!unzip -o fonts.zip
from wordcloud import WordCloud



def get_word_map(hotel_name_list):

    word_dict ={}

    for hotel_name in tqdm_notebook(hotel_name_list):

        hotel_name = hotel_name.replace('(', '')

        hotel_name = hotel_name.replace(')', '')

        word_list = list(jieba.cut(hotel_name, cut_all=False))

        for word in word_list:

            if word == '大连' or len(word) < 2:

                continue

            if word not in word_dict:

                word_dict[word] = 0

            word_dict[word] += 1

    

    font_path = 'fonts/SourceHanSerifK-Light.otf'

    wc = WordCloud(font_path=font_path, background_color='white', max_words=1000, 

                            max_font_size=120, random_state=42, width=800, height=600, margin=2)

    wc.generate_from_frequencies(word_dict)

    

    return wc
part1 = df_in_ana[df_in_ana['HotelPrice'] <= 150]['HotelName'].values

part2 = df_in_ana[df_in_ana['HotelPrice'] > 500]['HotelName'].values

fig, axes = plt.subplots(1, 2, figsize=(15, 8))

axes[0].set_title('价格较低酒店的名字词云')

axes[0].imshow(get_word_map(part1), interpolation='bilinear')

axes[1].set_title('价格较高酒店的名字词云')

axes[1].imshow(get_word_map(part2), interpolation='bilinear')
df_in_ana['HotelPrice'].median()
df_in_ana['PriceLabel'] = df_in_ana['HotelPrice'].apply(lambda x:1 if x <= 150 else 0)

df_new_hotel['PriceLabel'] = df_new_hotel['HotelPrice'].apply(lambda x:1 if x <= 150 else 0)
# 设定分词方式

def word_cut(x):

    x = x.replace('（', '')  # 去掉名称中出现的（）

    x = x.replace('）', '')

    return jieba.lcut(x)
#设置训练集和测试集

x_train = df_in_ana['HotelName'].apply(word_cut).values

y_train = df_in_ana['PriceLabel'].values

x_test = df_new_hotel['HotelName'].apply(word_cut).values

y_test = df_new_hotel['PriceLabel'].values
# 通过Word2Vec方法建立词向量浅层神经网络模型，并对分词之后的酒店名称进行词向量的求和计算

from gensim.models import Word2Vec

import warnings



warnings.filterwarnings('ignore')

w2v_model = Word2Vec(size=200, min_count=10)

w2v_model.build_vocab(x_train)

w2v_model.train(x_train, total_examples=w2v_model.corpus_count, epochs=5)



def sum_vec(text):

    vec = np.zeros(200).reshape((1, 200))

    for word in text:

        try:

            vec += w2v_model[word].reshape((1, 200)) 

        except KeyError:

            continue

    return vec 



train_vec = np.concatenate([sum_vec(text) for text in tqdm_notebook(x_train)])
# 构建神经网络分类器模型，并使用training data对模型进行训练

from sklearn.externals import joblib

from sklearn.neural_network import MLPClassifier

from IPython import display 



model = MLPClassifier(hidden_layer_sizes=(100, 50, 20), learning_rate='adaptive')

model.fit(train_vec, y_train)



# 绘制损失变化曲线，监控损失函数的变化过程

display.clear_output(wait=True)

plt.plot(model.loss_curve_)
# 之后对测试集进行词向量求和

test_vec = np.concatenate([sum_vec(text) for text in tqdm_notebook(x_test)])
# 用训练好的模型进行预测， 将结果倒入测试用表中

y_pred = model.predict(test_vec)

df_new_hotel['PredLabel'] = pd.Series(y_pred)
# 建模预测的结果

from sklearn.metrics import accuracy_score



accuracy_score(y_pred, y_test)
new_hotel_questionable = df_new_hotel[(df_new_hotel['PriceLabel'] ==0) & (df_new_hotel['PredLabel']==1)]

new_hotel_questionable = new_hotel_questionable.sort_values(by='HotelPrice', ascending=False)

new_hotel_questionable
plt.figure(figsize=(15, 7))

plt.imshow(get_word_map(new_hotel_questionable['HotelName'].values), interpolation='bilinear')
new_hotel_distri = df_new_hotel.groupby('HotelLocation')['HotelName'].count().sort_values(ascending=False)[:7]

plt.pie(new_hotel_distri.values, labels=new_hotel_distri.index, autopct='%.1f%%', shadow=True)
df_new_hotel['HotelLabel'] = df_new_hotel["HotelPrice"].apply(lambda x: '奢华' if x > 1000 \

                                                              else ('高端' if x > 500 \

                                                              else ('舒适' if x > 300 \

                                                              else('经济' if x > 100 \

                                                              else '廉价')))) 

new_hotel_label = df_new_hotel.groupby('HotelLabel')['HotelName'].count()

plt.pie(new_hotel_label.values, labels=new_hotel_label.index, autopct='%.1f%%', explode=[0, 0.1, 0.1, 0.1, 0.1], shadow=True)
df2 = df_new_hotel.groupby('HotelLabel')['HotelPrice'].mean().reset_index()

df1=df_in_ana.groupby('HotelLabel')['HotelPrice'].mean().reset_index()

price_change_percent = (df2['HotelPrice'] -  df1['HotelPrice'])/df1['HotelPrice'] * 100

plt.title('新开各档次酒店均价变化')

plt.bar(df1['HotelLabel'] ,price_change_percent, width = 0.35)

plt.ylim(-18, 18)

for x, y in enumerate(price_change_percent):

    if y < 0:

        plt.text(x, y, '{:.1f}%'.format(y), ha='center', fontsize=12, va='top')

    else:

        plt.text(x, y, '{:.1f}%'.format(y), ha='center', fontsize=12, va='bottom')