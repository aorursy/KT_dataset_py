import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head() # 前面已经设置最多显示5行数据
# Your code here
# 查看points列的中值
reviews_points_median = reviews.points.median()
# 打印数据进行查看
print('reviews_points_median =',reviews_points_median)

check_q1(reviews_points_median)
# 查看reviews.points.describe()
# 打印数据进行查看
print('reviews_points_describe =\n',reviews.points.describe())
# Your code here
# 读入数据集中的国家
reviews_country = reviews.country
# 查看数据
print('reviews_country =\n',reviews_country)

check_q2(reviews_country)
# Your code here
# 取出经常出现的国家相应信息
reviews_country_counts = reviews.country.value_counts()
# 查看国家以及出现频率的数据
print('reviews_country_counts =\n',reviews_country_counts)
print(reviews_country_counts.shape)

check_q3(reviews_country_counts)
# 读取reviews.country.describe()
reviews_country_describe = reviews.country.describe()
# 打印并查看数据
print(reviews_country_describe)

# 查看reviews_country_describe.top,即出现频率次数最高的国家
print('reviews_country_describe.top =',reviews_country_describe.top)
print('reviews_country_maxfreq =',reviews_country_counts[0])
print('reviews_country_describe.freq =',reviews_country_describe.freq)
# Your code here
# 取出price中值
reviews_price_median = reviews.price.median()
# 打印price_median
print('reviews_price_median =',reviews_price_median)

# 重新映射数据(减去价格中值)
reviews_price_remap = reviews.price.map(lambda price: price - reviews_price_median)
# 打印并查看数据
print('reviews_price_remap =\n',reviews_price_remap)

check_q4(reviews_price_remap)
# Your code here
# 得到点对价比率数据(points-to-price radio)
points_price_radio = reviews.points / reviews.price
# 查看点对价比率数据
print('points_price_radio =\n',points_price_radio)

# 得到最大比率的索引(index)
index_max_radio = points_price_radio.idxmax()
print('index_max_radio =',index_max_radio)

# 返回最大比率的红酒名
reviews_radio_max = reviews.title[index_max_radio]
print('reviews_radio_max =',reviews_radio_max)

check_q6(reviews.title[index_max_radio])
# 得到比率最大的红酒的title
reviews_radio_max2 = reviews.title[(reviews.points / reviews.price).idxmax()]
print('reviews_radio_max2 =',reviews_radio_max2)
# Your code here
# 使用map函数进行判断是否含有‘tropical’或‘fruity’
description_istropical = reviews.description.map(lambda string: "tropical" in string)
# 查看description中每行是否含有'tropical'
print('description_istropical =\n',description_istropical)
print('\n')
# 统计description中出现‘tropical’的次数
tropical_counts = description_istropical.value_counts()
# 查看description中出现‘tropical’的次数
print('tropical_counts =\n',tropical_counts)
print('\n')

# 使用map函数进行判断是否含有‘fruity’
description_isfruity = reviews.description.map(lambda string: "fruity" in string)
# 查看description中每行是否含有‘fruity’
print('description_isfruity =\n',description_isfruity)
print('\n')
# 统计description中出现‘fruity’的次数
fruity_counts = description_isfruity.value_counts()
# 查看description中出现‘fruity’的次数
print('fruity_counts =\n',fruity_counts)
print('\n')

# 创建Series对象合并‘tropical’‘fruity’出现的次数
wine_taste_counts = pd.Series(data = [tropical_counts[True],fruity_counts[True]],index=['tropical','fruity'])
# 打印对象及维度
print('wine_taste_counts =\n',wine_taste_counts)
print('\n')
print('wine_taste_counts.shape =\n',wine_taste_counts.shape)

check_q7(wine_taste_counts)
print(answer_q7()) 
# Your code here
# 创建一个DataFrame存储‘country’和‘variety’非空的数据
data_notnull = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull()),['country','variety']]
# 打印并查看数据
print('data_notnull=\n',data_notnull)
print('\n')
# 进行apply操作:data_notnull此处为DataFrame对象，故无法使用map函数
ans_7 = data_notnull.apply(lambda data_notnull: data_notnull.country + " - " + data_notnull.variety,axis=1)
# 打印结果查看
print('ans_7=\n',ans_7)
print('\n')

# 查看ans_7的value_counts()
ans_7.value_counts()
print(ans_7.value_counts())

check_q8(ans_7.value_counts())
print(answer_q8())