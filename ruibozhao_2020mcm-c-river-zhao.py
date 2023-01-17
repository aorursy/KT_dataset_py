import numpy as np 

import pandas as pd

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import os



# load file

data = pd.read_csv("../input/ini_review_dryer.csv")

data.info()
sell_counts = data.groupby(['product_id'],as_index=False)['product_id'].agg({'sell_cnt':'count'})

sell_counts
# DateFrame 类型

# 加权平均评论分数

df_avg_score = data.groupby("product_id").apply(lambda x: np.average(x['score'], weights=x['weight'])).to_frame()

# 加权平均分 rating

df_avg_star = data.groupby("product_id").apply(lambda x: np.average(x['star_rating'], weights=x['weight'])).to_frame()



df_tmp = pd.merge(df_avg_score, df_avg_star, how='left', on='product_id')

df = pd.merge(df_tmp, sell_counts, how='left', on='product_id')

# df = pd.merge(df_tmp2, df_s_rate, how='left', on='product_id')



# rename

df.columns = ['product_id','avg_score','avg_star','sell_cnt']

df
count = df['sell_cnt']

avg_score = df['avg_score']

avg_star = df['avg_star']
plt.scatter(avg_star,count)

plt.xlabel('avg_star')

plt.ylabel('sell_count')
plt.scatter(avg_score,count)

plt.xlabel('avg_score')

plt.ylabel('sell_count')
plt.scatter(avg_star,count)

plt.xlabel('avg_star')

plt.ylabel('sell_count')

plt.xlim(0.7,0.9)

plt.ylim(0,200)
plt.scatter(avg_score,count)

plt.xlabel('avg_score')

plt.ylabel('sell_count')

plt.xlim(55,65)

plt.ylim(0,200)