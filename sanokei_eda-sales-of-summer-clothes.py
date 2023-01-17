import numpy as np 

import pandas as pd

pd.set_option("max_columns",100)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
df = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
df.head(3)
df.tail(3)
null = df.isnull().sum().to_frame(name='nulls').T

dtype = df.dtypes.to_frame(name='dtypes').T

nunique = df.nunique().to_frame(name='unique').T

pd.concat([null, dtype, nunique], axis=0)
df = df.drop(['has_urgency_banner','urgency_text','merchant_profile_picture',

             'product_url','product_picture','product_id','merchant_id','title',

             'currency_buyer','theme','crawl_month'], 1)
round(df.describe(), 2)
def pivot_table_rating(columns):

    df_pivot = pd.pivot_table(df, index=columns, values=['rating_count','rating_five_count','rating_four_count',

                                                           'rating_three_count','rating_two_count','rating_one_count'],

                                                            aggfunc=np.sum)

    

    df_pivot['5_ratio'] = df_pivot['rating_five_count'] / df_pivot['rating_count']

    df_pivot['4_ratio'] = df_pivot['rating_four_count'] / df_pivot['rating_count']

    df_pivot['3_ratio'] = df_pivot['rating_three_count'] / df_pivot['rating_count']

    df_pivot['2_ratio'] = df_pivot['rating_two_count'] / df_pivot['rating_count']

    df_pivot['1_ratio'] = df_pivot['rating_one_count'] / df_pivot['rating_count']

    

    df_pivot['mean'] = (df_pivot['5_ratio']*5 + df_pivot['4_ratio']*4 + df_pivot['3_ratio']*3 + 

                        df_pivot['2_ratio']*2 + df_pivot['1_ratio']*1)

    return round(df_pivot[['5_ratio','4_ratio','3_ratio','2_ratio','1_ratio', 'mean']], 3)
df['b_price'] = (pd.cut(df['price'], [0,3,5,7,10,30,50]))

pivot_table_rating('b_price')
pivot_table_rating('uses_ad_boosts')
pivot_table_rating('shipping_is_express')
df['b_merchant_rating'] = (pd.cut(df['merchant_rating'], [2.5,3,3.5,4,4.5,5]))

pivot_table_rating('b_merchant_rating')
plt.figure(figsize=(20,4))

sns.barplot(data = pivot_table_rating('product_color').sort_values(by='mean', ascending=False),

            x=pivot_table_rating('product_color').index, y='mean')

plt.xticks(rotation=90)