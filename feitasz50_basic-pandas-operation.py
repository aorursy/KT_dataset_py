# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
kouri_df = pd.read_csv('../input/kouridatautf.csv')

kouri_df.head()
#get detail row

aomori_kouridetail_df = kouri_df[(kouri_df['業種'].str.match('\d\d\d')) & (kouri_df['都道府県']=='02_青森県')]

aomori_kouridetail_df.head()
# rename mistake column name 事業者数 to 事業所数

aomori_kouridetail_df.rename(columns={'事業者数': '事業所数'},inplace=True)
aomori_kouridetail_df.info()
aomori_kouridetail_df.describe()
oroshi_df = pd.read_csv('../input/oroshidatautf.csv')

oroshi_df.head()
aomori_oroshidetail_df = oroshi_df[(oroshi_df['業種'].str.match('\d\d\d')) & (oroshi_df['都道府県']=='02_青森県')]

aomori_oroshidetail_df.head()
aomori_oroshidetail_df.info()
aomori_oroshidetail_df['年間商品販売'].unique()
aomori_oroshidetail_df['年間商品販売'] = aomori_oroshidetail_df['年間商品販売'].replace('x',0).astype(int)

aomori_oroshidetail_df.head()
aomori_oroshidetail_df.info()
aomori_df = pd.concat([aomori_oroshidetail_df, aomori_kouridetail_df], join_axes=[aomori_oroshidetail_df.columns])

aomori_df
aomori_df.sort_values("年間商品販売",ascending=False)