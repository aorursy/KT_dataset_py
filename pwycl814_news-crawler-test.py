# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install newspaper3k
import tqdm

import json

import pandas as pd

from newspaper import Article



data = pd.read_csv("../input/123-news/123.csv")

df = pd.DataFrame(data)

df.head
# 利用load()方法从json文件中读取数据并存储为python对象

try:

    with open('texts.json','r') as f:

        datas = json.load(f)

        ind=datas[-1][0]+1

except:

    datas=[]

    ind=0
# 利用load()方法从json文件中读取数据并存储为python对象

# with open('texts.json','r') as f:

#     datas = json.load(f)



texts = []



for i in tqdm.tqdm(range(ind,len(df))):

# for i in tqdm.tqdm(range(ind,ind+100)):

    url = df.iloc[i,1]

    news = Article(url)

    

    try: 

        news .download() #先下载

        news .parse()#再解析

        text = news.text

    except :

        text = ''

    texts.append((i,text))

    

    if i%100 == 0:

        with open('texts.json','w') as fp:

            datas.extend(texts)

            json.dump(datas,fp)

            texts=[]



len(texts)
import json



with open('texts.json','w') as fp:

    datas.extend(texts)

    json.dump(datas,fp)
print(datas[-1][0])

print(texts[-1][0])



for i in datas:

    print(i[0])