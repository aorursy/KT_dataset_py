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
samples = 1000
df_raw = pd.read_csv('/kaggle/input/safebooru/all_data.csv', nrows = samples)

df_raw.head(samples)
df_raw.info()
features = ["sample_url", "tags"]

df_X = df_raw[features]

df_X.columns
df_X.head(samples)
import re

tag = []



for i in df_X.tags:

    tokens = re.split("[ ]",i)

    for token in tokens:

        if token not in tag:

            tag.append(token)

print("There are", len(tag), "different tags")



tag[:10]
chose = []

dic = {}

for i in df_X.tags:

    tokens = re.split("[ ]",i)

    for token in tokens:

        if token in ['1girl', 'bag','black_hair','blush','bob_cut']:

            chose.append(token)

for j in chose:

     dic[j] = dic.get(j,0)+1      

print("5 tags occurences:",dic)
dict = {}

list = []

for i in df_X.tags:

    tokens = re.split("[ ]",i)

    for token in tokens:

        list.append(token)

for i in list:

    dict[i] = dict.get(i,0)+1



item = sorted(dict.items(), key = lambda x:x[1],reverse = True)

print("10 top tags:")

for i in range(0,10):

    print(item[i])

    