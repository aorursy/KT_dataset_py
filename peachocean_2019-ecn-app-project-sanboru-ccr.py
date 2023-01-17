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

print("50 top tags:")

for i in range(0,50):

    print(item[i])

    
tag_girl = [("girl" in i.split() or "1girl" in i.split()) for i in df_X.tags]

tag_boy = [("male" in i.split() or "boy" in i.split() or "1boy" in i.split())for i in df_X.tags]

tag_solo = [("solo" in i.split() or "1girl" in i.split() or "1boy" in i.split()) for i in df_X.tags]

target = []



for i in range(0,len(tag_girl)):

    value = 0 # encode not valid sample

    if tag_solo[i] and tag_girl[i] != tag_boy[i]:

        if tag_girl[i]:

            value = 1 # 1 encode girl sample

        else:

            value = 2 # 2 encode boy sample

    target.append(value)



df_X["target"] = target

df_X
df_X = df_X[df_X.target != 0]

df_X.drop(['tags'], axis = 1, inplace = True)

df_X
!mkdir "train"

!ls
import requests

 

def download_url(url):

  # assumes that the last segment after the / represents the file name

  # if the url is http://abc.com/xyz/file.txt, the file name will be file.txt

    file_name_start_pos = url.rfind("/") + 1

    file_name = "train/" + url[file_name_start_pos:]

 

    r = requests.get(url, stream=True)

    if r.status_code == requests.codes.ok:

        with open(file_name, 'wb') as f:

            for data in r:

                f.write(data)

    return file_name
urls = ["http:" + i for i in df_X.sample_url]
from multiprocessing.pool import ThreadPool

pd.options.mode.chained_assignment = None



# Run 5 multiple threads. Each call will take the next element in urls list

result = ThreadPool(5).imap_unordered(download_url, urls)

total = 0

pct =0

for r in result:

    total += 1

    if total % int(0.1*df_X.shape[0]) == 0:

        pct += 10

        print(pct,"% downloaded")
images_path = [url[url.rfind("/") + 1:] for url in urls]

images_path[:10]
df_X["file"] = images_path

df_X.drop(['sample_url'], axis=1, inplace=True)

df_X
from IPython.display import display, Image



fille = df_X[df_X.target == 1].head(10).file

for x in fille:

    display(Image("train/"+x))

    
garcon = df_X[df_X.target == 2].head(12).file

for x in garcon:

    display(Image("train/"+x))
df_X.to_csv("sanboru_gender_dataset.csv", index=False)
import shutil

shutil.make_archive("sanboru_gender_dataset", 'zip', "train")

!rm train/*

!rm -d "train"
!ls