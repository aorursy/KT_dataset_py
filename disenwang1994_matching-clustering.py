# Pandas for managing datasets
import pandas as pd

# Matplotlib for additional customization
from matplotlib import pyplot as plt

import numpy as np
# Seaborn for plotting and styling
import seaborn as sns


import datetime 
from collections import Counter
import re
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly import tools
import seaborn as sns
from PIL import Image
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train-labels/train_labels.csv', index_col=0)
# remove [ and ] from label lists
df['labels'] = df['labels'].str[1:]
df['labels'] = df['labels'].str[:-1]
df['labels'] = df['labels'] + ','
df_count = Counter(" ".join(df["labels"]).split(',')).most_common(200)
all_label_count = pd.DataFrame(df_count)
all_label_count.columns = ['label','count']
all_label_count['percentage'] = all_label_count['count']/len(df.index)
print (all_label_count.head())

import time
script_start_time = time.time()

import pandas as pd
import numpy as np
import json
import gc

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.plotly as py
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
plt.rcParams["figure.figsize"] = 12,8
sns.set(rc={'figure.figsize':(20,12)})
plt.style.use('fivethirtyeight')

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
import warnings
warnings.filterwarnings('ignore')

# Data path
data_path = '.'

# 1. Load data =================================================================
print('%0.2f min: Start loading data'%((time.time() - script_start_time)/60))
train={}
test={}
validation={}
with open('../input/imaterialist-challenge-fashion-2018/train.json',encoding='utf-8') as json_data:
    train= json.load(json_data)
with open('../input/imaterialist-challenge-fashion-2018/test.json',encoding='utf-8') as json_data:
    test= json.load(json_data)
with open('../input/imaterialist-challenge-fashion-2018/validation.json',encoding='utf-8') as json_data:
    validation = json.load(json_data)

print('Train No. of images: %d'%(len(train['images'])))
print('Test No. of images: %d'%(len(test['images'])))
print('Validation No. of images: %d'%(len(validation['images'])))

# JSON TO PANDAS DATAFRAME
# train data
train_img_url=train['images']
train_img_url=pd.DataFrame(train_img_url)
train_ann=train['annotations']
train_ann=pd.DataFrame(train_ann)
train=pd.merge(train_img_url, train_ann, on='imageId', how='inner')

# test data
test=pd.DataFrame(test['images'])

# Validation Data
val_img_url=validation['images']
val_img_url=pd.DataFrame(val_img_url)
val_ann=validation['annotations']
val_ann=pd.DataFrame(val_ann)
validation=pd.merge(val_img_url, val_ann, on='imageId', how='inner')

del (train_img_url, train_ann, val_img_url, val_ann)
gc.collect()

print('%0.2f min: Finish loading data'%((time.time() - script_start_time)/60))
print('='*50)



train.index += 1
def match_labels(google_label):
    t=df[df['labels'].str.contains(google_label)]
    safe2 = pd.merge(t, train, left_index = True, right_index = True)
    l = []
    for index, row in safe2.iterrows():
        for i in row['labelId']:
            l.append(i)
    df_count2 = Counter("".join(str(l)).split(','))
    for key, cnts in list(df_count2.items()):   # list is important here
        if cnts < 0.05*len(l):
            del df_count2[key]

    #print (df_count2)
    tem = []
    for i in df_count2:
        tem.append(re.findall(r'\d+',i))
    #for i in tem:
     #   print (i[0])

    #for i in tem:
     #   for n in top_labels:
      #          if n == i[0]:
       #             tem.remove(i)

    final = []
    for i in tem:
        final.append(i[0])
    #print (final)
    return (final)
all_label_count2 = all_label_count.iloc[10:]
print (all_label_count2.head())
all_label_count2['matched']=''
for index, row in all_label_count2.iterrows():
    all_label_count2.set_value(index, 'matched', match_labels(row['label']))
print (all_label_count2.head())
# remove [ and ] from label lists
all_label_count2['matched'] = all_label_count2['matched'].astype(str)
all_label_count2['matched'] = all_label_count2['matched'].str[1:]
all_label_count2['matched'] = all_label_count2['matched'].str[:-1]
all_label_count2['matched'] = all_label_count2['matched'] + ','
print (all_label_count2.head())
all_label_count2['matched'] = all_label_count2['matched'].astype(str)
all_label_count2['matched'] = all_label_count2['matched'].map(lambda x: ''.join([i for i in x if i.isdigit() or i.isspace()]))
all_label_count2['matched'] = all_label_count2['matched'] + ' '
print (all_label_count2['matched'].head())
test = pd.read_csv('../input/labeled-test/labeled_test.csv', index_col=0,encoding = "ISO-8859-1")
test['labels'] = test['labels'].str[1:]
test['labels'] = test['labels'].str[:-1]
print (test.head())
test['prediction'] = ''
for index, trow in test.iterrows():
    for index, arow in all_label_count2.iterrows():
        if arow['label'] in trow['labels']:
            trow['prediction']=trow['prediction']+arow['matched']
df = train.head(10000).drop(columns=['url'])
# print (df['labelId'])
df['labelId'] = df['labelId'].astype(str)

# Note that the result of this block takes a while to show
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer( max_features=200000,
                                  stop_words='english',
                                 use_idf=True)

%time tfidf_matrix = tfidf_vectorizer.fit_transform(df['labelId']) #fit the vectorizer to synopses


print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
len(terms)
from sklearn.cluster import KMeans

num_clusters = 60

km = KMeans(n_clusters=num_clusters)

%time km.fit(tfidf_matrix)

clusters = km.labels_.tolist()


df = train.head(10000).drop(columns=['url'])

df_cluster = pd.DataFrame(clusters)
df_cluster.columns = ["cluster"]
print (df_cluster.head())
df = pd.merge(df, df_cluster, left_index = True, right_index = True)

df.index += 1

train_labels = pd.read_csv('../input/train-labels/train_labels.csv', index_col=0)

print (train_labels.head())
df = pd.merge(df, train_labels, left_index = True, right_index = True)
df['labels'] = df['labels'].str[1:]
df['labels'] = df['labels'].str[:-1]
df['labels'] = df['labels'] + ','
print (df.head())
x = df.groupby('cluster')['labels'].apply(lambda x: x.sum())

x.columns = ["labels", "frequent_labels"]

x = x.to_frame()

x.columns = ["labels"]
x['frequent_labels']=""
x['labels'] = x['labels'].astype(str).replace("''", "")
print (x.head())
from collections import Counter
for index, row in x.iterrows():
    df_count = Counter("".join(row['labels']).split(',')).most_common(5)
    l = []
    for i in df_count:
        l.append(i[0])
    row['frequent_labels']=l
y = df.groupby('cluster')['labelId'].apply(lambda x: x.sum())
y = y.to_frame()
y.columns = ["wish_labels"]
y['frequent_wish_labels']=""

print (y.head())
for index, row in y.iterrows():
    df_count = Counter(",".join(row['wish_labels']).split(',')).most_common(10)
    l = []
    for i in df_count:
        l.append(i[0])
    row['frequent_wish_labels']=l
cluster_train = pd.concat([x,y],axis=1)
cluster_train= cluster_train[['frequent_labels','frequent_wish_labels']]
cluster_add = cluster_train
cluster_add['frequent_wish_labels'] = cluster_add['frequent_wish_labels'].astype(str)
cluster_add['frequent_wish_labels'] = cluster_add['frequent_wish_labels'].map(lambda x: ''.join([i for i in x if i.isdigit() or i.isspace()]))
test['cluster'] = ""
for index, row in test.iterrows():
    x = 0
    for cdex,crow in cluster_train.iterrows():
        for n in crow['frequent_wish_labels']:
            n = str(n)
            
            if n in str(row['prediction']):
                x +=1
    if x >= 5:
        test.set_value(index, 'cluster', str(row['prediction']) +" " + str(cluster_add.at[cdex,'frequent_wish_labels']))
for index, row in test.iterrows():
    row['cluster'] = row['cluster'].split(" ")
    test.set_value(index, 'cluster', set(row['cluster']))
test['cluster'] = test['cluster'].astype(str)
test['cluster'] = test['cluster'].str[1:]
test['cluster'] = test['cluster'].str[:-1]
test['cluster'] = test['cluster'] + ','
test['cluster'] = test['cluster'].map(lambda x: ''.join([i for i in x if i.isdigit() or i.isspace()]))
print (test['cluster'].head())
