import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import json 

import os

import gc



%matplotlib inline 

sns.set()
print(os.listdir("../input"))
sub = pd.read_csv('../input/sample_submission.csv')
with open('../input/train_annotations.json/train_annotations.json','r') as anno_train:

    train = json.load(anno_train)

    

with open('../input/test_annotations.json/test_annotations.json','r') as anno_test:

    test = json.load(anno_test)
train.keys()
test.keys()
test_df = pd.DataFrame() 

test_df = test_df.append(test['images'], ignore_index=True)
train_df = pd.DataFrame() 

train_df = train_df.append(train['images'], ignore_index=True)

train_df_anno = pd.DataFrame() 

train_df_anno = train_df_anno.append(train['annotations'], ignore_index=True)

train_df['category_id'] = train_df_anno['category_id']

del train_df_anno

gc.collect()
test_df.head()
train_df.head()
sub.head()
print(len(train_df)/len(test_df))
category_numbers =  [0]*(np.max(train_df['category_id'])+1)
for i in train_df['category_id']:

    category_numbers[i]+=1
plt.xlabel('class')

plt.ylabel('number of samples')

x = list(range(len(category_numbers)))

plt.bar(x[1:], category_numbers[1:])
max(category_numbers[1:])
category_numbers[0]
plt.xlabel('class')

plt.ylabel('number of samples')

plt.plot(x[1:], sorted(category_numbers[1:], reverse=True))
few_shot = 0

for i in category_numbers[1:]:

    if i <= 200:

        few_shot+=1
few_shot/len(category_numbers)