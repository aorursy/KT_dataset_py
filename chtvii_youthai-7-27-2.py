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
import random

import jieba
corpus_file = open("../input/santi.txt","r")

raw_text = corpus_file.readlines()

text = " "

for line in raw_text:

    text += line.strip() # 将lines加入空的text,并删除空格、回车

corpus_file.close()



split_mode = "jieba" # 选择分字或分词模式

if split_mode =="char":

    token_list = [char for char in text]

elif split_mode == "jieba":

    token_list = [word for word in jieba.cut(text)]

ngram_len = 3 # 即ngram算法中n的值
ngram_dict = {}

for i in range(1,ngram_len):

    for j in range(len(token_list) - i - 1):

        key = "".join(token_list[j:j+i+1]) # key与value都是string

        value = "".join(token_list[j+i+1])

        if key not in ngram_dict: # dic初始化

            ngram_dict[key] = {}

        if value not in ngram_dict[key]:

            ngram_dict[key][value] = 0

        ngram_dict[key][value] += 2**i
start_text = "程心和"

gen_len = 200 # 生成200位文字

topn = 5 # 5个预测值



if split_mode =="char":

    word_list = [char for char in start_text]

elif split_mode == "jieba":

    word_list = [word for word in jieba.cut(start_text)]



for i in range(gen_len):

    temp_list = [] # 为5个预测值建表格

    for j in range(1,ngram_len):

        if j >= len(word_list):

            continue

        prefix = "".join(word_list[-(j+1):]) # 预考虑的文字历史

        if prefix in ngram_dict:

            temp_list.extend(sorted(ngram_dict[prefix].items(),key=lambda d:d[1],reverse=True)[:topn]) 

            # 将value降序排序取，前五放入temp_list（通过lamda函数提取）

    next_word = random.choice(sorted(temp_list,key=lambda d:d[1],reverse=True)[:topn])[0]

    # 将temp_list中前五随机挑一个加入word_list

    word_list.append(next_word)



print("".join(word_list))