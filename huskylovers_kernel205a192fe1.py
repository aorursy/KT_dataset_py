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
from collections import Counter,defaultdict

import re

import gc

import seaborn as sns

from tqdm import tqdm

from math import log
prepared_data = pd.read_csv('/kaggle/input/prepared-word/combied_word.csv')

prepared_data.head()
def select_the_chinese(x): #提取‘’间的位置

    p = re.compile(r'\'(.*?)\'')

    chinese_list = re.findall(p,x)

    return chinese_list

prepared_data['two_combined_word'] = prepared_data['two_combined_word'].apply(lambda x: select_the_chinese(x))

prepared_data['thr_combied_word'] = prepared_data['thr_combied_word'].apply(lambda x: select_the_chinese(x))

prepared_data['split_word'] = prepared_data['split_word'].apply(lambda x: select_the_chinese(x))
prepared_data.head()
split_data = list(prepared_data['split_word'])

two_combined_data = list(prepared_data['two_combined_word'])

thr_combined_data = list(prepared_data['thr_combied_word'])
word_counter = Counter([word for sentence in two_combined_data for word in sentence] + [word for sentence in thr_combined_data for word in sentence]+

                      [word for sentence in split_data for word in sentence])
new_word_df = pd.DataFrame({'new_word':list(word_counter.keys()),'count':list(word_counter.values())})

new_word_df.head()
L_counter = defaultdict(list)

for sentence in tqdm(prepared_data['split_word']):

    for i in range(len(sentence)):

        if i == 0:

            L_counter[sentence[i]].append('开始')

            L_counter[''.join(sentence[i:i+2])].append('开始')

            L_counter[''.join(sentence[i:i+3])].append('开始')

        else:

            L_counter[sentence[i]].append(sentence[i-1])

            if i < len(sentence) - 2:

                L_counter[''.join(sentence[i:i+2])].append(sentence[i-1])

            if i < len(sentence) - 3:

                L_counter[''.join(sentence[i:i+3])].append(sentence[i-1])



for i,j in tqdm(L_counter.items()):

    try_array = np.array(list(Counter(j).values()))

    try_array = try_array / np.sum(try_array)

    try_array = np.sum(-try_array * np.log(try_array))

    L_counter[i] = try_array
new_word_df['IE_L'] = new_word_df['new_word'].apply(lambda x:L_counter[x])

del L_counter

gc.collect()
R_counter = defaultdict(list)

for sentence in tqdm(prepared_data['split_word']):

    for i in range(len(sentence)):

        if i == len(sentence)-1:

            R_counter[sentence[i]].append('结束')

            R_counter[''.join(sentence[i-1:i+1])].append('结束')

            R_counter[''.join(sentence[i-2:i+1])].append('结束')

        else:

            R_counter[sentence[i]].append(sentence[i+1])

            if i > 0:

                R_counter[''.join(sentence[i-1:i+1])].append(sentence[i+1])

            if i > 1:

                R_counter[''.join(sentence[i-2:i+1])].append(sentence[i+1])



for i,j in tqdm(R_counter.items()):

    try_array = np.array(list(Counter(j).values()))

    try_array = try_array / np.sum(try_array)

    try_array = np.sum(-try_array * np.log(try_array))

    R_counter[i] = try_array
new_word_df['IE_R'] = new_word_df['new_word'].apply(lambda x:R_counter[x])

del R_counter

gc.collect()
new_word_df.head()
positive_word = []

with open('/kaggle/input/postive-word/positive_dict.txt','r') as fr:

    for line in fr.readlines():

        positive_word.append(line[:-1])

positive_word[:10]
new_word_df['tag'] = new_word_df['new_word'].apply(lambda x:1 if x in positive_word else 0)
#词频比较

print(new_word_df['count'].describe())

print(new_word_df[new_word_df['tag']==1]['count'].describe())
#辅助数据

assist_df = pd.read_csv('/kaggle/input/prepared-word/assist.csv')

assist_df.head()
assist_df['two'] = assist_df['two'].apply(lambda x:select_the_chinese(x))

assist_df['thr'] = assist_df['thr'].apply(lambda x:select_the_chinese(x))

assist_df['four'] = assist_df['four'].apply(lambda x:select_the_chinese(x))

assist_df.head()
try_list = list(assist_df['two'])

two_try = []

for _ in try_list:

    two_try += _

two_try = list(set(two_try))
try_list = list(assist_df['thr'])

thr_try = []

for _ in try_list:

    thr_try += _

thr_try = list(set(thr_try))
try_list = list(assist_df['four'])

four_try = []

for _ in try_list:

    four_try += _

four_try = list(set(four_try))
AV_R_count = defaultdict(int)

AV_L_count = defaultdict(int)



for data in tqdm(two_try):

    try_data = data.split()

    AV_R_count[try_data[0]] += 1

    AV_L_count[try_data[1]] += 1



for data in tqdm(thr_try):

    try_data = data.split()

    AV_R_count[try_data[0]+try_data[1]] += 1

    AV_L_count[try_data[1]+try_data[2]] += 1

for data in tqdm(four_try):

    try_data = data.split()

    AV_R_count[try_data[0]+try_data[1]+try_data[2]] += 1

    AV_L_count[try_data[1]+try_data[2]+try_data[3]] += 1
new_word_df['AV_L'] = new_word_df['new_word'].apply(lambda x: AV_L_count[x])

new_word_df['AV_R'] = new_word_df['new_word'].apply(lambda x: AV_R_count[x])
new_word_df.head()
#AV_L作比较

print(new_word_df[new_word_df['tag']==1]['AV_L'].describe())

print(new_word_df['AV_L'].describe())
#AV_R作比较

print(new_word_df[new_word_df['tag']==1]['AV_R'].describe())

print(new_word_df['AV_R'].describe())
prepared_data.head()
words_ = defaultdict(int)

pairs_ = defaultdict(int)

consist_p = defaultdict(float)

words = list(prepared_data['split_word'])

total = 0

for sentecne in tqdm(words):

    if len(sentecne)>0:

        words_[sentecne[0]] += 1

        for i in range(len(sentecne) - 1):

            words_[sentecne[i+1]] += 1

            pairs_[' '.join(sentecne[i:i+2])] += 1

            total += 1

for i,j in tqdm(pairs_.items()):

    k = i.split()

    i = i.replace(' ','')

    consist_p[i] = log(total*j/(words_[k[0]]*words_[k[1]]))
thrs_ = defaultdict(int)

consist_p_max = defaultdict(float)

consist_p_min = defaultdict(float)



words = list(prepared_data['split_word'])

total = 0

for sentecne in tqdm(words):

    if len(sentecne)>0:

        for i in range(len(sentecne) - 2):

            thrs_[' '.join(sentecne[i:i+3])] += 1

            total += 1



for i,j in tqdm(thrs_.items()):

    former,later = ' '.join(i.split()[:2]), ' '.join(i.split()[1:])

    former_word, later_word = i.split()[2], i.split()[0]

    try:

        try_former = log(total*j/(pairs_[former]* words_[former_word]))

        try_later = log(total*j/(pairs_[later] * words_[later_word]))

        i = i.replace(' ','')

        consist_p_max[i] = max([try_former,try_later])

        consist_p_min[i] = min([try_former,try_later])

    except:

        print(i)
c_p_max = {}

c_p_max.update(consist_p)

c_p_max.update(consist_p_max)



c_p_min = {}

c_p_min.update(consist_p)

c_p_min.update(consist_p_min)



c_max = []

c_min = []

for word in tqdm(new_word_df['new_word']):

    try:

        c_max.append(c_p_max[word])

        c_min.append(c_p_min[word])

    except:

        c_max.append(None)

        c_min.append(None)

new_word_df['c_p_max'] = c_max

new_word_df['c_p_min'] = c_min

new_word_df.head()
#c_p_max作比较

print(new_word_df[new_word_df['tag']==1]['c_p_max'].describe())

print(new_word_df['c_p_max'].describe())
#c_p_min作比较

print(new_word_df[new_word_df['tag']==1]['c_p_min'].describe())

print(new_word_df['c_p_min'].describe())
new_word_df.to_csv('new_word.csv')