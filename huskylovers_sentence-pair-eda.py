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
import matplotlib.pyplot as plt

from wordcloud import WordCloud

import jieba

from collections import Counter

import Levenshtein

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/pkdata/pk/train.csv')

df_train.head()
print('total number of question pairs:{}'.format(len(df_train)))

print('positive tag:{}%'.format(round(df_train['label'].mean()*100, 2)))



question_series = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

print("question num:{}".format(len(question_series)))

print('unique question num:{}'.format(len(np.unique(question_series))))
plt.figure(figsize=(12, 5))

plt.hist(question_series.value_counts(), bins=50)

plt.yscale('log', nonposy='clip')

plt.title('Log-Histogram of question apperance counts')

plt.xlabel('Number of occurence of question')

plt.ylabel('Number of questions')
train_qs = question_series.astype(str)

dist_train = train_qs.apply(len)

plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=30, normed=True ,label='train')

plt.title('Normalised histogram of character count in questions', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)

print('mean train character length:{:.2f}'.format(dist_train.mean()))
train_qs = question_series.apply(lambda x: ' '.join(jieba.cut(x)).split())

dist_train = train_qs.apply(len)

plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=30, normed=True ,label='train')

plt.title('Normalised histogram of word count in questions', fontsize=15)

plt.legend()

plt.xlabel('Number of words', fontsize=15)

plt.ylabel('Probability', fontsize=15)

print('mean train character length:{:.2f}'.format(dist_train.mean()))
# from pylab import mpl

# mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体

# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
words = ' '.join(jieba.cut(" ".join(train_qs.astype(str))))

cloud = WordCloud(width=1440, height=1080, font_path='/kaggle/input/simhei/SimHei.ttf').generate(words)

plt.figure(figsize=(29, 15))

plt.imshow(cloud)

plt.axis('off')
qmarks = np.mean(train_qs.apply(lambda x: '?' in x or '吗' in x or '怎么' in x))

numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))



print('obvious question:{:.2f}%'.format(qmarks*100))

#print('Question with [math] tags:{:.2f}'.format(math))

print('Question with numbers:{:.2f}%'.format(numbers*100))
df_train['distance'] = df_train[['question1', 'question2']].apply(lambda x:

            Levenshtein.distance(x['question1'], x['question2']), axis=1)

df_train['distance1'] = df_train[['question1', 'question2']].apply(lambda x:

            Levenshtein.distance(x['question1'], x['question2']) / max(len(x['question1']), len(x['question2'])), axis=1)

df_train['distance2'] = df_train[['question1', 'question2']].apply(lambda x:

            Levenshtein.distance(x['question1'], x['question2']) / max(1, abs(len(x['question1'])-len(x['question2']))), axis=1)

df_train['ratio'] = df_train[['question1', 'question2']].apply(lambda x: Levenshtein.ratio(x['question1'], x['question2']), axis=1)

df_train['jaro'] = df_train[['question1', 'question2']].apply(lambda x: Levenshtein.jaro(x['question1'], x['question2']), axis=1)

df_train['jaro_winkler'] = df_train[['question1', 'question2']].apply(lambda x: Levenshtein.jaro_winkler(x['question1'], x['question2']), axis=1)

df_train.head()
q_list = list(np.unique(list(df_train['question1']) + list(df_train['question2'])))

q_id = {}

for idx, q in enumerate(q_list):

    q_id[q] = idx

df_train['q1_id'] = df_train['question1'].apply(lambda x: q_id[x])

df_train['q2_id'] = df_train['question2'].apply(lambda x: q_id[x])

df_train.head()
neighbor_matrix = np.zeros((len(q_id), len(q_id)))

df_graph = df_train[df_train['label'] == 1]

for index, row in df_graph.iterrows():

    i = row['q1_id']

    j = row['q2_id']

    neighbor_matrix[i, j] += 1
def compute_indot(text, ng_matrix, q_id):

    i = q_id[text]

    in_dot = np.sum(ng_matrix[i, :])

    return in_dot



def compute_outdot(text, ng_matrix, q_id):

    i = q_id[text]

    out_dot = np.sum(ng_matrix[:, i])

    return out_dot



df_train['q1_indot'] = df_train['question1'].apply(lambda x: compute_indot(x, neighbor_matrix, q_id))

df_train['q1_outdot'] = df_train['question1'].apply(lambda x: compute_outdot(x, neighbor_matrix, q_id))

df_train['q1_dot'] = df_train['q1_indot'] + df_train['q1_outdot']

df_train['q2_indot'] = df_train['question2'].apply(lambda x: compute_indot(x, neighbor_matrix, q_id))

df_train['q2_outdot'] = df_train['question2'].apply(lambda x: compute_outdot(x, neighbor_matrix, q_id))

df_train['q2_dot'] = df_train['q2_indot'] + df_train['q2_outdot']
df_train.head()
df_train.corrwith(df_train['label'])
def graph_feature(df):

    def q_index(df_train):

        q_list = list(np.unique(list(df_train['question1']) + list(df_train['question2'])))

        q_id = {}

        for idx, q in enumerate(q_list):

            q_id[q] = idx

        return q_id

    def ng_matrix(df_train, q_id):

        neighbor_matrix = np.zeros((len(q_id), len(q_id)))

        df_graph = df_train[df_train['label'] == 1]

        for index, row in df_graph.iterrows():

            i = row['q1_id']

            j = row['q2_id']

            neighbor_matrix[i, j] += 1

        return neighbor_matrix

    def compute_indot(text, ng_matrix, q_id):

        i = q_id[text]

        in_dot = np.sum(ng_matrix[i, :])

        return in_dot

    def compute_outdot(text, ng_matrix, q_id):

        i = q_id[text]

        out_dot = np.sum(ng_matrix[:, i])

        return out_dot

    df = df[['question1', 'question2', 'label']]

    q_id = q_index(df)

    df['q1_id'] = df['question1'].apply(lambda x: q_id[x])

    df['q2_id'] = df['question2'].apply(lambda x: q_id[x])

    neighbor_matrix = ng_matrix(df, q_id)

    df['q1_indot'] = df['question1'].apply(lambda x: compute_indot(x, neighbor_matrix, q_id))

    df['q1_outdot'] = df['question1'].apply(lambda x: compute_outdot(x, neighbor_matrix, q_id))

    df['q1_dot'] = df['q1_indot'] + df['q1_outdot']

    df['q2_indot'] = df['question2'].apply(lambda x: compute_indot(x, neighbor_matrix, q_id))

    df['q2_outdot'] = df['question2'].apply(lambda x: compute_outdot(x, neighbor_matrix, q_id))

    df['q2_dot'] = df['q2_indot'] + df['q2_outdot']

    columns = ['q1_indot', 'q1_outdot', 'q1_dot', 'q2_indot', 'q2_outdot', 'q2_dot']

    return np.asarray(df[columns], dtype=np.int32)
graph_feature(df_train)