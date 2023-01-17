!pip install paddlepaddle==1.7.1 paddlehub==1.6.1 pkuseg==0.0.22
import paddlehub as hub

import pandas as pd

import pkuseg

import itertools

import numpy as np

import matplotlib.pyplot as plt

import os

from matplotlib import cm

from wordcloud import WordCloud

from collections import Counter
luckin_weibo_data = pd.read_csv('../input/luckin-weibo/luckin_weibo.csv')

luckin_weibo_data[: 10]
luckin_weibo_texts = luckin_weibo_data[:]['content'].values.tolist()

print(luckin_weibo_texts[: 5])
with open('../input/hit-stopwords/hit_stopwords.txt', 'r') as f:

    stopwords = f.read().split()

print(len(stopwords))

print(stopwords[: 50])

print(stopwords[-50: ])
def generate_counter(texts, stopwords):

    seg = pkuseg.pkuseg()

    counter = Counter(itertools.chain(*[seg.cut(text) for text in texts]))

    for word in stopwords:

        if word in counter:

            counter.pop(word)

    return counter
def generate_word_cloud(counter, mask=None):

    wc = WordCloud(background_color='white',

                font_path='../input/simsun/simsun.ttf',

                max_words=400,

                width=1000,

                height=500,

                mask=mask)

    wc.generate_from_frequencies(counter)

    return wc
counter = generate_counter(luckin_weibo_texts, stopwords)

wc = generate_word_cloud(counter)

plt.figure(figsize=(10, 6))

plt.axis('off')

plt.imshow(wc);
senta = hub.Module(name="senta_bilstm")
sample_texts = [

    '那你可真是个弟弟',

    '真是太好吃了！！！',

    '哈哈哈哈哈哈哈'

]

sample_input_dict = {'text': sample_texts}

sample_predictions = senta.sentiment_classify(data=sample_input_dict)

sample_predictions
input_dict = {'text': luckin_weibo_texts}

predictions = senta.sentiment_classify(data=input_dict)
pos_texts = [r['text'] for r in predictions if r['sentiment_label'] == 1]

neg_texts = [r['text'] for r in predictions if r['sentiment_label'] == 0]
pos_counter = generate_counter(pos_texts, stopwords)

pos_wc = generate_word_cloud(pos_counter)

plt.figure(figsize=(10, 6))

plt.axis('off')

plt.imshow(pos_wc);
neg_counter = generate_counter(neg_texts, stopwords)

neg_wc = generate_word_cloud(neg_counter)

plt.figure(figsize=(10, 6))

plt.axis('off')

plt.imshow(neg_wc);
senti_score = [r['positive_probs'] for r in predictions]

plt.figure(figsize=(10, 6))

plt.xlabel('Sentiment Score')

plt.hist(senti_score, bins=20, edgecolor='white');
with open('../input/ziyemaodun/ziye.txt', 'r') as f:

    ziye_data = f.read()



print(ziye_data[: 300])
os.listdir('../input/qingchunyouni/qingchunyouni')
qcyn_weibo_data = pd.read_csv('../input/qingchunyouni/qingchunyouni/0329.csv')

qcyn_weibo_data[: 10]