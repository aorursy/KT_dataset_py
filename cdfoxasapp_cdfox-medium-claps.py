import pandas as pd

medium_data = pd.read_csv("/kaggle/input/medium-articles-dataset/medium_data.csv")

medium_data
from sklearn.model_selection import train_test_split



train_data, test_data = train_test_split(medium_data, test_size=0.10, random_state=42)

print(f'train_data: {len(train_data)} rows')

print(f'test_data: {len(test_data)} rows')
from collections import Counter

from nltk.corpus import stopwords

import string



import re



def cleanhtml(raw_html):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, '', raw_html)

    return cleantext



def remove_punct(s):

    return s.translate(str.maketrans('', '', string.punctuation))



def process_title(title):

    return remove_punct(cleanhtml(title))



punct = set(string.punctuation)



stopwords = set(stopwords.words('english'))

vocab = Counter(word.lower() 

                for title in train_data['title'] 

                for word in process_title(title).split() 

                if word.lower() not in stopwords and len(word) > 1)



vocab.most_common(40)
clap_counter = Counter()



for i, row in train_data.iterrows():

    claps = row['claps']

    title = process_title(row['title'])

    words = [word for word in title.lower().split() if word in vocab]

    for word in words:

        clap_counter[word] += claps

    

clap_counter.most_common(20)
vocab_stats = []

for word, _ in vocab.most_common(1000):

    vocab_stats.append((word, clap_counter[word] / vocab[word], vocab[word]))

vocab_stats.sort(key=lambda tup: tup[1], reverse=True)

vocab_stats[:20]
for i, row in train_data.iterrows():

    if '15' in row['title'].lower():

        print(row['publication'], '--', row['title'], '--', row['claps'])
average_claps = sum(train_data['claps']) / len(train_data)



def predict_claps(title):

    title = process_title(title)

    words = [word for word in title.lower().split() if word in vocab]

    average_claps_for_each_word = [clap_counter[word] / vocab[word] for word in words]

    if len(average_claps_for_each_word):

        prediction = sum(average_claps_for_each_word) / len(average_claps_for_each_word)

    else:

        prediction = average_claps

    return prediction
predict_claps('asdg')
average_claps
import math



errors = []

num_shown = 0

for i, row in test_data.iterrows():

    title = row['title']

    predicted_claps = predict_claps(title)

    actual_claps = row['claps']

    errors.append(abs(actual_claps - predicted_claps))

    if num_shown  < 5:

        num_shown += 1

        print(title, predicted_claps, actual_claps, abs(actual_claps - predicted_claps))

    

sum(errors) / len(errors)