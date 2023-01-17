# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gc



import matplotlib.pyplot as plt

import seaborn as sns



from nltk import ngrams

import re

from collections import defaultdict
df = pd.read_csv('../input/all-posts-public-main-chatroom/freecodecamp_casual_chatroom.csv',

                 usecols=['fromUser.id', 'text'])

print(df.head())

gc.collect()
# Renaming columns for my better understanding

df.rename(columns={'fromUser.id': 'id'}, inplace = True)
df.text = df.text.astype(str)

df.dtypes
id_count = df.id.value_counts().reset_index(drop=False).head(10)

#id_count.head()

plt.figure(figsize=(12, 8))

g = sns.barplot(x='index', y='id', data=id_count, color='green')

g.set_xticklabels(labels=id_count['index'], rotation=45)

plt.title('Top Active users')

plt.xlabel("User ID")

plt.ylabel("Count of Message")
df = df[df.id=='55b977f00fc9f982beab7883']

df.head()
def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', ' ', text)

    text = re.sub(r'@\S+', ' ', text)

    text = re.sub(r'#\S+', ' ', text)

    text = re.sub(r'[^a-z]', ' ', text)

    text = ' '.join(['' if len(word)<2 else word for word in text.split()])

    text = re.sub(r' +', ' ', text)

    text = text.strip()

    return text
df.text = df.text.map(clean_text)
df['word_count'] = df.text.apply(lambda text: len(text.split(' ')))
# Creating text corpus

text = df.text.str.cat(sep=' ')
print(len(text))

print(text[:300])
trigram = ngrams(text.split(), n=3)
model = defaultdict(lambda: defaultdict(lambda: 0))



for w1, w2, w3 in trigram:

    model[(w1, w2)][w3] += 1
for key in model:

    total_count_sum = sum(model[key].values())

    total_count_sum = float(total_count_sum)

    for index in model[key]:

        model[key][index] = model[key][index]/total_count_sum
start = ['sends', 'brownie']

stop = False

first = True

generated_text = start



while not stop:

    if first:

        # Covers only initial scenario.

        try:

            words = list(model[start[0], start[1]].keys())

            random = np.random.randint(len(words))

            word = words[random]

            # print(word)

        except Exception as e:

            print(str(e))

            break

        first = False

        generated_text.append(word)

    start = generated_text[-2:]

    if len(start)==0:

        stop=True

    try:

        words = list(model[start[0], start[1]].keys())

        random = np.random.randint(len(words))

        word = words[random]

    except Exception as e:

        print(str(e))

        break



    generated_text.append(word)



    if len(generated_text) > 100:

        break
final_text = ' '.join(generated_text)



print(final_text)
gc.collect()
start = ['welcome', 'to']



max(model[start[0], start[1]], key=model[start[0], start[1]].get)