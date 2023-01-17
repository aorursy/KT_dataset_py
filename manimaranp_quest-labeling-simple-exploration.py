import pandas as pd

from pathlib import Path

import pickle

from wordcloud import WordCloud as wc

import matplotlib.pyplot as plt
path = Path('/kaggle/input/google-quest-challenge')

list(path.iterdir())
data = pd.read_csv(path/'train.csv', index_col=[0])

data.head()
data.info()
data['category'].value_counts().plot.bar(title='Category')
data['host'].value_counts().plot.bar(title='Host')
text_cols = ['question_title', 'question_body', 'answer']

text_cols
tab_cols = ['category', 'host']

tab_cols
targets = data.columns[10:].tolist()

targets
needed_cols = set(text_cols + tab_cols + targets)

all_cols = set(data.columns)

unneeded_cols = all_cols - needed_cols

unneeded_cols
corr = data[targets].corr(method='spearman')
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
with open('target.pkl', 'wb') as f: pickle.dump(targets, f)

with open('tab_cols.pkl', 'wb') as f: pickle.dump(tab_cols, f)

with open('text_cols.pkl', 'wb') as f: pickle.dump(text_cols, f)
q_text = ' '.join((data['question_title'] + data['question_body']).tolist())
wcloud = wc().generate(q_text)
plt.figure()

plt.imshow(wcloud)

plt.axis("off")

plt.show()
a_text = ' '.join(data['answer'].tolist())
wcloud = wc().generate(a_text)
plt.figure()

plt.imshow(wcloud)

plt.axis("off")

plt.show()