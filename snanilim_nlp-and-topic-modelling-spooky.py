import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import nltk

import re

import string



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/spooky-author-identification/train.zip")

# train_data.head()
train_data.head()
train_data = train_data.drop('id', axis='columns')

train_data.head()
def clean_train_data(x):

    text = x

    text = text.lower()

    text = re.sub('\[.*?\]', '', text) # remove square brackets

    text = re.sub(r'[^\w\s]','',text) # remove punctuation

    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers

    text = re.sub('\n', '', text)

    return text
clean_data = train_data.copy()
clean_data['text'] = train_data.text.apply(lambda x : clean_train_data(x))

clean_data.head()
eng_stopwords = nltk.corpus.stopwords.words("english")
def remove_eng_stopwords(text):

    token_text = nltk.word_tokenize(text)

    remove_stop = [word for word in token_text if word not in eng_stopwords]

    join_text = ' '.join(remove_stop)

    return join_text
remove_stop_data = clean_data.copy()
remove_stop_data['text'] = clean_data.text.apply(lambda x : remove_eng_stopwords(x))

remove_stop_data.head()
print("Before remove stopwords", len(clean_data['text'][0]))

print("After remove stopwords", len(remove_stop_data['text'][0]))
from itertools import chain

from collections import Counter
list_words = remove_stop_data['text'].str.split()

list_words_merge = list(chain(*list_words))



d = Counter(list_words_merge)

df = pd.DataFrame(data=d, index=['count'])

top_common_words = df.T.sort_values(by=['count'], ascending=False).reset_index().head(50)

top_common_words.head()
plt.figure(figsize=(15,7))

sns.set(style="darkgrid")

sns.barplot(x="index", y='count', data=top_common_words)

plt.xticks(rotation=90)
common_words_value = top_common_words['index'].values

remove_words = ['man', 'life', 'night', 'house', 'heart']

new_stop_words = [x for x in common_words_value if x not in remove_words]

new_stop_words
def remove_new_stopwords(text):

    token_text = nltk.word_tokenize(text)

    remove_stop = [word for word in token_text if word not in new_stop_words]

    join_text = ' '.join(remove_stop)

    return join_text
new_stop_data = remove_stop_data.copy()
new_stop_data['text'] = remove_stop_data.text.apply(lambda x : remove_new_stopwords(x))

new_stop_data.head()
print("Before remove stopwords", len(remove_stop_data['text'][4]))

print("After remove stopwords", len(new_stop_data['text'][4]))
plt.figure(figsize=(10,7))

sns.set(style="darkgrid")

sns.countplot(x="author", data=train_data)

plt.title('Author text distribution')
all_words_after = train_data['text'].str.split()

merged = list(chain(*all_words_after))

d = Counter(merged)

df = pd.DataFrame(data=d, index=['count'])

top_count_words = df.T.sort_values(by=['count'], ascending=False).reset_index().head(50)

top_count_words.head()
plt.figure(figsize=(15,7))

sns.set(style="darkgrid")

sns.barplot(x="index", y='count', data=top_count_words)

plt.xticks(rotation=90)



plt.title("Most common words before removing stop words")
all_words_before = new_stop_data['text'].str.split()

merged = list(chain(*all_words_before))

d = Counter(merged)

df = pd.DataFrame(data=d, index=['count'])

before_top_words = df.T.sort_values(by=['count'], ascending=False).reset_index().head(50)

before_top_words.head()
plt.figure(figsize=(15,7))

sns.set(style="darkgrid")

sns.barplot(x="index", y='count', data=before_top_words)

plt.xticks(rotation=90)



plt.title("Most common words after removing stop words")
eap_cloud = train_data[train_data.author == 'EAP'].text.values

hpl_cloud = train_data[train_data.author == 'HPL'].text.values

mws_cloud = train_data[train_data.author == 'MWS'].text.values
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

wc = WordCloud(stopwords=STOPWORDS, background_color="white", colormap="Dark2",

               max_font_size=150, random_state=42)

plt.figure(figsize=(20, 15))



plt.subplot(1, 3, 1)

asd = " ".join(eap_cloud)

wc.generate(asd)

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.title('Edgar Allen Poe')



plt.subplot(1, 3, 2)

asd = " ".join(hpl_cloud)

wc.generate(asd)

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.title('HP Lovecraft')



plt.subplot(1, 3, 3)

asd = " ".join(mws_cloud)

wc.generate(asd)

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.title('Mary Shelley')



plt.figtext(.5,.63,'All writers, word clouds before stop word removal', color='b', fontsize=25, ha='center')
eap_cloud_before = new_stop_data[new_stop_data.author == 'EAP'].text.values

hpl_cloud_before = new_stop_data[new_stop_data.author == 'HPL'].text.values

mws_cloud_before = new_stop_data[new_stop_data.author == 'MWS'].text.values
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

wc = WordCloud(stopwords=STOPWORDS, background_color="white", colormap="Dark2",

               max_font_size=150, random_state=42)
plt.figure(figsize=(20, 15))



plt.subplot(1, 3, 1)

asd = " ".join(eap_cloud_before)

wc.generate(asd)

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.title('Edgar Allen Poe')



plt.subplot(1, 3, 2)

asd = " ".join(hpl_cloud_before)

wc.generate(asd)

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.title('HP Lovecraft')



plt.subplot(1, 3, 3)

asd = " ".join(mws_cloud_before)

wc.generate(asd)

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.title('Mary Shelley')



plt.figtext(.5,.63,'All writers, word clouds After stop word removal', color='b', fontsize=25, ha='center')
full_name = {'EAP': 'Edgar Allen Poe', 'MWS': 'Mary Shelley', 'HPL': 'HP Lovecraft'}

writer_name = ['EAP', 'MWS', 'HPL']

writer_count_obj = {'writer_full_name': [], 'total_words': [], 'unique_words': []}

for name in writer_name:

    name_all_words = new_stop_data[new_stop_data.author == name].text.str.split()

    name_merged = list(chain(*name_all_words))

    name_total_len = len(name_merged)

    myset = set(name_merged)

    

    writer_count_obj['writer_full_name'].append(full_name[name])

    writer_count_obj['total_words'].append(name_total_len)

    writer_count_obj['unique_words'].append(len(myset))
words_df = pd.DataFrame(writer_count_obj)

words_df
tidy = words_df.melt(id_vars='writer_full_name').rename(columns=str.title)

tidy
fig, ax1 = plt.subplots(figsize=(15, 10))

tidy = words_df.melt(id_vars='writer_full_name').rename(columns=str.title)

sns.barplot(x='Writer_Full_Name', y='Value', hue='Variable', data=tidy, ax=ax1)

sns.despine(fig)
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
lemm = WordNetLemmatizer()
def word_lemmatizer(text):

    token_text = nltk.word_tokenize(text)

    remove_stop = [lemm.lemmatize(w) for w in token_text]

    join_text = ' '.join(remove_stop)

    return join_text
lemmatize_data = new_stop_data.copy()

lemmatize_data['text'] = new_stop_data.text.apply(lambda x : word_lemmatizer(x))

lemmatize_data.head()
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(stop_words='english')

data_count_vec = count_vec.fit_transform(lemmatize_data.text)

data_count_vec
data_count_df = pd.DataFrame(data_count_vec.toarray(), columns=count_vec.get_feature_names())

data_count_df.index = lemmatize_data.author

data_count_df
from sklearn.decomposition import NMF, LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method = 'online', learning_offset = 50.,random_state = 0)
lda_model.fit(data_count_vec)
lda_model.components_
print_words = 20

get_feature_names = count_vec.get_feature_names()

for index, topic in enumerate(lda_model.components_):

    words = " ".join([get_feature_names[i] for i in topic.argsort()[:-print_words - 1 :-1]])

    print(f"Topic - {index}:")

    print(words)

    print("-"*100)

    print('\n')
from gensim import matutils, models

import scipy.sparse
data_count_df.index.name = None

new_dtm_t_data = data_count_df.T
spare_counts = scipy.sparse.csr_matrix(new_dtm_t_data)

new_corpus = matutils.Sparse2Corpus(spare_counts)

new_corpus
cv = count_vec

id2word = dict((v, k) for k, v in cv.vocabulary_.items())

# id2word
gensim_lda_topic = models.LdaModel(corpus=new_corpus, id2word=id2word, num_topics=5, passes=10)

gensim_lda_topic.print_topics()