# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

file_paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        file_paths.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
for path in file_paths:

    if 'train.csv' in path:

        print('train.csv')

        train_df = pd.read_csv(path)

    elif 'test.csv' in path:

        print('test.csv')

        test_df = pd.read_csv(path)
train_df.info()
print('Null Values in Training Data')

train_df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



ax = sns.countplot(x='target',  data=train_df)

plt.show()
keyword_cnt = train_df.keyword.value_counts()

keyword_cnt
df_train_fake = train_df[train_df['target'] == 1]

keyword_cnt_fake = df_train_fake.keyword.value_counts()

keyword_cnt_fake
train_df['target_mean'] = train_df.groupby('keyword')['target'].transform('mean')



fig = plt.figure(figsize=(8, 72), dpi=100)



sns.countplot(y=train_df.sort_values(by='target_mean', ascending=False)['keyword'],

              hue=train_df.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()



train_df.drop(columns=['target_mean'], inplace=True)
from wordcloud import WordCloud, STOPWORDS
# Credit : https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()
train_df0 = train_df[train_df['target']==0]

train_df1 = train_df[train_df['target']==1]

plot_wordcloud(train_df0['text'], title="Real Tweet Word Cloud")

plot_wordcloud(train_df1['text'], title="Fake Tweet Word Cloud")
# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert

import string

def generate_ngrams(text, n_gram=1):

    table = str.maketrans(dict.fromkeys(string.punctuation))

    token = [token.translate(table) for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]

#     print(token)

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [' '.join(ngram) for ngram in ngrams]
def plot_ngram(df_fake_ngram, df_real_ngram, N=100):

    fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)

    plt.tight_layout()



    sns.barplot(y=df_fake_ngram[0].values[:N], x=df_fake_ngram[1].values[:N], ax=axes[0], color='red')

    sns.barplot(y=df_real_ngram[0].values[:N], x=df_real_ngram[1].values[:N], ax=axes[1], color='green')



    for i in range(2):

        axes[i].spines['right'].set_visible(False)

        axes[i].set_xlabel('')

        axes[i].set_ylabel('')

        axes[i].tick_params(axis='x', labelsize=13)

        axes[i].tick_params(axis='y', labelsize=13)



    axes[0].set_title(f'Top {N} most common unigrams in Disaster Tweets', fontsize=15)

    axes[1].set_title(f'Top {N} most common unigrams in Non-disaster Tweets', fontsize=15)



    plt.show()
from collections import defaultdict

unigram_fake = defaultdict(int)

unigram_real = defaultdict(int)



for tweet in train_df0['text']:

#     print(tweet)

    for word in generate_ngrams(tweet):

#         print(word)

        unigram_real[word] += 1

        

for tweet in train_df1['text']:

    for word in generate_ngrams(tweet):

        unigram_fake[word] += 1



df_fake_unigrams = pd.DataFrame(sorted(unigram_fake.items(), key=lambda x: x[1])[::-1])

df_real_unigrams = pd.DataFrame(sorted(unigram_real.items(), key=lambda x: x[1])[::-1])

plot_ngram(df_fake_unigrams, df_real_unigrams)
bigram_fake = defaultdict(int)

bigram_real = defaultdict(int)



for tweet in train_df0['text']:

    for word in generate_ngrams(tweet, 2):

        bigram_real[word] += 1

        

for tweet in train_df1['text']:

    for word in generate_ngrams(tweet, 2):

        bigram_fake[word] += 1



df_fake_bigrams = pd.DataFrame(sorted(bigram_fake.items(), key=lambda x: x[1])[::-1])

df_real_bigrams = pd.DataFrame(sorted(bigram_real.items(), key=lambda x: x[1])[::-1])

plot_ngram(df_fake_bigrams, df_real_bigrams)
trigram_fake = defaultdict(int)

trigram_real = defaultdict(int)



for tweet in train_df0['text']:

    for word in generate_ngrams(tweet, 3):

        trigram_real[word] += 1

        

for tweet in train_df1['text']:

    for word in generate_ngrams(tweet, 3):

        trigram_fake[word] += 1



df_fake_trigrams = pd.DataFrame(sorted(trigram_fake.items(), key=lambda x: x[1])[::-1])

df_real_trigrams = pd.DataFrame(sorted(trigram_real.items(), key=lambda x: x[1])[::-1])

plot_ngram(df_fake_trigrams, df_real_trigrams)
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import bert_tokenization as tokenization
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train_df.text.values, tokenizer, max_len=160)

test_input = bert_encode(test_df.text.values, tokenizer, max_len=160)

train_labels = train_df.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split=0.25,

    epochs=3,

    callbacks=[checkpoint],

    batch_size=64

)
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

model.load_weights('model.h5')

test_pred = model.predict(test_input)

submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)
submission