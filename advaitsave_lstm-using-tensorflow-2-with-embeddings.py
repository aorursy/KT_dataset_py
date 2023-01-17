!pip install tensorflow-gpu==2.0.0-alpha0
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import operator

from collections import Counter

from wordcloud import WordCloud, STOPWORDS



from tqdm import tqdm

tqdm.pandas()



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer



import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D

from tensorflow.keras.models import Model, Sequential

from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from tensorflow.keras import layers



%matplotlib inline

import matplotlib.pyplot as plt



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 1000)
print(tf.__version__)
x = tf.random.uniform([3, 3])



print("Is there a GPU available: "),

print(tf.test.is_gpu_available())



print("Is the Tensor on GPU #0:  "),

print(x.device.endswith('GPU:0'))



print("Device name: {}".format((x.device)))
print(tf.executing_eagerly())
print(tf.keras.__version__)
df  = pd.read_csv("../input/train.csv")

df_test = pd.read_csv('../input/test.csv')

df[df.target==1].head(10)
print("Number of questions: ", df.shape[0])
df.target.value_counts()
print("Percentage of insincere questions: {}".format(sum(df.target == 1)*100/len(df.target)))
# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

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

plot_wordcloud(df[df.target == 0]["question_text"], title="Word Cloud of Sincere Questions")
plot_wordcloud(df[df.target == 1]["question_text"], title="Word Cloud of Insincere Questions")
stopwords = set(STOPWORDS)
sincere_words = df[df.target==0].question_text.apply(lambda x: x.lower().split()).tolist()

insincere_words = df[df.target==1].question_text.apply(lambda x: x.lower().split()).tolist()



sincere_words = [item for sublist in sincere_words for item in sublist if item not in stopwords]

insincere_words = [item for sublist in insincere_words  for item in sublist if item not in stopwords ]
print('Number of sincere words',len(sincere_words))

print('Number of insincere words',len(insincere_words))
sincere_words_counter = Counter(sincere_words)

insincere_words_counter = Counter(insincere_words)
most_common_sincere_words = sincere_words_counter.most_common()[:10]

most_common_sincere_words = pd.DataFrame(most_common_sincere_words)

most_common_sincere_words.columns = ['word', 'freq']

most_common_sincere_words['percentage'] = most_common_sincere_words.freq *100 / sum(most_common_sincere_words.freq)

most_common_sincere_words
most_common_insincere_words = insincere_words_counter.most_common()[:10]

most_common_insincere_words = pd.DataFrame(most_common_insincere_words)

most_common_insincere_words.columns = ['word', 'freq']

most_common_insincere_words['percentage'] = most_common_insincere_words.freq *100 / sum(most_common_insincere_words.freq)

most_common_insincere_words
def generate_ngrams(words, n):

    

    # Use the zip function to help us generate n-grams

    # Concatentate the tokens into ngrams and return

    ngrams = zip(*[words[i:] for i in range(n)])

    return [" ".join(ngram) for ngram in ngrams]
n = 3
sincere_ngram_counter = Counter(generate_ngrams(sincere_words, n))

insincere_ngram_counter = Counter(generate_ngrams(insincere_words, n))
most_common_sincere_ngram = sincere_ngram_counter.most_common()[:10]

most_common_sincere_ngram = pd.DataFrame(most_common_sincere_ngram)

most_common_sincere_ngram.columns = ['word', 'freq']

most_common_sincere_ngram['percentage'] = most_common_sincere_ngram.freq *100 / sum(most_common_sincere_ngram.freq)

most_common_sincere_ngram
most_common_insincere_ngram = insincere_ngram_counter.most_common()[:10]

most_common_insincere_ngram = pd.DataFrame(most_common_insincere_ngram)

most_common_insincere_ngram.columns = ['word', 'freq']

most_common_insincere_ngram['percentage'] = most_common_insincere_ngram.freq *100 / sum(most_common_insincere_ngram.freq)

most_common_insincere_ngram
# config values

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use
X_train, X_test  = train_test_split(df, test_size=0.1, random_state=2019)

y_train, y_test = X_train['target'].values, X_test['target'].values
X_train = X_train['question_text'].fillna('_NA_').values

X_test = X_test['question_text'].fillna('_NA_').values

X_submission = df_test['question_text'].fillna('_NA_').values
X_train.shape
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

X_submission = tokenizer.texts_to_sequences(X_submission)
X_train = pad_sequences(X_train, maxlen=maxlen)

X_test = pad_sequences(X_test, maxlen=maxlen)

X_submission = pad_sequences(X_submission, maxlen=maxlen)
def data_prep(df):

    print("Splitting dataframe with shape {} into training and test datasets".format(df.shape))

    X_train, X_test  = train_test_split(df, test_size=0.1, random_state=2019)

    y_train, y_test = X_train['target'].values, X_test['target'].values

    

    print("Filling missing values")

    X_train = X_train['question_text'].fillna('_NA_').values

    X_test = X_test['question_text'].fillna('_NA_').values

    X_submission = df_test['question_text'].fillna('_NA_').values

    

    print("Tokenizing {} questions into words".format(df.shape[0]))

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(X_train))

    X_train = tokenizer.texts_to_sequences(X_train)

    X_test = tokenizer.texts_to_sequences(X_test)

    X_submission = tokenizer.texts_to_sequences(X_submission)

    

    print("Padding sequences for uniform dimensions")

    X_train = pad_sequences(X_train, maxlen=maxlen)

    X_test = pad_sequences(X_test, maxlen=maxlen)

    X_submission = pad_sequences(X_submission, maxlen=maxlen)

    

    print("Completed data preparation, returning training, test and submission datasets, split as dependent(X) and independent(Y) variables")

    

    return X_train, X_test, y_train, y_test, X_submission
model1 = Sequential()

model1.add(Embedding(max_features, embed_size, input_length=maxlen))

model1.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))

model1.add(GlobalMaxPool1D())

model1.add(Dropout(0.2))

model1.add(Dense(64, activation='relu'))

model1.add(Dropout(0.2))

model1.add(Dense(32, activation='relu'))

model1.add(Dropout(0.2))

model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model1.summary()
%time model1.fit(X_train, y_train, batch_size=512, epochs=2, validation_data=(X_test, y_test), verbose = 1)
pred_test_y = model1.predict([X_test], batch_size=1024, verbose=1)

opt_prob = None

f1_max = 0



for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(y_test, (pred_test_y > thresh).astype(int))

    print('F1 score at threshold {} is {}'.format(thresh, f1))

    

    if f1 > f1_max:

        f1_max = f1

        opt_prob = thresh

        

print('Optimal probabilty threshold is {} for maximum F1 score {}'.format(opt_prob, f1_max))
pred_submission_y = model1.predict([X_submission], batch_size=1024, verbose=1)

pred_submission_y = (pred_submission_y > opt_prob).astype(int)



df_submission = pd.DataFrame({'qid': df_test['qid'].values})

df_submission['prediction'] = pred_submission_y

#df_submission.to_csv("submission.csv", index=False)
def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding="utf8") if len(o)>100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
print("Extracting GloVe embedding")

embed_glove = load_embed(glove)

#print("Extracting Paragram embedding")

#embed_paragram = load_embed(paragram)

#print("Extracting FastText embedding")

#embed_fasttext = load_embed(wiki_news)
def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
vocab = build_vocab(df['question_text'])
print("Glove : ")

oov_glove = check_coverage(vocab, embed_glove)

#print("Paragram : ")

#oov_paragram = check_coverage(vocab, embed_paragram)

#print("FastText : ")

#oov_fasttext = check_coverage(vocab, embed_fasttext)
type(embed_glove)
dict(list(embed_glove.items())[20:22])
df['processed_question'] = df['question_text'].apply(lambda x: x.lower())
vocab_low = build_vocab(df['processed_question'])
print("Glove : ")

oov_glove = check_coverage(vocab_low, embed_glove)

#print("Paragram : ")

#oov_paragram = check_coverage(vocab_low, embed_paragram)

#print("FastText : ")

#oov_fasttext = check_coverage(vocab_low, embed_fasttext)
oov_glove[1:20]
def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")
print("Glove : ")

add_lower(embed_glove, vocab)

#print("Paragram : ")

#add_lower(embed_paragram, vocab)

#print("FastText : ")

#add_lower(embed_fasttext, vocab)
print("Glove : ")

oov_glove = check_coverage(vocab_low, embed_glove)

#print("Paragram : ")

#oov_paragram = check_coverage(vocab_low, embed_paragram)

#print("FastText : ")

#oov_fasttext = check_coverage(vocab_low, embed_fasttext)
oov_glove[1:20]
punctuations = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in punctuations:

        x = x.replace(punct, '')

    return x
df["processed_question"] = df["processed_question"].progress_apply(lambda x: clean_text(x))
vocab_low = build_vocab(df['processed_question'])
print("Glove : ")

oov_glove = check_coverage(vocab_low, embed_glove)

#print("Paragram : ")

#oov_paragram = check_coverage(vocab_low, embed_paragram)

#print("FastText : ")

#oov_fasttext = check_coverage(vocab_low, embed_fasttext)
df['question_text'] = df['processed_question']
X_train, X_test, y_train, y_test, X_submission = data_prep(df)
model1 = Sequential()

model1.add(Embedding(max_features, embed_size, input_length=maxlen, weights = [embed_glove]))

model1.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))

model1.add(GlobalMaxPool1D())

model1.add(Dropout(0.2))

model1.add(Dense(64, activation='relu'))

model1.add(Dropout(0.2))

model1.add(Dense(32, activation='relu'))

model1.add(Dropout(0.2))

model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model1.summary()
%time model1.fit(X_train, y_train, batch_size=512, epochs=5, validation_data=(X_test, y_test), verbose = 1)
pred_test_y = model1.predict([X_test], batch_size=1024, verbose=1)

opt_prob = None

f1_max = 0



for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(y_test, (pred_test_y > thresh).astype(int))

    print('F1 score at threshold {} is {}'.format(thresh, f1))

    

    if f1 > f1_max:

        f1_max = f1

        opt_prob = thresh

        

print('Optimal probabilty threshold is {} for maximum F1 score {}'.format(opt_prob, f1_max))
pred_submission_y = model1.predict([X_submission], batch_size=1024, verbose=1)

pred_submission_y = (pred_submission_y > opt_prob).astype(int)



df_submission = pd.DataFrame({'qid': df_test['qid'].values})

df_submission['prediction'] = pred_submission_y

df_submission.to_csv("submission.csv", index=False)