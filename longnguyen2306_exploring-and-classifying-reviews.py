import numpy as np

import pandas as pd

import warnings

import pandarallel

import nltk



pandarallel.pandarallel.initialize()



warnings.filterwarnings('ignore')



df = pd.read_csv('study/studycheck.csv')



del df["url"]



df.head()
df.info()
len(df)
import matplotlib.pyplot as plt

from collections import Counter



df_positive = df[df["weiter_empfehlung"] == True]

df_negative = df[df["weiter_empfehlung"] == False]



counter = Counter(df['weiter_empfehlung'])

plt.bar(counter.keys(), counter.values())

plt.show()
positive_corpus = df_positive["inhalt"].str.cat(sep = '\n')

positive_corpus[0:1000]
len(positive_corpus)
negative_corpus = df_negative["inhalt"].str.cat(sep = '\n')

negative_corpus[0:1000]
plt.bar([0, 1], [len(negative_corpus), len(positive_corpus)])

plt.show()
positive_text = nltk.Text(positive_corpus)

negative_text = nltk.Text(negative_corpus)

positive_text, negative_text
def lexical_diversity(text):

    return len(text) / len(set(text))



negative_diversity = lexical_diversity(negative_text)

positive_diversity = lexical_diversity(positive_text)



plt.bar([0, 1], [negative_diversity, positive_diversity])
from nltk.corpus import stopwords



stop_words = set(stopwords.words('german'))
from nltk.probability import FreqDist
negative_tokens = nltk.word_tokenize(negative_corpus)

negative_tokens = [x for x in negative_tokens if x not in stop_words]

negative_tokens = [x for x in negative_tokens if len(x) > 5]

negative_freq = FreqDist(negative_tokens)

negative_freq
positive_tokens = nltk.word_tokenize(positive_corpus)

positive_tokens = [x for x in positive_tokens if x not in stop_words]

positive_tokens = [x for x in positive_tokens if len(x) > 5]

positive_freq = FreqDist(positive_tokens)



positive_freq
plt.figure(figsize=(20, 12))

negative_freq.plot(25)

plt.show()
plt.figure(figsize=(20, 12))

positive_freq.plot(25)

plt.show()
!pip install joblib
from nltk import word_tokenize



negative_tokens = word_tokenize(negative_corpus)
from joblib import Parallel, delayed



negative_tokens = Parallel(n_jobs=24)(delayed(word_tokenize)(line) for line in negative_corpus.split("\n"))
import itertools



negative_tokens = list(itertools.chain(*negative_tokens))



negative_tokens = [x for x in negative_tokens if x not in stop_words]

negative_tokens = [x for x in negative_tokens if len(x) > 5]



negative_tokens[:10]
plt.figure(figsize=(20, 12))

negative_freq = FreqDist(negative_tokens)

negative_freq.plot(50)

plt.show()
positive_tokens = word_tokenize(positive_corpus)
positive_tokens = Parallel(n_jobs=24)(delayed(word_tokenize)(line) for line in positive_corpus.split("\n"))
positive_tokens = list(itertools.chain(*positive_tokens))





positive_tokens = [x for x in positive_tokens if x not in stop_words]

positive_tokens = [x for x in positive_tokens if len(x) > 5]



positive_tokens[:10]
plt.figure(figsize=(20, 12))

positive_freq = FreqDist(positive_tokens)

positive_freq.plot(50)

plt.show()
from nltk.stem import SnowballStemmer



stemmer = SnowballStemmer("german")



positive_tokens = [stemmer.stem(x) for x in positive_tokens]



plt.figure(figsize=(20, 12))



positive_freq = FreqDist(positive_tokens)



positive_freq.plot(50)



plt.show()
negative_tokens = [stemmer.stem(x) for x in negative_tokens]



plt.figure(figsize=(20, 12))



negative_freq = FreqDist(negative_tokens)



negative_freq.plot(50)



plt.show()
def count_words(text):

    return len(text.split(" "))



df["word_count"] = df["inhalt"].parallel_apply(count_words)

df.head()
import seaborn as sns

import numpy as np



plt.figure(figsize=(20, 10))

sns.distplot(df['word_count'])

plt.title(f"Range from {np.min(df['word_count'])} to {np.max(df['word_count'])}")

plt.plot()
df_positive = df[df["weiter_empfehlung"] == True]



plt.figure(figsize=(20, 10))

sns.distplot(df_positive['word_count'])

plt.title(f"Range from {np.min(df_positive['word_count'])} to {np.max(df_positive['word_count'])}")

plt.plot()
df_negative = df[df["weiter_empfehlung"] == False]



plt.figure(figsize=(20, 10))

sns.distplot(df_negative['word_count'])

plt.title(f"Range from {np.min(df_negative['word_count'])} to {np.max(df_negative['word_count'])}")

plt.plot()
df_without_outlier = df[df['word_count'] < 301]

plt.figure(figsize=(20, 10))

sns.distplot(df_without_outlier['word_count'])

plt.title(f"Range from {np.min(df_without_outlier['word_count'])} to {np.max(df_without_outlier['word_count'])}")

plt.plot()
df_without_outlier_positive = df_without_outlier[df_without_outlier["weiter_empfehlung"] == True]

plt.figure(figsize=(20, 10))

sns.distplot(df_without_outlier_positive['word_count'])



min_range = np.min(df_without_outlier_positive['word_count'])

max_range = np.max(df_without_outlier_positive['word_count'])

standard_deviation = np.std(df_without_outlier_positive['word_count'])



plt.title(f"Range from {min_range} to {max_range} with a standard deviation of {standard_deviation}")

plt.plot()
df_without_outlier_negative = df_without_outlier[df_without_outlier["weiter_empfehlung"] == False]

plt.figure(figsize=(20, 10))

sns.distplot(df_without_outlier_negative['word_count'])



min_range = np.min(df_without_outlier_negative['word_count'])

max_range = np.max(df_without_outlier_negative['word_count'])

standard_deviation = np.std(df_without_outlier_negative['word_count'])



plt.title(f"Range from {min_range} to {max_range} with a standard deviation of {standard_deviation}")

plt.plot()
def count_ich(text: str):

    ret = 0

    stems = ["ich", "mir", "mich"]

    for stem in stems:

        appearances = str(text).lower().split(" ").count(stem)

        ret += appearances

    return ret



df["ich_appearance"] = df["inhalt"].parallel_apply(count_ich)

df.head()
plt.figure(figsize=(20, 12))

sns.distplot(df["ich_appearance"])

plt.show()
df["word_count"].corr(df["ich_appearance"], method="pearson")
df["word_count"].corr(df["ich_appearance"], method="kendall")
df["word_count"].corr(df["ich_appearance"], method="spearman")
threshold = 30

df_many_ich= df[df["ich_appearance"] > threshold]

print(f"There are about {len(df_many_ich)} entries with {threshold} or more appearances of ich")

df_many_ich.head()
df["ich_appearance"].quantile(0.99)
threshold = 9

df_many_ich= df[df["ich_appearance"] <= threshold]

sns.distplot(df_many_ich["ich_appearance"])

plt.show()
df_many_ich_positive = df_many_ich[df_many_ich["weiter_empfehlung"] == True]

sns.distplot(df_many_ich["ich_appearance"])

plt.show()
df_many_ich_positive = df_many_ich[df_many_ich["weiter_empfehlung"] == False]

sns.distplot(df_many_ich["ich_appearance"])

plt.show()
data = df["inhalt"]

label = np.array(df["weiter_empfehlung"]).astype(np.int32)
from nltk.corpus import stopwords



stop_words = set(stopwords.words('german'))

def remove_stopwords(text):

    tokens = text.split(" ")

    tokens = [x.lower() for x in tokens if x not in stop_words]

    return " ".join(tokens)

data = data.parallel_apply(remove_stopwords)
from sklearn.model_selection import train_test_split



X_train, x_test, Y_train, y_test = train_test_split(data, label)



X_train.shape, x_test.shape, Y_train.shape, y_test.shape
from sklearn.feature_extraction.text import TfidfVectorizer



tfv = TfidfVectorizer() 



X_train = tfv.fit_transform(X_train)



x_test = tfv.transform(x_test)



X_train.shape, x_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report



lr = LogisticRegression()



lr.fit(X_train, Y_train)



print(classification_report(lr.predict(x_test), y_test))
from sklearn.naive_bayes import MultinomialNB



nb = MultinomialNB()



nb.fit(X_train, Y_train)



print(classification_report(nb.predict(x_test), y_test))
df_positive.head()
df_negative.head()
df_positive_sampled = df_positive.sample(n=len(df_negative), random_state=42)

print(len(df_positive_sampled))

df_positive_sampled.head()
balanced_df = pd.concat([df_negative, df_positive_sampled])

print(len(balanced_df))

balanced_df.head()
data = balanced_df["inhalt"]

target = balanced_df["weiter_empfehlung"]
from sklearn.model_selection import train_test_split



X_train, x_test, Y_train, y_test = train_test_split(data, target, test_size=0.15)



X_train.shape, x_test.shape, Y_train.shape, y_test.shape
from sklearn.pipeline import Pipeline

from sklearn.base import TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer



class DataTransformer(TransformerMixin):

    def __init__(self):

        self.stop_words = set(stopwords.words('german'))

        

    def remove_stopwords(self, text):

        tokens = text.split(" ")

        tokens = [x.lower() for x in tokens if x not in stop_words]

        return " ".join(tokens)

    

    def fit(self, X,  y=None, **kwargs):

        return self

        

    def transform(self, X,  y=None, **kwargs):

        return X.parallel_apply(self.remove_stopwords)



class TfIdfTransformer(TransformerMixin):

    def __init__(self):

        self.tfidf = TfidfVectorizer()

    

    def fit(self, X, y=None, **kwargs):

        self.tfidf.fit(X)

        return self

    

    def transform(self, X,  y=None, **kwargs):

        return self.tfidf.transform(X)

        



class TargetTransformer(TransformerMixin):

    def fit(self, X, y=None, **kwargs):

        return self

    

    def transform(self, X, y=None, **kwargs):

        return np.array(y).astype(np.int32)

    

data_transformer = DataTransformer()

tfidf_transformer = TfIdfTransformer()

target_transformer = TargetTransformer()



X_train = data_transformer.fit_transform(X_train)

x_test = data_transformer.fit_transform(x_test)

X_train = tfidf_transformer.fit_transform(X_train)

x_test = tfidf_transformer.transform(x_test)

Y_train = target_transformer.transform(X=X_train, y=Y_train)

y_test = target_transformer.transform(X=x_test, y=y_test)



X_train.shape, Y_train.shape, x_test.shape, y_test.shape
lr = LogisticRegression()



lr.fit(X_train, Y_train)



print(classification_report(lr.predict(x_test), y_test))
nb = MultinomialNB()



nb.fit(X_train, Y_train)



print(classification_report(nb.predict(x_test), y_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()



rf.fit(X_train, Y_train)



print(classification_report(rf.predict(x_test), y_test))
from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier()



gb.fit(X_train, Y_train)



print(classification_report(gb.predict(x_test), y_test))
from sklearn.ensemble import AdaBoostClassifier



ab = AdaBoostClassifier()



ab.fit(X_train, Y_train)



print(classification_report(ab.predict(x_test), y_test))
from sklearn.ensemble import BaggingClassifier



bc = BaggingClassifier()



bc.fit(X_train, Y_train)



print(classification_report(bc.predict(x_test), y_test))
from sklearn.ensemble import ExtraTreesClassifier



et = ExtraTreesClassifier()



et.fit(X_train, Y_train)



print(classification_report(et.predict(x_test), y_test))
from sklearn.ensemble import VotingClassifier



vc = VotingClassifier(estimators=[

    ("lr", lr),

    ("rf", rf),

    ("nb", nb),

    ("gb", gb),

    ("ab", ab),

])



vc.fit(X_train, Y_train)



print(classification_report(vc.predict(x_test), y_test))
from nltk.corpus import stopwords

import pandas as pd



df = pd.read_csv('study/studycheck.csv') 



stop_words = set(stopwords.words('german'))

def remove_stopwords(text):

    tokens = text.split(" ")

    tokens = [x.lower().replace(",", "").replace(".", "").replace(":", "") for x in tokens if x not in stop_words]

    return " ".join(tokens)



df_positive = df[df["weiter_empfehlung"] == True]



df_negative = df[df["weiter_empfehlung"] == False]



df_positive_sampled = df_positive.sample(n=len(df_negative), random_state=42)



balanced_df = pd.concat([df_negative, df_positive_sampled])



balanced_df["inhalt"] = balanced_df["inhalt"].parallel_apply(remove_stopwords)



balanced_df.head()
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(balanced_df, 

                                     test_size=0.15,

                                     random_state=42)



len(df_train), len(df_test)
documents = [review.split() for review in df_train.inhalt] 

documents[:1]
import gensim

from gensim.models.word2vec import Word2Vec





W2V_SIZE = 300

W2V_WINDOW = 10

W2V_MIN_COUNT = 5

SEQUENCE_LENGTH = 300

w2v_model = Word2Vec(size=W2V_SIZE, 

                     window=W2V_WINDOW, 

                     min_count=W2V_MIN_COUNT, 

                     workers=24)



w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()

len(words)
W2V_EPOCH = 20



w2v_model.train(documents, 

                total_examples=len(documents), 

                epochs=W2V_EPOCH)
w2v_model.most_similar("schlecht")
from tensorflow.keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer()

tokenizer.fit_on_texts(df_train.inhalt)



vocab_size = len(tokenizer.word_index)



print("Total words", vocab_size)
from itertools import islice



def take(n, iterable):

    "Return first n items of the iterable as a list"

    return list(islice(iterable, n))



for k, v in take(10, tokenizer.word_index.items()):

    print(k, v)
from tensorflow.keras.preprocessing.sequence import pad_sequences



x_train = pad_sequences(

    tokenizer.texts_to_sequences(df_train.inhalt), 

    maxlen=SEQUENCE_LENGTH)



x_test = pad_sequences(

    tokenizer.texts_to_sequences(df_test.inhalt), 

    maxlen=SEQUENCE_LENGTH)



x_train[0]
tokenizer.sequences_to_texts(x_train)[0]
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

encoder.fit(df_train.weiter_empfehlung.tolist())



y_train = encoder.transform(df_train.weiter_empfehlung.tolist())

y_test = encoder.transform(df_test.weiter_empfehlung.tolist())



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("y_train",y_train.shape)

print("y_test",y_test.shape)
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))

for word, i in tokenizer.word_index.items():

    if word in w2v_model.wv:

        embedding_matrix[i] = w2v_model.wv[word]

embedding_matrix
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM



embedding_layer = Embedding(vocab_size, 

                            W2V_SIZE, 

                            weights=[embedding_matrix], 

                            input_length=SEQUENCE_LENGTH, 

                            trainable=False)



model = Sequential()

model.add(embedding_layer)

model.add(Dropout(0.5))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])



model.summary()
EPOCHS = 5

BATCH_SIZE = 64



history = model.fit(x_train, y_train,

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS)
import pickle



model.save('study/model.h5')

w2v_model.save('study/w2v_model.w2v')

pickle.dump(tokenizer, open("study/tokenizer.pkl", "wb"), protocol=0)

pickle.dump(encoder, open("study/encoder.pkl", "wb"), protocol=0)
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print("ACCURACY:",score[1])

print("LOSS:",score[0])
def predict(text, ):

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)

    score = model.predict([x_test])[0]

    return score



predict("Informatik ist schon ganz gut.")
!shutdown now