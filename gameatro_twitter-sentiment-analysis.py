import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer



import nltk

from nltk.corpus import stopwords

from  nltk.stem import SnowballStemmer



import gensim
nltk.download('stopwords')
data = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv",encoding="ISO-8859-1")

data.head()
data.columns = ["target","ids","date","flag","user","text"]

data.head()
data.drop(["ids","date","flag","user"],axis=1,inplace=True)
target_dict = {0:"Negative",2:"Neutral",4:"Positive"}



def target_map(target):

    return target_dict[target]
data.target = data.target.apply(lambda x: target_map(x))



data.head()
neg_count = (data["target"] == "Negative").sum()

neu_count = (data["target"] == "Neutral").sum()

pos_count = (data["target"] == "Positive").sum()



plt.bar(["Negative","Neutral","Positive"],[neg_count,neu_count,pos_count])

plt.title("Distribution according to Target")
from wordcloud import WordCloud,STOPWORDS
stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")
spl_char = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



def pre_process(text):

    text = re.sub(spl_char, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            tokens.append(stemmer.stem(token))

    return " ".join(tokens)

        
data["text"] = data["text"].apply(lambda x: pre_process(x))

data.head()
data_neg = data[data["target"] == "Negative"]

plt.figure(figsize = (20,20))

wc = WordCloud(max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data_neg.text))

plt.imshow(wc , interpolation = 'bilinear')
data_pos = data[data["target"] == "Positive"]

plt.figure(figsize = (20,20))

wc = WordCloud(max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data_pos.text))

plt.imshow(wc , interpolation = 'bilinear')
df_train, df_test = train_test_split(data, test_size=0.2)
words = [txt.split() for txt in df_train.text]
w2v = gensim.models.word2vec.Word2Vec(size=300, window=7, min_count=10, workers=8)
w2v.build_vocab(words)
words1 = w2v.wv.vocab.keys()

words1
w2v.train(words, total_examples=len(words), epochs=32)
w2v.most_similar("love")
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df_train.text)

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=300)

x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=300)
labels = df_train.target.unique().tolist()

labels.append("Neutral")

labels
encoder = LabelEncoder()

encoder.fit(df_train.target.tolist())



y_train = encoder.transform(df_train.target.tolist())

y_test = encoder.transform(df_test.target.tolist())



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("y_train",y_train.shape)

print("y_test",y_test.shape)