!pip install nlpaug #numpy matplotlib python-dotenv
# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt
%matplotlib inline

# Scikit-learn
from sklearn.model_selection import train_test_split

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# DATASET
dataset_path = "../input/sentiment140/training.1600000.processed.noemoticon.csv"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.9
SAMPLING_FRAC = 0.01

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS).sample(frac=SAMPLING_FRAC, random_state=1)
print("Dataset size:", len(df))
df.head(5)
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]
%%time
df.target = df.target.apply(lambda x: decode_sentiment(x))
target_cnt = Counter(df.target)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")
nltk.download('stopwords')
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)
%%time
df.text = df.text.apply(lambda x: preprocess(x))
df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
df_train.name = "Original"
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
aug = nac.KeyboardAug() ##define th text augmentation method
df_keyboard = df_train.copy(deep=True)
df_keyboard.text= df_keyboard.text.apply(lambda _text: aug.augment(_text))
df_keyboard.name =  "random keyboard"

print("Original:")
print(df_train.text.iloc[1])
print("Augmented Text:")
print(df_keyboard.text.iloc[1])
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path="../input/sentiment140-w2v/sentiment140_w2v.bin",action="substitute")

df_Sub = df_train.copy(deep=True)
df_Sub.text = df_Sub.text.apply(lambda _text: aug.augment(_text))
df_Sub.name = "Synonym Replacement"
print("Original:")
print(df_train.text.iloc[1])
print("Augmented Text:")
print(df_Sub.text.iloc[1])
aug = naw.RandomWordAug(action="swap")
df_RS = df_train.copy(deep=True)
df_RS.text= df_RS.text.apply(lambda _text: aug.augment(_text))
df_RS.name = "Random Swap"
print("Original:")
print(df_train.text.iloc[1])
print("Augmented Text:")
print(df_RS.text.iloc[1])
aug = naw.RandomWordAug(action="delete")
df_RD = df_train.copy(deep=True)
df_RD.text= df_RD.text.apply(lambda _text: aug.augment(_text))
df_RD.name = "Random Deletion"
print("Original:")
print(df_train.text.iloc[1])
print("Augmented Text:")
print(df_RD.text.iloc[1])
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path="../input/sentiment140-w2v/sentiment140_w2v.bin",action="insert")

df_RI = df_train.copy(deep=True)
df_RI.text = df_RI.text.apply(lambda _text: aug.augment(_text))
df_RI.name = "Random Insertion"
print("Original:")
print(df_train.text.iloc[1])
print("Augmented Text:")
print(df_RI.text.iloc[1])
def word_feats(words):
    return dict([(word, True) for word in words.split() if word not in stop_words])

featureset_test = [(word_feats(row.text), row.target) for index, row in df_test.iterrows()]



for df_train in (df_train,df_keyboard, df_Sub, df_RS, df_RD, df_RI):
    featureset_train = [(word_feats(row.text), row.target) for index, row in df_train.iterrows()]
    classifier = nltk.NaiveBayesClassifier.train(featureset_train)
    print("Testset accuracy of",df_train.name,"is: ",nltk.classify.accuracy(classifier, featureset_test))

