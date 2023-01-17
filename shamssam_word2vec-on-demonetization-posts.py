import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


def preprocessing(
    text,
    remove_stopwords=False,
    stem_words=False,
    stopwords_addition=[],
    stopwords_exclude=[],
    HYPHEN_HANDLE=1
):
    """
    convert string text to lower and split into words.
    most punctuations are handled by replacing them with empty string.
    some punctuations are handled differently based on their occurences in the data.
    -  replaced with ' '
    few peculiar cases for better uniformity.
    'non*' replaced with 'non *'
    few acronyms identified
    SCID  Severe Combined ImmunoDeficiency
    ADA   Adenosine DeAminase
    PNP   Purine Nucleoside Phosphorylase
    LFA-1 Lymphocyte Function Antigen-1
    """
    text = text.lower().split()

    if remove_stopwords:
        stops = list(set(stopwords.words('english')) -
                     set(stopwords_exclude)) + stopwords_addition
        text = [w for w in text if w not in stops]

    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=_]", " ", text)
    text = re.sub(r"(that|what)(\'s)", r"\g<1> is", text)
    text = re.sub(r"i\.e\.", "that is", text)
    text = re.sub(r"(^non| non)", r"\g<1> ", text)
    text = re.sub(r"(^anti| anti)", r"\g<1> ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    if HYPHEN_HANDLE == 1:
        text = re.sub(r"\-", "-", text)
    elif HYPHEN_HANDLE == 2:
        text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.lower().split()

    if remove_stopwords:
        stops = list(set(stopwords.words('english')) -
                     set(stopwords_exclude)) + stopwords_addition
        text = [w for w in text if w not in stops]

    text = " ".join(text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)
from functools import partial
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
preprocessing = partial(preprocessing, HYPHEN_HANDLE=2)
def generate_data(corpus, _slice=3):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus = tokenizer.texts_to_sequences(corpus)
    
    data = []
    targets = []
    for sentence in corpus:
        slices = [sentence[i: i+_slice] for i in range(0, len(sentence) - (_slice-1))]
        center = int(np.floor(_slice/2))
        for s in slices:
            data.append([s[center]])
            targets.append([_ for idx, _ in enumerate(s) if idx != center])
    
    X = np.zeros((len(data), len(tokenizer.word_index)+1))
    y = np.zeros((len(data), len(tokenizer.word_index)+1))
    for idx, (i, j) in enumerate(zip(data, targets)):
        X[idx][i] = 1
        y[idx][j] = 1

    print("X_shape:", X.shape)
    print("y_shape:", y.shape)
    print("# Words:", len(tokenizer.word_index))

    return X, y, tokenizer
df_data = pd.read_csv('../input/demonetization-tweets.csv', encoding='latin-1', usecols=['text'])
df_data.drop_duplicates(inplace=True)
df_data.dropna(inplace=True)
df_data.text = df_data.text.apply(preprocessing)
corpus = [_ for sent in df_data.text.tolist() for _ in sent.split(".")]
X, y, tokenizer = generate_data(corpus, 5)
model = Sequential([
    Dense(2, input_shape=(X.shape[1],)),
    Dense(X.shape[1]),
    Activation('softmax')
])
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.summary()
try:
    model.fit(X, y, epochs=10, verbose=1)
except KeyboardInterrupt:
    print('\n\nExited by User')
points = model.layers[0].get_weights()[0]
word_embedding = {word: embedding for word, embedding in zip(tokenizer.word_index.keys(), points[1:])}
inverse_idx = {v: k for k, v in tokenizer.word_index.items()}
def closest(word, _top=5):
    word = word_embedding[word]
    cos_sim = cosine_similarity(word.reshape(1, -1), points)
    top_n = cos_sim.argsort()[0][-_top:][::-1]
    return [inverse_idx[_] for _ in top_n if _ in inverse_idx]

def similarity(word_1, word_2):
    return cosine_similarity(
        word_embedding[word_1].reshape(1, -1), 
        word_embedding[word_2].reshape(1, -1)
    ).flatten()[0]
print('Similarity between "money" and "cash" %.4f' % similarity('money', 'cash'))
print('Similarity between "atm" and "bank" %.4f' % similarity('atm', 'bank'))
print('Similarity between "congress" and "rahul" %.4f' % similarity('liar', 'rahul'))
print('Similarity between "bjp" and "modi" %.4f' % similarity('liar', 'modi'))
import matplotlib.pyplot as plt

plt_x = points.transpose()[0, 1:]
plt_y = points.transpose()[1, 1:]
fig = plt.figure(figsize=(15, 250))
ax = fig.subplots()
ax.scatter(plt_x, plt_y)

for i, txt in enumerate([_ for _ in tokenizer.word_index]):
    if i%5 == 0:
        ax.annotate(txt, (plt_x[i], plt_y[i]))

plt.show()