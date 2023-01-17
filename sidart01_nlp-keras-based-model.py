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
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import string
import re
import nltk
import tqdm
import matplotlib.pyplot as plt
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_train.tail()
df_train
df_train = df_train.dropna(thresh=2)
df_train['keyword'] = df_train['keyword'].fillna("INVALIDkeyword")
df_test['keyword'] = df_test['keyword'].fillna("INVALIDkeyword")
df_train['location'] = df_train['location'].fillna("INVALIDlocation")
df_train['target'].value_counts().plot.bar()
tqdm.tqdm(nltk.download('punkt'))
punct = string.punctuation
stopwords = nltk.corpus.stopwords.words()
wl = nltk.WordNetLemmatizer()


def help(text):
    text = text.lower()
    text = "".join(word for word in text if word not in punct)
    text = re.split("\W+", text)
    text = " ".join(wl.lemmatize(word) for word in text if word not in stopwords)
    return text
df_train['refine_text'] = df_train['text'].apply(lambda x: help(x))
df_test['refine_text'] = df_test['text'].apply(lambda x: help(x))
df_train.head()

vocab_size = 10000
embedding_dim = 16
max_len = 150
trunc_type = "post"
oov_tok = "<OOV>"
training_size = 20000
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df_train['refine_text'])
word_index = tokenizer.word_index

train_word_sequence = tokenizer.texts_to_sequences(df_train['refine_text'])
train_padd_sequence = pad_sequences(train_word_sequence, maxlen=max_len, truncating=trunc_type)

test_word_sequence = tokenizer.texts_to_sequences(df_test['refine_text'])
test_padd_sequence = pad_sequences(test_word_sequence, maxlen=max_len, truncating=trunc_type)

df_train['text_len'] = df_train['text'].apply(lambda x : len(x) - x.count(" "))
df_test['text_len'] = df_test['text'].apply(lambda x : len(x) - x.count(" "))

def count_capital(text):
    text = re.split("\W+", text)
    count = sum([1 for word in text if word.isupper()])
    return count

df_train['capital'] = df_train['text'].apply(lambda x: count_capital(x))
df_test['capital'] = df_test['text'].apply(lambda x: count_capital(x))

df_train.head()
pd.plotting.hist_series(df_train['text_len'])
print(df_train['text_len'].max())



bins = np.linspace(0,150,20)
plt.hist(df_train['text_len'] ** 1/2, bins)
plt.title("Text len distribution")
plt.show()
from sklearn.feature_extraction.text import CountVectorizer
tfidf_vect= CountVectorizer()
#print(df_train['keyword'])
x_feature_keyword = tfidf_vect.fit_transform(df_train['keyword'])
x_feature_keyword = x_feature_keyword.toarray()

x_test_keyword = tfidf_vect.fit_transform(df_test['keyword'])
x_test_keyword = x_test_keyword.toarray()
X_features = pd.concat([df_train['capital'], pd.DataFrame(x_feature_keyword), pd.DataFrame(train_padd_sequence)],axis=1 )
X_test_features = pd.concat([df_test['capital'], pd.DataFrame(x_test_keyword), pd.DataFrame(test_padd_sequence)],axis=1 )
X_features.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, df_train['target'], test_size = 0.2)
print(X_train.shape)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len+2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])
model.summary()
model.compile(optimizer="adam", loss = tf.keras.losses.binary_crossentropy, metrics = ["accuracy"])
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 2)
y_pred = model.predict_classes(X_test_features)

sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

y_pred = np.round(y_pred).astype(int).reshape(3263)
sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pred})
sub.to_csv('submission.csv',index=False)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(sample_sub['target'],y_pred)

print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)

plt.title('Confusion matrix')
fig.colorbar(cax)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



