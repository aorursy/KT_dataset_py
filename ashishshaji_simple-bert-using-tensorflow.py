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
sms=pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin1')
sms = sms.iloc[:,[0,1]]

sms.head()
sms.columns = ["label", "message"]
df=sms
sms.describe()
df.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x="label", data=sms);
plt.show()
from wordcloud import WordCloud

plt.figure(figsize = (15,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(sms[sms.label == "spam"].message))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (15,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(sms[sms.label == "ham"].message))
plt.imshow(wc , interpolation = 'bilinear')
import nltk
import string
from nltk.corpus import stopwords
import re
def rem_punctuation(text):
  return text.translate(str.maketrans('','',string.punctuation))

def rem_numbers(text):
  return re.sub('[0-9]+','',text)


def rem_urls(text):
  return re.sub('https?:\S+','',text)


def rem_tags(text):
  return re.sub('<.*?>'," ",text)



df['message'].apply(rem_urls)
df['message'].apply(rem_punctuation)
df['message'].apply(rem_tags)
df['message'].apply(rem_numbers)
stop = set(stopwords.words('english'))

def rem_stopwords(df_news):
    
    words = [ch for ch in df_news if ch not in stop]
    words= "".join(words).split()
    words= [words.lower() for words in df_news.split()]
    
    return words    

df['message'].apply(rem_stopwords)
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
  lemmas = []
  for word in text.split():
    lemmas.append(lemmatizer.lemmatize(word))
  return " ".join(lemmas)


df['message'].apply(lemmatize_words)
df.dtypes
encode = ({'ham': 0, 'spam': 1} )
#new dataset with replaced values
df = df.replace(encode)
import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras 
from keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM,Dense,Bidirectional,Input
from keras.models import Model
from sklearn.model_selection import train_test_split
import transformers
x_train,x_test,y_train,y_test = train_test_split(df.message,df.label,random_state = 0,stratify = df.label)
from tokenizers import BertWordPieceTokenizer
# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased' , lower = True)
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True)
fast_tokenizer
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=400):

    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
x_train = fast_encode(x_train.values, fast_tokenizer, maxlen=400)
x_test = fast_encode(x_test.values, fast_tokenizer, maxlen=400)
def build_model(transformer, max_len=400):
    
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
bert_model = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
model = build_model(bert_model, max_len=400)
model.summary()
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs = 4)
print("Accuracy of the model on Testing Data is - " , model.evaluate(x_test,y_test)[1]*100 , "%")