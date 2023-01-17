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
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.layers as L
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import seaborn as sns
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint  
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from tqdm import tqdm
nltk.download('stopwords')

df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.isnull().sum()

df.head()

df['sentiment'].value_counts()

def encode_sentiment(sentiment) -> int:
  encoder = {"positive" : 1 , "negative": 0 ,"Positive":1 , "Negative" : 0}
  if sentiment!=None:
    return encoder[sentiment]

def decode_sentiment(sentiment) -> str:
  decoder = {1: "positive" , 0: "negative"}
  if sentiment!=None:
    return decoder[sentiment]
train_df = df


train_df['new_sentiment'] = train_df['sentiment'].apply(encode_sentiment)
train_df = train_df.sample(frac=1)

def process_text(text) -> str:
  #Convert string 
  process_text = str(text)
  #Convert string to lower
  process_text = process_text.lower()
  #Removing html tags
  process_text = re.sub("<.*?>"," ",process_text)
  #Removing all digits and having only letters
  process_text = re.sub("[^a-zA-Z]"," ",process_text)
  #Removing all stop words
  process_text = process_text.split(" ")
  process_text = " ".join([word for word in process_text if word not in stopwords.words("english")])

  return process_text
tqdm.pandas()

train_df['processed_text'] = train_df['review'].progress_apply(process_text)
positive_sentiment = train_df.loc[train_df['new_sentiment']==1]
negative_sentiment = train_df.loc[train_df['new_sentiment']==0]
positive_sentiment.head()

positive_string = ' '.join([word for word in positive_sentiment['processed_text'].values])
# Display the generated image:
wordcloud = WordCloud().generate(positive_string)
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
negative_string = ' '.join([word for word in negative_sentiment['processed_text'].values])
# Display the generated image:
negative = WordCloud().generate(negative_string)
plt.figure(figsize=(10,8))
plt.imshow(negative, interpolation='bilinear')
plt.axis("off")
plt.show()
train = train_df[['processed_text','new_sentiment']]

text = train['processed_text'].values
sentiment = train['new_sentiment'].values
tokenizer = Tokenizer()

tokenizer.fit_on_texts(text)
print(f"Total vocab :  {len(tokenizer.word_index)+1}")
sequence_text = tokenizer.texts_to_sequences(text)

sequence_text = pad_sequences(sequence_text,padding="post")
x_train , x_test , y_train ,y_test = train_test_split(sequence_text,sentiment,
                                                      test_size=0.3,random_state=11)

print(f"""
X Train :{x_train.shape}
X TEST : {x_test.shape}
Y TRAIN : {y_train.shape}
Y TEST  : {y_test.shape}
""")

VOCAB_SIZE = len(tokenizer.word_index)+1
EMBEDDING_VEC = 30
model = tf.keras.Sequential([
    L.Embedding(VOCAB_SIZE,EMBEDDING_VEC, input_length=x_train.shape[1]),
    L.Bidirectional(L.LSTM(128,return_sequences=True)),
    L.GlobalMaxPool1D(),
    L.Dropout(0.4),
    L.Dense(128, activation="relu"),
    L.Dropout(0.4),
    L.Dense(2)
    ])
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',metrics=['accuracy']
             )
model.fit(x_train,y_train , validation_data=(x_test,y_test),batch_size=32
          ,epochs=2)
predictions = model.predict_classes(x_test)
print(f"The accuracy of the model is : {accuracy_score(y_test,predictions)*100}%")
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True,fmt='d')
plt.title("Confusion Matrix")
print(classification_report(y_test,predictions))
decoded = [decode_sentiment(i) for i in predictions]
sns.countplot(decoded)
plt.title("Total number of positive and negative predictions over 15000 samples")
sample_data = [["""I know everyone wants to compare this to the Animated version, but don't. Take it as it comes and you will thoroughly enjoy it. It does stay pretty faithful to the animated version I think. Will Smith as the genie could never be the Robyn Williams genie, but I don't think he tries to. He does fantastically well in his own right. Absolutely loved the Prince Ali song where Aladdin enters the city as the prince. Brilliantly colorful spectacle captured really well. Jafar missed a little for me as had lost the smarmy-ness of the animated version. The songs were great and the Aladdin and Jasmine characterization was pretty spot on. 
I think kids would love this and I would definitely recommend it. """],
              [ """
               Everything felts small and Bollywood, for being the middle east. felt like I was watching a b film remake of Arabian Nights, not Aladdin from the Disney family. 
               way to kill a childhood of mine...save your money for Lion King.
               """],
               ["""
                This movie sucks
               """],
               ]
for i in sample_data:
    process = process_text(i)
    process = tokenizer.texts_to_sequences([process])
    process = pad_sequences(process,padding='post')
    prediction = int(model.predict_classes(process))
    print(f"Predicted : {decode_sentiment(prediction)}")