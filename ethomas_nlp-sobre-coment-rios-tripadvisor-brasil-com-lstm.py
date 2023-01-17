# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import bs4
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Tripadvisor_SampleData.csv")
df.head()
df.drop(df[df['stars'] == '3'].index, inplace=True)
df.loc[df['stars'] < '3', 'Y'] = int(0)
df.loc[df['stars'] > '3', 'Y'] = int(1)
df.head()
df['comment'] = df['content'] + ' ' + df['title']
df.drop(['content', 'title', 'stars'], axis=1, inplace=True)
df.head()
comments = df['comment'].get_values()
sentiment = df['Y'].get_values()
import re
import unidecode
from nltk.corpus import stopwords

from tqdm import tqdm_notebook, tqdm

clean_comments=[]

pbar = tqdm_notebook(total=len(comments))
for i in comments:
    
    a = bs4.BeautifulSoup(i, 'html.parser').get_text()
    
    #remover acentuação
    a = unidecode.unidecode(a)
    
    #remover caracteres excepcionais
    a = re.sub('[^a-zA-Z ]', '', a)
    
    #converter para caixa baixa
    a = a.lower()
    
    #remover as stopwords
    a = a.split()
    novo_a = []
    for word in a:
        if not word in stopwords.words('portuguese'):
            novo_a.append(word)
    
    clean_comments.append(novo_a)
    
    pbar.update(1)
clean_comments[0]
from collections import Counter
word_freq = Counter()

pbar = tqdm_notebook(total=len(comments))
for comment in clean_comments:
    for word in comment:
        word_freq[word] += 1
    pbar.update(1)

print('Tamanho do dicionário de palavras gerado: ', len(word_freq))
top_words = word_freq.most_common(3000)

top_words[:10]
word_to_id = {}

idx = 3000
for t in top_words:
    word_to_id[t[0]] = idx
    idx -= 1

    
#Para as palavras que não estão entre as top 3000
#será aplicado o ID 0 (zero)
for word in word_freq.keys():
    if not word in word_to_id:
        word_to_id[word] = 1
len(word_freq)
data = []

for comment in clean_comments:
    aux = []
    for word in comment:
        aux.append(word_to_id[word])
        
    data.append(aux)
    
data[1]
from keras.preprocessing.sequence import  pad_sequences

data = pad_sequences(data, maxlen=100, padding='post')
data, sentiment
data.shape
from keras.layers import *
from keras.models import Model
input_node = Input(shape=(100,))
embedding_layer = Embedding(input_dim=3000, input_length=100, output_dim=32)(input_node)
dropout = Dropout(0.5)(embedding_layer)
lstm = LSTM(100)(dropout)
dropout = Dropout(0.5)(lstm)
output_node = Dense(1, activation='sigmoid')(dropout)

model = Model(input_node, output_node)

model.summary()
model.compile('Adam', loss='binary_crossentropy', metrics=['accuracy'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, sentiment, test_size=0.33)

from keras.callbacks import *

early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5)

cb_list = [early_stopping]

model.fit(X_train, y_train, batch_size=64, epochs=20,
         validation_data=(X_test, y_test), callbacks=cb_list)


new_comment = 'Hotel razoável, mas existem problemas sérios no atendimento. Funcionários sem compromisso e mal humorados'

review_text = bs4.BeautifulSoup(new_comment, 'html.parser').get_text()
    
#remover acentuação
a = unidecode.unidecode(review_text)
    
#remover caracteres excepcionais
a = re.sub('[^a-zA-Z ]', '', a)
    
#converter para caixa baixa
a = a.lower()
    
#remover as stopwords
a = a.split()
novo_a = []
for word in a:
    if not word in stopwords.words('portuguese'):
        novo_a.append(word)

processed_new_reviews = []
for word in novo_a:
    processed_new_reviews.append(word_to_id[word])

processed_data = np.asarray(processed_new_reviews).reshape(1, len(processed_new_reviews))
processed_data = pad_sequences(processed_data, maxlen=100, padding='post')
y_pred = model.predict(processed_data)[0]

if np.round(y_pred) == 1:
    sent = 'positivo'
else:
    sent = 'negativo'

print('A predição do sentimento para a entrada \"{}\" é {}'.format(new_comment, sent))
y_pred