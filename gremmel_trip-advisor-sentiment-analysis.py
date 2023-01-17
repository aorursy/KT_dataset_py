# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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
df.loc[df['stars']> '3', 'sentiment'] = int(1)
df.loc[df['stars']< '3', 'sentiment'] = int(0)

df["opinion"] = df["content"]+ " " + df["title"]

df.head()

reviews = df['opinion'].get_values()
sentiment = df['sentiment'].get_values()

print(reviews[:5])
import bs4
import re
from unicodedata import normalize
from nltk.corpus import stopwords
import operator
from keras.preprocessing import sequence
from tqdm import tqdm_notebook, tqdm
clean_reviews = []   
pbar = tqdm_notebook(total=len(reviews))

for review in reviews:
    # Remover tags HTML
    review_text = bs4.BeautifulSoup(review, 'html.parser').get_text()

    # Remover caracteres especiais, pontuacao e numeros
    review_text = re.sub('[^a-zA-Z]', ' ', review_text)

    # Converter para caixa baixa
    review_text = review_text.lower()

    # Vetorizar o comentário
    review_words = review_text.split()

    # Remover stopwords
    stops = stopwords.words('portuguese')

    meaningful_words = [word for word in review_words if not word in stops]

    clean_reviews.append(meaningful_words)
    pbar.update(1)


# Construindo dicionário de frequencia
freq_dict = {}
topwords = 5000
maxlen = 100
pbar = tqdm_notebook(total=len(reviews))

for review in clean_reviews:
    pbar.update(1)
    for word in review:
        if not word in freq_dict:
            freq_dict[word] = 0
        freq_dict[word] += 1

# Selecionar as top-K palavras (jeito inteligente Ass: carlos)
sorted_tup = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)

word_to_id = {}
cnt = topwords - 1
# Top-K palavras
for i in sorted_tup[:topwords]:
    pbar.update(1)
    word_to_id[i[0]] = cnt
    cnt -= 1
# Restante
for i in sorted_tup[topwords:]:
    pbar.update(1)    
    word_to_id[i[0]] = 0

# Mapeando palavras para um id do dicionário
processed_data = []

for review in clean_reviews:
    pbar.update(1)
    aux = []
    for word in review:
        aux.append(word_to_id[word])

    processed_data.append(aux)

# Realizando o padding dos comentarios
## importar sequence de keras.preprocessing
processed_data = np.asarray(processed_data)
processed_data = sequence.pad_sequences(processed_data, maxlen)
from keras.models import Model
from keras.layers import *
input_node = Input(shape=(100,))

embedding = Embedding(input_dim=5000, 
                      input_length=100, 
                      output_dim=32)(input_node)
dropout = Dropout(0.5)(embedding)
lstm_1 = LSTM(100)(dropout)
dropout = Dropout(0.5)(lstm_1)
fc1 = Dense(1, activation='sigmoid')(dropout)

model = Model(input_node, fc1)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='Adam',
              metrics=['accuracy'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_data, sentiment, 
                                                    test_size=0.33)
from keras.callbacks import *
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=3)
cb_list = [early_stopping]
model.fit(X_train, y_train, batch_size=64, epochs=20,
         validation_data=(X_test, y_test), callbacks=cb_list)
new_review = 'cama dura, bom para a coluna'

# Remover tags HTML
review_text = bs4.BeautifulSoup(new_review, 'html.parser').get_text()
# Remover caracteres especiais, pontuacao e numeros
review_text = re.sub('[^a-zA-Z]', ' ', review_text)
# Converter para caixa baixa
review_text = review_text.lower()
# Vetorizar o comentário
review_words = review_text.split()
# Remover stopwords
stops = stopwords.words('portuguese')

meaningful_words = [word for word in review_words if not word in stops]

processed_new_reviews = []
for word in meaningful_words:
    if word not in word_to_id:
        processed_new_reviews.append(0)
    else:
        processed_new_reviews.append(word_to_id[word])

processed_data = np.asarray(processed_new_reviews).reshape(1, len(processed_new_reviews))
processed_data = sequence.pad_sequences(processed_data, 100)

processed_data
y_pred = model.predict(processed_data)[0]
if np.round(y_pred) == 1:
    sent = 'positivo'
else:
    sent = 'negativo'

print('A predição do sentimento para a entrada \"{}\" é {}'.format(new_review, sent))
y_pred