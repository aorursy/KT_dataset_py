import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

df_test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
df_train.head()
labels = ['obscene', 'insult', 'toxic', 'severe_toxic', 'identity_hate', 'threat']



for label in labels :

    print("Entidade: ", label)

    print(df_train[label].value_counts(), '\n')
df_test.head()
submission.head()
print("Quantidade de valores faltantes nos dados de treino:")

df_train.isna().sum()
print("Quantidade de valores faltantes nos dados de teste:")

df_test.isna().sum()
coment_nulo = {}

coment_nulo['Treino'] = {'Quantidade' : len(df_train[df_train['comment_text'].isnull()])}

coment_nulo['Teste'] = {'Quantidade' : len(df_test[df_test['comment_text'].isnull()])}



print("Comentários nulos nos dados de:")

for key in coment_nulo :

    print(str(key) + ' = ' + str(coment_nulo[key]['Quantidade']))
df_train.describe()
df_train.shape, df_test.shape
comments_unlabelled_train = df_train[(df_train['toxic'] != 1) & (df_train['severe_toxic'] != 1) & 

                                     (df_train['obscene'] != 1) & (df_train['threat'] != 1) & 

                                     (df_train['insult'] != 1) & (df_train['identity_hate'] != 1)]



print('Percentual de comentários sem classificação: ', str(len(comments_unlabelled_train) / len(df_train)*100) + 

      '%\nQuantidade de comentários de cada categoria:')

print(df_train[labels].sum())
data = df_train[labels]



colormap = plt.cm.coolwarm

plt.figure(figsize = (8,8))



sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', 

            annot=True);
df_train['comment_text']
df_test['comment_text']
def padroniza_df(df, func) :

    

    df = df.map(lambda coment : func(coment))

                

    return df
import re



def padroniza_texto(texto):

    

    texto = texto.encode('ascii', errors = 'ignore').decode() #Decodificando caracteres em ASCII

    texto = texto.lower() #Apenas caracteres minúsculos

    texto = re.sub(r'http\S+', ' ', texto) #Evitando links

    texto = re.sub(r'#+', ' ', texto)

    texto = re.sub(r'@[A-Za-z0-9]+', ' ', texto)

    texto = re.sub(r"([A-Za-z]+)'s", r"\1 is", texto)

    texto = re.sub(r"what's", "what is ", texto) #Evitando contrações

    texto = re.sub(r"\'s", " ", texto) #Evitando contrações

    texto = re.sub(r"won't", "will not ", texto) #Evitando contrações

    texto = re.sub(r"\'ve", " have ", texto) #Evitando contrações

    texto = re.sub(r"can't", "can not ", texto) #Evitando contrações

    texto = re.sub(r"n't", " not ", texto) #Evitando contrações

    texto = re.sub(r"isn't", "is not ", texto) #Evitando contrações

    texto = re.sub(r"i'm", "i am ", texto) #Evitando contrações

    texto = re.sub(r"\'re", " are ", texto) #Evitando contrações

    texto = re.sub(r"\'d", " would ", texto) #Evitando contrações

    texto = re.sub(r"\'ll", " will ", texto) #Evitando contrações

    texto = re.sub(r"\'scuse", " excuse ", texto) #Evitando contrações

    texto = re.sub('\W', ' ', texto)

    texto = re.sub('\s+', ' ', texto)

    texto = re.sub(r'\d+', ' ', texto)

    texto = texto.strip(' ') #Removendo espaços do começo e fim 

    

    return texto
df_train['comment_text'] = padroniza_df(df_train['comment_text'], padroniza_texto)

df_test['comment_text'] = padroniza_df(df_test['comment_text'], padroniza_texto)
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



def remove_stopwords(texto):

    

    stop_words = set(stopwords.words('english')) 

  

    word_tokens = word_tokenize(texto) 

  

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

  

    filtered_sentence = [] 

  

    for w in word_tokens: 

        if w not in stop_words: 

            filtered_sentence.append(w) 

        

    return filtered_sentence
df_train['comment_text'] = padroniza_df(df_train['comment_text'], remove_stopwords)

df_test['comment_text'] = padroniza_df(df_test['comment_text'], remove_stopwords)
df_train['comment_text'].head()
df_test['comment_text'].head()
X_train = df_train['comment_text']

X_test = df_test['comment_text']
X_train.shape, X_test.shape
y_train = np.asarray(df_train[labels].values)
y_train.shape
# Imports necessários

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
# Função que conta a quantidade de palavras

def word_count(vector):

    

    count = 0

    

    for word in vector :

        count += 1



    return count
new_df_train = df_train

new_df_train['number_of_words'] = df_train['comment_text'].apply(lambda x: word_count(x))



new_df_train.nlargest(5, 'number_of_words')
new_df_test = df_test

new_df_test['number_of_words'] = df_test['comment_text'].apply(lambda x: word_count(x))



new_df_test.nlargest(5, 'number_of_words')
X_train_tokenizer = Tokenizer(num_words=1500)

X_train_tokenizer.fit_on_texts(X_train)
X_train_tokens = X_train_tokenizer.texts_to_sequences(X_train)

X_train_tokens = pad_sequences(X_train_tokens, maxlen=100)

                               

X_train_tokens
X_train.shape, X_train_tokens.shape
X_test_tokenizer = Tokenizer(num_words=1500)

X_test_tokenizer.fit_on_texts(X_test)
X_test_tokens = X_test_tokenizer.texts_to_sequences(X_test)

X_test_tokens = pad_sequences(X_test_tokens, maxlen=100)



X_test_tokens
X_test.shape, X_test_tokens.shape
# Imports necessários

from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, Dense

from tensorflow.keras import Sequential
model = Sequential()

model.add(Embedding(input_dim=1500, output_dim=64))

model.add(Bidirectional(LSTM(64)))

model.add(Dense(64, activation='relu'))

model.add(Dense(6, activation='sigmoid'))
from tensorflow.keras.optimizers import Adam



model.compile(loss='binary_crossentropy',

              optimizer=Adam(1e-4),

              metrics=['accuracy'])
model.summary()
history = model.fit(X_train_tokens, y_train, batch_size=32, verbose=1, epochs=10, validation_split=0.02, shuffle=True)
history.history
y_test_pred = model.predict_proba(X_test_tokens, batch_size=32)
submission[labels] = y_test_pred



submission.head()
submission.to_csv('submission.csv', index=False)