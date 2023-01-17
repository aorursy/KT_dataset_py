import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import string

from tqdm import tqdm

import re



# nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords





# sklearn

from sklearn.model_selection import train_test_split



# keras and tensorflow

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from keras.optimizers import Adam



import tensorflow as tf



stop = set(stopwords.words('english'))
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')
df_train.head()
df_train.tail()
# shape

print(f"There are {df_train.shape[0]} rows and {df_train.shape[1]} columns. ")
# Class Distribution

# 0 (Non Disaster) is more than 1 (Disaster) Tweets

class_dist = df_train.target.value_counts()

sns.barplot(class_dist.index,class_dist)
# Misssing vs Non Missing

# 'keyword' & 'location' columns have missing values. Looks like 'location' column is very dirty so lets not use it. Lets also ignore 'keyword' column for now.

null_vals = df_train.isnull().sum()

sns.barplot(null_vals.index,null_vals)
# Removing <> tags



def remove_spec(text):

    text = re.sub('<.*?>+', '', text)

    text = text.lower()

    return text



# Rmoving puntuctions



def remove_punctuation(text):

    table = str.maketrans('','',string.punctuation)

    return text.translate(table)



# Removing URL



def remove_urls(text):

    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+",'',text)

    return text



# Removing Emojis



def remove_emoji(text):

    emojis = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    

    text = re.sub(emojis,'',text)

    return text

df_train['cleaned_text'] = df_train['text'].apply(lambda x : remove_punctuation(x))

df_train['cleaned_text'] = df_train['cleaned_text'].apply(lambda x : remove_urls(x))

df_train['cleaned_text'] = df_train['cleaned_text'].apply(lambda x : remove_emoji(x))

df_train['cleaned_text'] = df_train['cleaned_text'].apply(lambda x : remove_spec(x))
# Creating Words Corpus



def create_corpus(dataset):

    corpus = []

    for review in tqdm(dataset['cleaned_text']):

        words = [ word.lower() for word in word_tokenize(review) if (word.isalpha() == 1 ) & (word not in stop) ]

        corpus.append(words)



    return corpus



corpus = create_corpus(df_train)

# Creating Embedding Dictionary



embedding_dict={}

with open('../input/glove-100d/glove.6B.100d.txt','r', encoding='utf8') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
# Tokenize : break the sentence into single word/token

# texts_to_sequences : convert tokenized word into an encoded sequnce

# pad_sequence : change the length of sequence by either adding or truncating



MAX_LEN = 20 

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)



sequences = tokenizer.texts_to_sequences(corpus)



corpus_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
# Unique words present

word_index = tokenizer.word_index

print(f"Number of unique words : {len(word_index)}")
# Creating embedding matrix with GloVe using enbedding_dict we created above

num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec
# Long Short Term Memory network.



# We need sequential model to process sequence of text data

model=Sequential()



# Embedding(input_dimension, output_dimension,embeddings_initializer = initialize the embedding matrix we created, trainable = do not train)

embedding=Embedding(num_words,100,

                    embeddings_initializer=Constant(embedding_matrix),

                    input_length=MAX_LEN,

                    trainable=False)

# Adding Embedding Layer

model.add(embedding)



# Drops 40% of entire row

model.add(SpatialDropout1D(0.4))



# Recurrent Layer LSTM(dimensionality of the output space, dropout = 20%, recurrent_dropout = 20%) 

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))



# Decide what we are going to output Dense(units, activation function)

model.add(Dense(1, activation='sigmoid'))



# Compile the model compile(loss = binary crossentropy, use Adam(adaptive moment estimation) optimizer with learning rate 1e-3,evaluate based on accuracy)

model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=1e-4),metrics=['accuracy'])



model.summary()
X_train,X_test,y_train,y_test = train_test_split(corpus_pad, df_train['target'].values, test_size = 0.25, random_state = 0 )



print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)



print('Shape of train',y_train.shape)

print("Shape of Validation ",y_test.shape)
history=model.fit(X_train,y_train,batch_size=32,epochs=30,validation_data=(X_test,y_test),verbose=2)
# Accuracy vs Epoch

plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show();
# Loss vs Epoch

# Visualize learning curve. Here learning curve is not ideal. It should be much smoother as it decreases.



epoch_count = range(1, len(history.history['loss']) + 1)

plt.plot(epoch_count, history.history['loss'], 'r--')

plt.plot(epoch_count, history.history['val_loss'], 'b-')

plt.legend(['Training Loss', 'Validation Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()
# Clean Test data



df_test['cleaned_text'] = df_test['text'].apply(lambda x : remove_punctuation(x))

df_test['cleaned_text'] = df_test['cleaned_text'].apply(lambda x : remove_urls(x))

df_test['cleaned_text'] = df_test['cleaned_text'].apply(lambda x : remove_emoji(x))

df_test['cleaned_text'] = df_test['cleaned_text'].apply(lambda x : remove_spec(x))
# Creating corpus

test_corpus = create_corpus(df_test)
# Encoding Test Text to Sequences

test_sequences = tokenizer.texts_to_sequences(test_corpus)



test_corpus_pad = pad_sequences(test_sequences, maxlen=MAX_LEN, truncating='post', padding='post')
# Predictions

predictions = model.predict(test_corpus_pad)

predictions = np.round(predictions).astype(int).reshape(3263)
# Creating submission file 

submission = pd.DataFrame({'id' : df_test['id'], 'target' : predictions})

submission.to_csv('final_submission.csv', index=False)



submission.head()