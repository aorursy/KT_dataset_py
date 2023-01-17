import pandas as pd
import numpy as np

train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
test  = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
train.columns


test.describe()




train.describe()
train['length'] = train['comment_text'].apply(len)
train.head()
train.describe()
train['length'].hist(bins=1000)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['comment_text'])
train_x = tokenizer.texts_to_sequences(train['comment_text'])
train_x = pad_sequences(train_x, maxlen=300)


print("Word count:",len(tokenizer.word_counts))

print("Document count:", tokenizer.document_count)

print("Word Index:",len(tokenizer.word_index))

print("Word Document",len(tokenizer.word_docs))

embeddings_index = {}
file = '../input/glove840b300dtxt/glove.840B.300d.txt'
with open(file, encoding='utf8') as f:
    for line in f:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
 
print('Loaded %s word vectors.' % len(embeddings_index))
#Now prepare coefficient matrix
num_words = min(len(embeddings_index),  len(tokenizer.word_index))
print(num_words)
embedding_matrix = np.zeros((num_words, 300))
total_invalid_record_count = 0
for item in tokenizer.word_index.items():
    try:
        word, index = item[0], item[1]
        embedding_matrix[index] = embeddings_index[word]
    except Exception as e:
        print("Exception occured for record:", word)
        total_invalid_record_count += 1
print("total_invalid_record_count:",total_invalid_record_count)
        
    
   

print("total_invalid_record_count:",total_invalid_record_count)
print("Word Index:",len(tokenizer.word_index))
correct_records = len(tokenizer.word_index) - total_invalid_record_count
print("Total correct records:",correct_records)
embedding_matrix = np.zeros((correct_records, 300))
total_invalid_record_count = 0
index_val = 0
for item in tokenizer.word_index.items():
    try:
        word, index = item[0], item[1]
        embedding_matrix[index_val] = embeddings_index[word]
        index_val += 1
    except Exception as e:
        total_invalid_record_count += 1
print("total_invalid_record_count:",total_invalid_record_count)
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras.models import Sequential
model = Sequential()
model.add(Embedding(correct_records, 300, weights=[embedding_matrix], input_length=300))
model.add(LSTM(128))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
from sklearn.model_selection import train_test_split
#train_x = train['comment_text']
train_y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33, random_state=42)
model.fit(X_train, y_train, epochs=1, batch_size=64)
tokenizer.fit_on_texts(test['comment_text'])
test_x = tokenizer.texts_to_sequences(test['comment_text'])
test_x = pad_sequences(test_x, maxlen=300)
predictions = model.predict(test_x, batch_size=64, verbose=1)
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = predictions
submission.to_csv('submission.csv', index=False)
