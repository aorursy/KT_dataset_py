import pandas as pd

df = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

df.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['sentiment'] = le.fit_transform(df.sentiment)

df.head()
### Defining hyperparameters for Text Processing
TOP_WORDS      = 5000      # to keep only 5000 top used words

MAX_REVIEW_LEN = 500       # caps the sequence length to this number (Keras requires all sequences to be of same length)

OOV_TOKEN      = '<OOV>'   # any out of vocabulary word (not part of top words) is replaced with this text

TRUNC_TYPE     = 'post'

PADDING_TYPE   = 'post'

TEST_SIZE      = 0.5

EMBEDDING_LEN  = 32 

EPOCHS         = 10

BATCH_SIZE     = 64
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer(num_words = TOP_WORDS, oov_token = OOV_TOKEN)

tokenizer.fit_on_texts(df.review.to_numpy())

word_index = tokenizer.word_index

word_index_inv = dict([(v,k) for (k,v) in word_index.items()])
def decode_sentence(text):

    return ' '.join([word_index_inv.get(i, '?') for i in text])



sample_seq = [' '.join(df.review[0].split(' ')[:50])]

tokenized_sample = tokenizer.texts_to_sequences(sample_seq)

print (sample_seq[0])

print ('------------------')

print (tokenized_sample[0])

print ('------------------')

print (decode_sentence(tokenized_sample[0]))
reviews = df.review.to_numpy()

labels  = df.sentiment.to_numpy()



train_count      = int(len(reviews) * (1 - TEST_SIZE))

training_reviews = reviews[:train_count]

testing_reviews  = reviews[train_count:]

y_train          = labels[:train_count]

y_test           = labels[train_count:]



print ('Training Count :', len(training_reviews))

print ('Testing Count :', len(testing_reviews))
training_sequences = tokenizer.texts_to_sequences(training_reviews)

X_train            = pad_sequences(training_sequences, maxlen = MAX_REVIEW_LEN, padding = PADDING_TYPE, truncating = TRUNC_TYPE)



testing_sequences  = tokenizer.texts_to_sequences(testing_reviews)

X_test             = pad_sequences(testing_sequences,  maxlen = MAX_REVIEW_LEN, padding = PADDING_TYPE, truncating = TRUNC_TYPE)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D

from tensorflow.keras.optimizers import Adam



model = Sequential() 

model.add(Embedding(TOP_WORDS, EMBEDDING_LEN, input_length=MAX_REVIEW_LEN))

model.add(GlobalAveragePooling1D())

model.add(Dense(100, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid')) 

model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0005), metrics = ['accuracy']) 

model.summary()

%time history = model.fit(X_train, y_train, validation_data =(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose = 0)
results = model.evaluate(X_test, y_test, batch_size = 128, verbose = 0)

print (f'Accuracy : {round(results[1]*100, 2)} %')
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()





plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()   
model = Sequential() 

model.add(Embedding(TOP_WORDS, EMBEDDING_LEN, input_length=MAX_REVIEW_LEN))

model.add(LSTM(100))

model.add(Dense(1, activation = 'sigmoid')) 

model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.005), metrics = ['accuracy']) 

model.summary()

%time history = model.fit(X_train, y_train, validation_data =(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose = 0)
results = model.evaluate(X_test, y_test, batch_size = 128, verbose = 0)

print (f'Accuracy : {round(results[1]*100, 2)} %')
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()





plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()   
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D, Dropout



model = Sequential() 

model.add(Embedding(TOP_WORDS, EMBEDDING_LEN, input_length=MAX_REVIEW_LEN))

model.add(Dropout(0.2)) 

model.add(LSTM(100))

model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid')) 

model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.005), metrics = ['accuracy']) 

model.summary()

%time history = model.fit(X_train, y_train, validation_data =(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose = 0)
results = model.evaluate(X_test, y_test, batch_size = 128, verbose = 0)

print (f'Accuracy : {round(results[1]*100, 2)} %')
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()





plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()   
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D



model = Sequential() 

model.add(Embedding(TOP_WORDS, EMBEDDING_LEN, input_length = MAX_REVIEW_LEN))

model.add(Conv1D(32, (3), activation = 'relu')) 

model.add(MaxPooling1D(2)) 

model.add(LSTM(100))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.005), metrics = ['accuracy']) 

model.summary()

%time history = model.fit(X_train, y_train, validation_data =(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose = 0)
results = model.evaluate(X_test, y_test, batch_size = 128, verbose = 0)

print (f'Accuracy : {round(results[1]*100, 2)} %')
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('LOSS')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()





plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ACCURACY')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()   