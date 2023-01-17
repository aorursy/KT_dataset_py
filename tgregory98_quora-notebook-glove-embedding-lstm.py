# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

path = "../input" # Change to "../input" on Kaggle, or "all" on local

print(os.listdir(path))



# Any results you write to the current directory are saved as output.



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model

from keras.layers import Embedding, Flatten, Dense, LSTM, Bidirectional, Dropout, Activation, GlobalMaxPool1D, Input

from tqdm import tqdm

from sklearn import metrics
train = pd.read_csv(path + "/train.csv")

test = pd.read_csv(path + "/test.csv")

sample = pd.read_csv(path + "/sample_submission.csv")



print("training data: " + str(train.shape))

# print(train.head()) I have commented this code out to save space



print("test data: " + str(test.shape))

# print(test.head()) I have commented this code out to save space



print("sample data: " + str(sample.shape))

# print(sample.head()) I have commented this code out to save space
max_words = 60000 # This has been tested in the range 50,000-100,000 and this gave a good score



tokenizer_class = Tokenizer(num_words=max_words, lower=True) # Setting up the tokenizer class

tokenizer_class.fit_on_texts(train['question_text'])

train_data = tokenizer_class.texts_to_sequences(train['question_text']) # List of number sequences, each corresponding to a question

output_test_data = tokenizer_class.texts_to_sequences(test['question_text']) # Test sequences for later



train_targets = train['target']



word_dict = tokenizer_class.word_index # The dictionary for this tokenizer class

number_of_words = min(max_words, len(word_dict)) # We work out how many words we're actually using (capped by max_words value)
sequence_lengths = []

for sequence in train_data:

    sequence_lengths.append(len(sequence))



print(np.amax(sequence_lengths))

print(np.mean(sequence_lengths))

print(np.std(sequence_lengths))

print(np.sum(np.array(sequence_lengths) > 70)) # Number of sequences of length greater than 70
maxlen = 70 # Maximum number of words to use in each question

train_data = pad_sequences(train_data, maxlen=maxlen) # Padding the lengths of sequences

output_test_data = pad_sequences(output_test_data, maxlen=maxlen) # Padding the lengths of the test_sequences for later
X_train, X_test, y_train, y_test = train_test_split(train_data, train_targets,

                                                    test_size = 0.1,

                                                    shuffle=False,

                                                    random_state = 42)
f = open(os.path.join('../input/embeddings/glove.840B.300d', 'glove.840B.300d.txt'))



embeddings_index = {}



for line in tqdm(f): # Wrapped this iterable with a tqdm to see the progress

    values = line.split(' ')

    word = values[0]

    embedding = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = embedding

f.close()
all_embs = np.stack(embeddings_index.values())

emb_mean = all_embs.mean()

emb_std = all_embs.std()



print(all_embs.shape) # Examine the dimensions of the vectors

embedding_dim = all_embs.shape[1] # Asign the vector dimension to a variable
embedding_matrix = np.random.normal(emb_mean,

                                    emb_std,

                                    (number_of_words, embedding_dim))
for word, i in word_dict.items():

    if i >= max_words:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
input_class = Input(shape=(maxlen,))

x = Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=False)(input_class)

x = Bidirectional(LSTM(64, return_sequences=True))(x)

x = Bidirectional(LSTM(32, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_class, outputs=x)



print(model.summary())
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])



history = model.fit(X_train, y_train,

                    epochs=3,

                    batch_size=256,

                    validation_data=(X_test, y_test))
pretrained_model_pred = model.predict([X_test], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_test, (pretrained_model_pred>thresh).astype(int))))
test_predictions = model.predict([output_test_data], batch_size=256, verbose=1)

test_predictions = (test_predictions>0.44).astype(int) # Using the best threshold from above

output = pd.DataFrame({"qid":test["qid"].values})

output['prediction'] = test_predictions

output.to_csv("submission.csv", index=False)