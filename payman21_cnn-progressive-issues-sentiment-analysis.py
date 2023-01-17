import pandas as pd

import re

import numpy as np

import tensorflow as tf

print(tf.__version__)

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
data = pd.read_csv("../input/progressive-tweet-sentiment.csv", encoding="ISO-8859-1")
data.columns
d2 = data.dropna(subset=['q1_from_reading_the_tweet_which_of_the_options_below_is_most_likely_to_be_true_about_the_stance_or_outlook_of_the_tweeter_towards_the_target'])
d2['label'] = np.nan
lbl_lst = []

for i in range(len(d2)):

    item =  d2.iloc[i]['q1_from_reading_the_tweet_which_of_the_options_below_is_most_likely_to_be_true_about_the_stance_or_outlook_of_the_tweeter_towards_the_target']

    splt = item.split(":")[0]

    lbl_lst.append(splt)

d2['label'] =  lbl_lst
d3 = d2[d2['label'] != 'NONE OF THE ABOVE']
labels = d3['label']
bin_lst = []

for label in labels:

    if label == "AGAINST":

        y = 0

        bin_lst.append(y)

    else:

        y = 1

        bin_lst.append(y)

d3['bin_label'] =  bin_lst
df = d3[['bin_label', 'tweet']].copy()

df = df.reset_index(drop=True)
df.head()
X = df['tweet']

y = df['bin_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
training_labels_final = np.array(y_train)

testing_labels_final = np.array(y_test)

training_sentences = X_train.values.tolist()

testing_sentences = X_test.values.tolist()
vocab_size = 10000

embedding_dim = 32

max_length = 20

trunc_type='post'

oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)



testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_review(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])



print(decode_review(padded[6]))

print(training_sentences[6])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Conv1D(128, 5, activation='relu'),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(6, activation='relu'),

    

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

num_epochs = 10

model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
# e = model.layers[0]

# weights = e.get_weights()[0]

# print(weights.shape) # shape: (vocab_size, embedding_dim)
results = model.evaluate(testing_padded, testing_labels_final)

print('test loss, test acc:', results)
testing_labels_final[3:6]
testing_sentences[3:6]
model.predict(testing_padded[3:6])
# import io



# out_v = io.open('./tmp/vecs_tweet.tsv', 'w', encoding='utf-8')

# out_m = io.open('./tmp/meta.tsv', 'w', encoding='utf-8')

# for word_num in range(1, vocab_size):

#   word = reverse_word_index[word_num]

#   embeddings = weights[word_num]

#   out_m.write(word + "\n")

#   out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

# out_v.close()

# out_m.close()