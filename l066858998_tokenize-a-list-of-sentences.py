import tensorflow as tf

import tensorflow.keras as keras
sentences = ["Hello, my name is Johnny.",

             "Learning deep learning is a trend.",

             "Deep learning is a key to the era of AI."]
# create tokenizer object

tokenizer = keras.preprocessing.text.Tokenizer(num_words=100,

                                   oov_token="<OOV Token>")
# fit tokenizer on text

tokenizer.fit_on_texts(sentences)
# show the index of words which are tokenize

tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences=sequences,

                                                              padding="post")
padded_sequences