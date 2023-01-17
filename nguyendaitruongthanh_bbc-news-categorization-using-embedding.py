import csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





from tensorflow import keras

from keras import layers

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
# read the csv file



df = pd.read_csv("../input/bbcnewsarchive/bbc-news-data.csv", sep="\t")



print(df)



df["category"].hist()
# shuffle the dataframe to evenly distribute the labels



df = df.sample(frac=1).reset_index(drop=True)

df
content = []

labels = []



for label in df.category:

    labels.append(label)

    

for con in df.content:

    for word in stopwords:

        token = " " + word + " "

        con = con.replace(token, " ")

        con = con.replace(" ", " ")

    content.append(con)



print(len(content))

print(len(labels))

print("\nContent:", content[0])

print("\nLabel:", labels[0])
# split the dataset into training set and test set



train_content, test_content = content[:1900], content[1900:]

train_labels, test_labels = labels[:1900], labels[1900:]



train_content = np.array(train_content)

test_content = np.array(test_content)



train_labels = np.array(train_labels)

test_labels = np.array(test_labels)



print(len(train_content))

print(len(train_labels))

print(len(test_content))

print(len(test_labels))
# check the distribution of labels in the training set and test set



unique_train_content, number_train_content = np.unique(train_labels, return_counts=True)



print("Training set labels:")

print(unique_train_content)

print(number_train_content)



unique_test_content, number_test_content = np.unique(test_labels, return_counts=True)



print("\nTest set labels:")

print(unique_test_content)

print(number_test_content)
# tokenize the content



vocab_size = 10000

embedding_dim = 32

max_len = 200

trunc_type = "post"

oov_tok = "<OOV>"



tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_content)



word_index = tokenizer.word_index



sequences = tokenizer.texts_to_sequences(train_content)

padded = pad_sequences(sequences, maxlen=max_len, truncating=trunc_type)



test_sequences = tokenizer.texts_to_sequences(test_content)

test_padded = pad_sequences(test_sequences, maxlen=max_len, truncating=trunc_type)



print(test_padded.shape)
# tokenize the labels



label_tokenizer = Tokenizer()

label_tokenizer.fit_on_texts(labels)



label_index = label_tokenizer.word_index



label_sequences = np.array(label_tokenizer.texts_to_sequences(train_labels))



test_label_sequences = np.array(label_tokenizer.texts_to_sequences(test_labels))



print(label_sequences.shape)

print(test_label_sequences.shape)
# define an NN model



model = keras.Sequential([layers.Embedding(vocab_size, embedding_dim, input_length=max_len),

                        

                         layers.GlobalAveragePooling1D(), #simpler and faster than Flatten()

                         layers.Dense(128, activation="relu"),

                         layers.Dense(6, activation="softmax")])



model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



model.summary()

keras.utils.plot_model(model)
# train the model



num_epochs = 10



history = model.fit(padded,

                   label_sequences,

                   epochs=num_epochs,

                   validation_data = (test_padded, test_label_sequences))
# plot accuracy and loss



acc = history.history["accuracy"]

val_acc = history.history["val_accuracy"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]



epochs = range(1, len(acc) + 1)



# accuracy



plt.plot(epochs, acc, "b", label="Training accuracy")

plt.plot(epochs, val_acc, "b--", label="Validation accuracy")

plt.title("Training and validation accuracy")

plt.legend()

plt.show()



# loss



plt.plot(epochs, loss, "r", label="Training loss")

plt.plot(epochs, val_loss, "r--", label="Validation loss")

plt.title("Training and validation loss")

plt.legend()

plt.show()