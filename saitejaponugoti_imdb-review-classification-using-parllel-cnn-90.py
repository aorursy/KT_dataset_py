# import required packages

import os

import re

import tarfile

import numpy as np

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.utils import get_file

from bs4 import BeautifulSoup

from tensorflow import keras

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, Model

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

from tensorflow.keras.preprocessing.sequence import pad_sequences
# function to download data

def download_data():

    data_dir = get_file('aclImdb_v1.tar.gz', 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',

                        cache_subdir="datasets", hash_algorithm="auto", extract=True, archive_format="auto")

    my_tar = tarfile.open(data_dir)

    my_tar.extractall('./data/')  # specify which folder to extract to

    my_tar.close()
# function to remove html tags from the reviews

def remove_html_tags(data_raw):

    pro_data = []

    for i in range(len(data_raw)):

        soup = BeautifulSoup(data_raw[i], "html.parser")

        pro_data.append(soup.get_text())



    return pro_data

# function to remove special charecters from the reviews

def remove_special_char(data_raw):

    pro_data = []

    for i in range(len(data_raw)):

        review = re.sub('\[[^]]*\]', ' ', data_raw[i])

        review = re.sub('[^a-zA-Z]', ' ', data_raw[i])

        pro_data.append(review)



    return pro_data
# function to remove stop words

def remove_stop_words(data_raw):

    pro_data = []

    for i in range(len(data_raw)):

        text =  ' '.join([word for word in 

        data_raw[i].split() if word.lower() not in my_stop_words])

        pro_data.append(text)



    return pro_data
# function to pad the sequences to constant length

def pad_sentence(data_raw, sentence_length):   

    return pad_sequences(data_raw, maxlen=sentence_length, dtype='int32', 

    padding='post', truncating='post')
# function to tokenize the sentences after removing html,secail cahrecters and stop words

def tokenize_sentences(train, test, vocab_size, sentence_length):

    train = remove_stop_words(train)

    test = remove_stop_words(test)

    tokenizer = Tokenizer(num_words=vocab_size, lower=True, oov_token="<OOV>")

    tokenizer.fit_on_texts(train)



    train = tokenizer.texts_to_sequences(train)

    test = tokenizer.texts_to_sequences(test)



    train = pad_sentence(train, sentence_length)

    test = pad_sentence(test, sentence_length)

    print(train.shape)

    return train, test
# function to perform all data preprocessing techniques and return the processed data

def get_data_preprocessed(train_data_dir, test_data_dir, vocab_size=6000, max_seq_length=600):

    # dictionary mapping label name to numeric id

    labels_index = {'pos': 1, 'neg': 0}



    # reading train directory

    train_texts = []  # list of text samples

    train_labels = []  # list of label ids

    for name in ["pos", "neg"]:

        if name == "pos" or "neg":

            path = os.path.join(train_data_dir, name)

            if os.path.isdir(path):

                label_id = labels_index[name]

                for fname in sorted(os.listdir(path)):

                    fpath = os.path.join(path, fname)

                    text = open(fpath).read()

                    train_texts.append(text)

                    train_labels.append(label_id)



    # reading test directory

    test_texts = []  # list of text samples

    test_labels = []  # list of label ids

    for name in ["pos", "neg"]:

        if name == "pos" or "neg":

            path = os.path.join(test_data_dir, name)

            if os.path.isdir(path):

                label_id = labels_index[name]

                for fname in sorted(os.listdir(path)):

                    fpath = os.path.join(path, fname)

                    text = open(fpath).read()

                    test_texts.append(text)

                    test_labels.append(label_id)



    print("Data loading from directory finished, proceeding to pre-processing")

    train_texts = remove_html_tags(train_texts)

    train_texts = remove_special_char(train_texts)

    print("Removed html tags from the data")

    test_texts = remove_html_tags(test_texts)

    test_texts = remove_special_char(test_texts)

    print("Removed special charecters from the data")



    train_texts, test_texts = tokenize_sentences(train_texts, test_texts, vocab_size, max_seq_length)

    print(train_texts.shape)

    print("Tokenized and padded the sequences to constant length")



    return np.array(train_texts), np.array(np.eye(2)[train_labels]),np.array(test_texts), np.array(np.eye(2)[test_labels])
def create_model(emb_dim, num_words, sentence_length, hid_dim, class_dim, dropout_rate):

    input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)



    layer = tf.keras.layers.Embedding(num_words, output_dim=emb_dim)(input_layer)



    layer_conv3 = tf.keras.layers.Conv1D(hid_dim, 3, activation="relu")(layer)

    layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)



    layer_conv4 = tf.keras.layers.Conv1D(hid_dim, 2, activation="relu")(layer)

    layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)



    layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3], axis=1)

    layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.keras.layers.Dropout(dropout_rate)(layer)



    output = tf.keras.layers.Dense(class_dim, activation="softmax")(layer)



    model = Model(inputs=input_layer, outputs=output)



    return model
# function to preprocess data set, create ,compile and train model

def run_model(x_train, y_train, emb_dim, hid_dim, batch_size, epochs, model_save_dir, num_of_classes, vocab_size,max_seq_length):

    # creating a model with required parameters

    model = create_model(emb_dim, vocab_size, max_seq_length, hid_dim, 2, DROPOUT_RATE)

    model.summary()



    # compiling the model

    model.compile(loss="categorical_crossentropy",

                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),

                  metrics=["accuracy"])



    # declaring path to save models

    if not os.path.exists(model_save_dir):

        os.makedirs(model_save_dir)



    filepath = model_save_dir + "/model-{epoch:02d}.hdf5"



    # declaring call back function

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy',

                                                             verbose=1, save_best_only=True,

                                                             save_weights_only=True, mode='auto')



    # training the model with a validation split of 0.2

    history = model.fit(x_train, y_train, batch_size=batch_size,

                        validation_split=0.2, epochs=epochs, callbacks=[checkpoint_callback], verbose=1)





    return history,model
# function to plot accuracy comparision graphs

def plot_graph_accuracy(hist, title_string):

    plt.figure(figsize=(12, 8))

    plt.plot(hist.history['accuracy'])

    plt.plot(hist.history['val_accuracy'])

    plt.title(title_string)

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train_accuracy', 'val_accuracy'], loc='lower right')

    plt.show()





# function to plot accuracy comparision graphs

def plot_graph_loss(hist, title_string):

    plt.figure(figsize=(12, 8))

    plt.plot(hist.history['loss'])

    plt.plot(hist.history['val_loss'])

    plt.title(title_string)

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train_loss', 'val_loss'], loc='lower right')

    plt.show()
TRAIN_DATA_DIR = "./data/aclImdb/train"  # source: http://ai.stanford.edu/~amaas/data/sentiment/

TEST_DATA_DIR = "./data/aclImdb/test"

labels_index = {'pos': 1, 'neg': 0}



#declaring a custom stop words list to be removed from the reviews

my_stop_words = ['the', 'and', 'a', 'of', 'to', 'is', 'it', 'in', 'i', 'this', 'that', 's', 'was', 'as', 'for', 'with']



#declaring drop out rate

DROPOUT_RATE = 0.9
if __name__ == "__main__":

    vocab_size = 10000

    max_seq_length = 1000

    # 1. load your training data

    # expecting that the dat is already available , so commenting the function

    download_data()

    print("downloaded data")



    # loading the train data and getting the data after pre-processing

    # creating data sets

    x_train, y_train, x_text, y_text = get_data_preprocessed(TRAIN_DATA_DIR, TEST_DATA_DIR,vocab_size, max_seq_length)

    print(x_train.shape)

    print("train data preprocessed 1")

    x_train, y_train = shuffle(x_train, y_train)

    print("train data preprocessed 2")



    # 2. Train your network

    # Make sure to print your training loss and accuracy within training to show progress

    history, model = run_model(x_train, y_train, 500, 1024, 512, 2, "./data/", 2, 10000, 1000)

    

       # print accuracy and loss graphs

    # Make sure you print the final training accuracy

    print("final training accuracy: ",history.history['accuracy'][-1])

    print("final training loss: ",history.history['loss'][-1])

    

    score = model.evaluate(x_test, y_test, batch_size=16)

    print("test_accuracy:", score)



    # print accuracy and loss graphs

    plot_graph_accuracy(history, "Accuracy")

    plot_graph_loss(history, "Loss")



    # 3. Save your model

    model.save("20841154_NLP_model.h5")