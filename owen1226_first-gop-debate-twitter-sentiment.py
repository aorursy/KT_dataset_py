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
import matplotlib.pyplot as plt

%matplotlib inline



import re

import math



import keras.layers as layers

from keras.models import Model, Sequential

from keras.initializers import glorot_uniform, he_uniform

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import to_categorical, layer_utils, plot_model
from sklearn.model_selection import train_test_split



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



def load_dataset(csv_dir, test_size=None):

    df = pd.read_csv(csv_dir)

    # Keeping only the neccessary columns

    df = df[['text','sentiment']]



#     df = df[df.sentiment != "Neutral"]

    

    # for idx, row in df.iterrows():

    #     row[0] = row[0].replace(row[0][row[0].find("RT ") : row[0].find(": ") + 2], '')

    df['text'] = [x.strip().replace(x[x.find("RT ") : x.find(": ") + 2], '') for x in df['text']]

    df['text'] = df['text'].apply(lambda x: re.sub('[^A-Za-z0-9 ,\?\'\"-._\+\!/\`@=;:]+', '', x.lower()))

    

#     df["sentiment"] = df.sentiment.astype('category')

#     classes = df.sentiment.cat.categories

    classes = ["Not Negative", "Negative"]

    

#     print("Negative: {}, Neutral: {}, Positive: {}".format(

#         df[df['sentiment'] == 'Negative'].size,

#         df[df['sentiment'] == 'Neutral'].size, 

#         df[df['sentiment'] == 'Positive'].size))



    print("Negative: {}, Not Negative: {}".format(

        df[df['sentiment'] == 'Negative'].size,

        df[df['sentiment'] != 'Negative'].size))



    tokenizer = Tokenizer(num_words = 10000, split = ' ')

    tokenizer.fit_on_texts(df['text'].values)

    

    X = tokenizer.texts_to_sequences(df['text'].values)

    X = pad_sequences(X, maxlen=32)

    Y = pd.get_dummies(df['sentiment']).values[:, 0]

    

#     for i in range(20):

#         print('{} - {}'.format(df['sentiment'].iloc[i], Y[i]))



    if test_size is None:

        return X, Y, tokenizer, classes

    else:

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size , random_state=1)

        return X_train, X_test, Y_train, Y_test, tokenizer, classes

    

    



X_train, X_test, Y_train, Y_test, tokenizer, CLASSES = load_dataset('../input/first-gop-debate-twitter-sentiment/Sentiment.csv', test_size=0.02)

print("Train shape: {}".format(X_train.shape))

print("Test shape: {}".format(X_test.shape))

print(X_test[: 3])

# print(tokenizer.word_index["the"])

print(CLASSES)
def load_vectors(file, total_num=0):

    with open(file, encoding='utf-8', mode = 'r') as f:

#         words = set()

        word_vec = {}

        i = 0

        for line in f:

            values = line.strip().split()

            curr_word = values[0]

#             words.add(curr_word)



            try:

                word_vec[curr_word] = np.array(values[1:], dtype=np.float64)



                i += 1

                if i % 1000 == 0:

                    print('Processed {0} of {1}'.format(i, total_num), end='\r')



            # except Exception as ex:

            #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"

            #     message = template.format(type(ex).__name__, ex.args)

            #     print(message)

            except ValueError: # For data errors

                # print("ValueError - ", curr_word)

                pass



        print('Processed {0} of {1}'.format(i, total_num))

    return word_vec



#         i = 1

#         words_to_index = {}

#         index_to_words = {}

#         for w in sorted(words):

#             words_to_index[w] = i

#             index_to_words[i] = w

#             i = i + 1

#     return words_to_index, index_to_words, word_to_vec_map





# word_vec = load_vectors('../input/glove6b50dtxt/glove.6B.50d.txt', 400000)

word_vec = load_vectors('../input/glove6b100dtxt/glove.6B.100d.txt', 400000)

print(word_vec["the"])
def pretrained_embedding_layer(word_vec, word_index):

    """

    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.



    Arguments:

    word_vec -- dictionary mapping words to their GloVe vector representation.

    word_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)



    Returns:

    embedding_layer -- pretrained layer Keras instance

    """



    vocab_len = len(word_index) + 1  # adding 1 to fit Keras embedding (requirement)

    emb_dim = word_vec["the"].shape[0]  # define dimensionality of your GloVe word vectors (= 50 or 100)



    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.

    embedding_layer = layers.Embedding(vocab_len, emb_dim, trainable=False)



    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".

    embedding_layer.build((None,))



    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)

    emb_matrix = np.zeros((vocab_len, emb_dim))



    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary

    for word, index in word_index.items():

#         emb_matrix[index, :] = word_vec[word]   # error happens when words are NOT in word_vec

        vec = word_vec.get(word)

        if vec is not None:

            emb_matrix[index, :] = vec

            

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.

    embedding_layer.set_weights([emb_matrix])



    return embedding_layer
def MyModel(input_shape, out_shape, word_vec, word_index):

    """

    Arguments:

    input_shape -- shape of the input, usually (max_len,)

    word_vec -- dictionary mapping every word in a vocabulary into its 50/100-dimensional vector representation

    word_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)



    Returns:

    model -- a model instance in Keras

    """



    # Define input sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).

    inputs = layers.Input(input_shape, dtype="int32")



    # Create the embedding layer pretrained with GloVe Vectors

    embedding_layer = pretrained_embedding_layer(word_vec, word_index)

    # Propagate input sentence_indices through your embedding layer, you get back the embeddings

    X = embedding_layer(inputs)

    

    # Add dropout with a probability

    X = layers.SpatialDropout1D(0.3)(X)

    

    # Propagate the embeddings through an LSTM layer with dimensional hidden state

    # The returned output is a batch of sequences.

    X = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(X)

    # Add dropout with a probability

    X = layers.Dropout(0.3)(X)

    

    # Propagate X trough another LSTM layer with dimensional hidden state

    # The returned output is a single hidden state, not a batch of sequences.

    X = layers.Bidirectional(layers.LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(X)

    

    # Add dropout with a probability

    X = layers.Dropout(0.3)(X)

    

    # Propagate X through a Dense layer with softmax activation to get back a batch of 1-dimensional vectors.

#     X = layers.Dense(out_shape)(X)

    # Add a sigmoid activation

#     outputs = Activation("sigmoid")(X)

    outputs = layers.Dense(out_shape, activation='sigmoid')(X)

    

#     model.add(Embedding(maxLen, embed_dim, input_length = X_input.shape[1]))

#     model.add(SpatialDropout1D(0.4))

#     model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))



    # Create Model instance which converts input sentence_indices into X.

    model = Model(inputs = inputs, outputs = outputs)



    return model
maxLen = len(max(X_train, key=len))



model = MyModel((maxLen,), 1, word_vec, tokenizer.word_index)

model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0003, decay=1e-6, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', verbose=1, save_best_only=True)   # Save the best model

hist = model.fit(X_train, Y_train, batch_size = 64, epochs = 50, verbose=1, callbacks=[monitor, checkpoint], validation_split=0.01, shuffle=True)
def plot_train_history(history):

    # plot the cost and accuracy 

    loss_list = history['loss']

    val_loss_list = history['val_loss']

    accuracy_list = history['acc']

    val_accuracy_list = history['val_acc']

    # epochs = range(len(loss_list))



    # plot the cost

    plt.plot(loss_list, 'b', label='Training cost')

    plt.plot(val_loss_list, 'r', label='Validation cost')

    plt.ylabel('cost')

    plt.xlabel('iterations')

    plt.title('Training and validation cost')

    plt.legend()

    

    plt.figure()

    

    # plot the accuracy

    plt.plot(accuracy_list, 'b', label='Training accuracy')

    plt.plot(val_accuracy_list, 'r', label='Validation accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('iterations')

    plt.title('Training and validation accuracy')

    plt.legend()





plot_train_history(hist.history)
score = model.evaluate(X_test, Y_test)



print ("Test Loss = " + str(score[0]))

print ("Test Accuracy = " + str(score[1]))
Y_test_pred = model.predict(X_test, batch_size=32, verbose=1)
from sklearn.metrics import roc_curve, auc



def calculate_optimal_threshold(Y, Y_pred):

    # ROC Curve

    fpr, tpr, thresholds = roc_curve(Y, Y_pred)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve')

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    plt.xlim([-0.025, 1.025])

    plt.ylim([-0.025, 1.025])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('RoC Curve')

    print("AUC: ", roc_auc)

    

    # Calculate the optimal threshold

    i = np.arange(len(tpr)) # index for df

    roc_df = pd.DataFrame({'threshold' : pd.Series(thresholds, index = i), 

                           'fpr': pd.Series(fpr, index=i), 

                           '1-fpr' : pd.Series(1-fpr, index = i), 

                           'tpr': pd.Series(tpr, index = i), 

                           'diff': pd.Series(tpr - (1-fpr), index = i) })

    opt_threshold = roc_df.iloc[roc_df['diff'].abs().argsort()[:1]]

    print(opt_threshold)

    

    return opt_threshold['threshold'].values[0]

    

    

threshold = calculate_optimal_threshold(Y_test, Y_test_pred)
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report



def analyze(Y, Y_pred, classes, activation="softmax", threshold=None):

    if activation == "sigmoid":

        Y_cls = Y

        Y_pred_cls = (Y_pred > threshold).astype(float)

    elif activation == "softmax":

        Y_cls = np.argmax(Y, axis=1)

        Y_pred_cls = np.argmax(Y_pred, axis=1)

    

    

    # Accuracy Score

    accuracy = accuracy_score(Y_cls, Y_pred_cls)

    print("Accuracy Score: {}\n".format(accuracy))

    

    

    # RMSE Score

    rmse = np.sqrt(mean_squared_error(Y, Y_pred))

    print("RMSE Score: {}\n".format(rmse))



    

    # Confusion Matrix

    print("Confusion Matrix:")

    cm = confusion_matrix(Y_cls, Y_pred_cls)

    print(cm)

    # Plot the confusion matrix as an image.

    plt.matshow(cm)

    # Make various adjustments to the plot.

    num_classes = len(classes)

    plt.colorbar()

    tick_marks = np.arange(num_classes)

    plt.xticks(tick_marks, range(num_classes))

    plt.yticks(tick_marks, range(num_classes))

    plt.xlabel('Predicted')

    plt.ylabel('True')

    

    

    # Classification Report

    print("Classification Report:")

    print(classification_report(Y_cls, Y_pred_cls, target_names=classes))







analyze(Y_test, Y_test_pred, CLASSES, "sigmoid", threshold)
def plot_mislabeled(X, Y, Y_pred, classes, activation="softmax", threshold=None, num_plot=0):

    """

    Plots images where predictions and truth were different.

    

    X -- original image data - shape(m, img_rows*img_cols)

    Y -- true labels - eg. [2,3,4,3,1,1]

    Y_pred -- predictions - eg. [2,3,4,3,1,2]

    """

    

    if activation == "sigmoid":

        Y_pred_cls = (np.squeeze(Y_pred) > threshold).astype(float)

    elif activation == "softmax":

        Y_pred_cls = np.argmax(Y_pred, axis=1)

    

    mislabeled_indices = np.where(Y != Y_pred_cls)[0]

    if num_plot < 1:

        num_plot = len(mislabeled_indices)

        

    for i, index in enumerate(mislabeled_indices[: num_plot]):

        sentence = []

        for id in X_test[index]:

            if id > 0:

                sentence += [k for k,v in tokenizer.word_index.items() if v == id]

#                 print(list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(id)])

        print(" ".join(sentence))

        print("Prediction: {} - {}\nClass: {}\n".format(classes[int(Y_pred_cls[index])], Y_pred[index], classes[int(Y[index])]))





        

plot_mislabeled(X_test, Y_test, Y_test_pred, CLASSES, "sigmoid", threshold, 5)
twt = ['He said Make America Great Again']



#vectorizing the tweet by the pre-fitted tokenizer instance

X_twt = tokenizer.texts_to_sequences(twt)

print(X_twt)



#padding the tweet to have exactly the same shape as 'embedding_2' input

X_twt = pad_sequences(X_twt, maxlen=32, dtype='int32', value=0)

print(X_twt)



Y_twt = model.predict(X_twt, verbose=1)

print(Y_twt)



if(Y_twt > threshold):

    print("negative")

else:

    print("positive")