import re, os, time, math, operator

import pandas as pd

import numpy as np

import nltk



from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras import layers # Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

                         # Bidirectional, GlobalMaxPool1D

from keras.layers import CuDNNGRU, LSTM               

from keras.models import Model, Sequential

from keras import initializers, regularizers, constraints, optimizers, layers

# Data filenames to use

EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

GLOVE_EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

WIKI_EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

PARAGRAM_EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

GOOGLE_EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

TRAINING_FILE = '../input/train.csv'

TESTING_FILE = '../input/test.csv'

embed_file_dict = { 'glove' : GLOVE_EMBEDDING_FILE, \

                    'wiki' : WIKI_EMBEDDING_FILE, \

                    'paragram' : PARAGRAM_EMBEDDING_FILE, \

                    'google' : GOOGLE_EMBEDDING_FILE }



# global configuration parameters

embed_size = 300      # default size of word vector



# parameters for neural net configuration (bigger #s mean more params to train!)

max_features = 10000  # number of unique words (features/columns) to use 

maxlen = 72           # max number of words in a question to use

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

%matplotlib inline

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

# and https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=80, figure_size=(12.0,10.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    #more_stopwords = 'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    #stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  



# look at sincere vs insincere questions separately

train_df = pd.read_csv(TRAINING_FILE)

test_df = pd.read_csv(TESTING_FILE)

dfs = train_df[train_df["target"] == 0]

dfi = train_df[train_df["target"] == 1]

plot_wordcloud(dfi["question_text"], title="word cloud insincere questions")

plot_wordcloud(dfs["question_text"], title="word cloud sincere questions")
print("Total questions in training dataset:",train_df.shape[0])

print("Percentage of insincere questions: {0:.2f}%".format(dfi.shape[0]/train_df.shape[0]))

print("Total questions in testing dataset:",test_df.shape[0])



def getQuestionLengths(df,colname = "question_text"):

    q_lengths = []

    for index, row in df.iterrows():

        toks = row[colname].split()

        q_lengths.append(len(toks))

    return q_lengths





# decimate training data to do more analysis

ntrain = 2000

train_df = train_df.loc[np.random.choice(train_df.index, ntrain, replace=False)]

dfs = train_df[train_df["target"] == 0]

dfi = train_df[train_df["target"] == 1]

qleni = getQuestionLengths(dfi)

qlens = getQuestionLengths(dfs)

avgleni = np.mean(qleni)

avglens = np.mean(qlens)



print("Average length of sincere question: {0:.2f}".format(avglens))

print("Average length of insincere question: {0:.2f}".format(avgleni))



print("\nSome sincere questions:")

print(dfs["question_text"].head().values)



print("\nSome insincere questions:")

print(dfi["question_text"].head().values)



plt.hist(qlens,density=True,alpha = .5,label = "sincere")

plt.hist(qleni,density=True,alpha = .5,label = "insincere")

plt.title("Length of questions")

plt.legend()

# Set of utility functions for tracking time and plotting history

def PrintVars(num_epochs):

    print("Launching with", max_features, "Max Feature words")

    print("Launching with", maxlen, "Max words per question")

    print("Launching with", num_epochs, "Epoches")

    



def StartTime():

    currentDT = datetime.datetime.now()

    stime = currentDT.strftime("%H:%M:%S")

    print("-- Start Time: ", stime)

    return currentDT



def EndTime(startTime):

    currentDT = datetime.datetime.now()

    durTime = currentDT - startTime

    etime = currentDT.strftime("%H:%M:%S")

    print("-- End Time: ", etime)

    secs = durTime.total_seconds()    

    print("Total Runtime in seconds = ", secs)

    print("Total Runtime in minutes = ", secs / 60)



# Import Data - splitting data into training and validation data, valFrac is the percentage of the data to be split off for validation purposes

def ImportData(valFrac = 0.1):

    print("Loading training dataset from file:", TRAINING_FILE)

    train_df = pd.read_csv(TRAINING_FILE)

    train_y = train_df['target']

    train_df = train_df.drop('target', axis=1)

    

    # split into train and validation dataset

    train_df, val_df, train_y, val_y = train_test_split(train_df, train_y, test_size=valFrac, random_state=2018)

    

    print("Loading testing dataset")

    test_df = pd.read_csv(TESTING_FILE)

    print("Datasets loaded")

    print("   Train shape: ",train_df.shape)

    print("   Val shape: ", val_df.shape)

    print("   Test shape: ",test_df.shape)

    print("Sample of data:\n", train_df.head())

    

    return train_df, train_y, val_df, val_y, test_df 
# This function will return the tokenizer with the requested number of most frequent words (since keras doesn't do this correctly...)

# Adapted from https://github.com/keras-team/keras/issues/8092#issuecomment-466909653

def GetNumWords(tk, num_words):

    sorted_by_word_count = sorted(tk.word_counts.items(), key=lambda kv: kv[1], reverse=True)

    tk.word_index = {}

    i = 0

    for word,count in sorted_by_word_count:

        if i == num_words:

            break

        tk.word_index[word] = i + 1    # <= because tokenizer is 1 indexed

        i += 1

    return tk



# This function strips the stopwords out of the questions and converts them to lower-case

def get_cleaned_words_from_df(df):

    """

    turn dataframe 'question_text' column into list of words 

    (with question marks, capitalizations and stopwords removed)

    """

    stopwords = set(nltk.corpus.stopwords.words('english'))

    y = list(df["question_text"].values)

    words = []

    for entry in y:

        words += entry.split()

    words = [w.replace("?","") for w in words] # remove question marks

    words = [w.lower() for w in words if w.lower() not in stopwords] # remove "stopwords"

    return words



# This function tokenizes a data set and zero pads each question, to give them all a constant length (~80 words)

def preprocess_set(data, tokenizer, maxlength):

    data_out = tokenizer.texts_to_sequences(data['question_text'])

    data_out = pad_sequences(data_out, maxlen=maxlength, padding='post' )

    return data_out, tokenizer



# This function adapted from:

# https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn

# This function creates the embedding matrix from the GloVe embedding matrix.  This function searches the embedding rows for 

# only the vocab words in the word_index.  

def create_embedding_matrix(filepath, word_index, embedding_dim):

    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index

    embedding_matrix = np.zeros((vocab_size, embedding_dim))



    print("Creating embedding matrix, with word index of length", len(word_index), "and vocab size", vocab_size)

    emb_df = pd.read_csv(filepath, sep=' ')    

    print("Finished reading embedding file in from csv, generating the embedding matrix...")

    

    cnt = 0

    num_rows = emb_df.shape[0]

    spn = 0

    for i in range(num_rows - 1, -1, -1 ):

        row = emb_df.iloc[i]

        word = str(row[0])

        w = word.lower()

        if w in word_index:

            idx = word_index[w]

            embedding_matrix[idx] = np.array(row[1:], dtype=np.float32)[:embedding_dim]

        else:

            cnt += 1

        

        spn += 1

        if (spn == 50):

            emb_df = emb_df[:-50]

            spn = 0

    print("Pruned", cnt, "embedding strings...")

    print("Final embedding matrix shape: ", embedding_matrix.shape)

    return embedding_matrix, vocab_size



# This is the main function for Preprocessing the X data sets, to prepare them for the model

# Enhanced from https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings

def PreprocessData(train_X, val_X, test_X):

    print("Starting to PreprocessData ...")      

    # tokenize the questions

    # https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do

    # tokenizer.texts_to_sequences Transforms each text in texts to a sequence of integers. (takes each word in the text and replaces it corresponding integer value from the word_index dictionary_.

    tokenizer = Tokenizer(num_words=max_features)

    

    tokenizer.fit_on_texts(get_cleaned_words_from_df(train_X))

    print("Tokenizer word_index len (train)", len(tokenizer.word_index))

    tokenizer.fit_on_texts(get_cleaned_words_from_df(val_X))

    print("Tokenizer word_index len (valid)", len(tokenizer.word_index))

    tokenizer.fit_on_texts(get_cleaned_words_from_df(test_X))

    print("Tokenizer word_index len (test)", len(tokenizer.word_index))

    

    print("Max Features: ", max_features) 

    # There is a bug in the tokenizer that doesn't limit the number of words correctly....

    tokenizer = GetNumWords(tokenizer, max_features)

    print("Tokenizer word_index len (final)", len(tokenizer.word_index))

    

    train_X, tokenizer = preprocess_set(train_X, tokenizer, maxlen)

    val_X, tokenizer = preprocess_set(val_X, tokenizer, maxlen)

    test_X, tokenizer = preprocess_set(test_X, tokenizer, maxlen)

    

    return train_X, val_X, test_X, tokenizer
# What do the tokenized, padded datasets look like?



train_df, train_y, val_df, val_y, test_df = ImportData() 

train_X, val_X, test_X, tokenizer = PreprocessData(train_df, val_df, test_df)
print("Original training data as read in as pandas DataFrame: ",type(train_df))

print("Total # of training entries:",train_df.shape)

n = train_df.shape[0]

print("Processed training data converted to:",type(train_X))

print("Entries split into train and validation set: ",train_X.shape,val_X.shape)

print("Example of what the processed data looks like:")

train_X

embedding_matrix, vocab_size = create_embedding_matrix('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', 

                                           tokenizer.word_index, embed_size)
def createModel(vocab_size, embedding_matrix):

    print("Setting up new model to use the embedding matrix")

    model = Sequential()

    model.add(layers.Embedding(vocab_size, embed_size, 

                               weights=[embedding_matrix], 

                               input_length=max_features, 

                               trainable=True))

                               #mask_zero=True))

    model.add(layers.GlobalMaxPool1D())

    model.add(layers.Dense(10, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    print("Compiling model")

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    print(model.summary())

    return model



def createBetterModel(vocab_size, embedding_matrix):

    inLen = min(vocab_size, max_features)

    print("Vocab_size:", vocab_size, "Max_feature:", max_features, "inLen:", inLen)

    print("Setting up new model to use the embedding matrix")

    model = Sequential()

    model.add(layers.Embedding(vocab_size, embed_size, 

                               weights=[embedding_matrix], 

                               input_length=maxlen, 

                               trainable=True))

                               #mask_zero=True))

    #model.add(layers.GlobalMaxPool1D())

    model.add(layers.Bidirectional(LSTM(10)))

    model.add(layers.Dense(16, activation="relu"))

    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(1, activation="sigmoid"))

    print("Compiling model...")

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    print(model.summary())

    return model
model = createBetterModel(vocab_size, embedding_matrix)
# This plot function comes from 

# https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn

def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.xlabel("# of Epochs")

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.xlabel("# of Epochs")

    plt.legend()



# Train the given model on the data, and plot accuacy vs loss

def PerformFit(model, X_train, y_train, X_test, y_test, num_epochs, bat_size ):

    history = model.fit(X_train, y_train,

                        epochs=num_epochs,

                        verbose=False,

                        validation_data=(X_test, y_test),

                        batch_size=bat_size)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

    print("Training Accuracy: {:.4f}".format(accuracy))

    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)

    return model
# Train the model on the training X,y

num_epochs = 6

model = PerformFit(model, train_X, train_y, val_X, val_y, num_epochs, bat_size=512)
# This function evaluates the predictions with the validation answers, to determine where is the appropriate threshold

# to split the soft scores into hard predictions

def DetermineThreshold(val_y, pred_y):

    print("Checking Prediction against various thresholds...")

    threshold = 0.1

    best_f1score = 0

    best_threshold = threshold

    for thresh in np.arange(0.1, 0.501, 0.01):

        thresh = np.round(thresh, 2)

        f1score = metrics.f1_score(val_y, (pred_y>thresh).astype(int))

        print("F1 score at threshold {0} is {1}\n".format(thresh, f1score))

        if f1score > best_f1score:

            best_f1score = f1score

            best_threshold = thresh

    return best_threshold



# This function writes the submission file to disk to be available for scoring

def WriteSubmissionFile(test_df, y_pred):

    print("Writing submission to file...")

    qid = test_df["qid"]

    sub = pd.DataFrame()

    sub["qid"] = qid

    sub["prediction"] = y_pred

    sub.to_csv("submission.csv", index=False)



    print("Done writing submission to file")
print("make predictions from validation dataset")

pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)



thresh = DetermineThreshold(val_y, pred_val_y)



print("make predictions from test dataset")

pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)



print("Applying threshold of {0} to predictions ...".format(thresh))

y_pred = (pred_test_y>thresh).astype(int)



WriteSubmissionFile(test_df, y_pred)

    