import pandas as pd



import matplotlib

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import nltk
df = pd.read_csv("../input/train.csv")



df_test = pd.read_csv("../input/test.csv")
df.head(10)
df.isna().sum()
df_test.head(10)
df_test.isna().sum()
def tokenize_the_text(phrases):

    

    from nltk.tokenize import word_tokenize

    from nltk.text import Text

    

    tokens = [word for word in phrases]

    tokens = [word.lower() for word in tokens]

    tokens = [word_tokenize(word) for word in tokens]

    

    return tokens



#crude_tokens = tokenize_the_text(df.question_text)

#print(crude_tokens[0:10])
def create_a_vocab(tokens):

    

    vocab = set()



    for sentence in tokens:

        for word in sentence:

            vocab.add(word)



    vocab = list(vocab)



    return vocab

    

#vocab = create_a_vocab(crude_tokens)



vocab = create_a_vocab(tokenize_the_text(df.question_text))



print("Vocabulary size:", len(vocab), "words")
def removing_stopwords(tokens_custom_cleaned):



    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')

    tokens_custom_cleaned_and_without_stopwords = []

    for sentence in tokens_custom_cleaned:

        tokens_custom_cleaned_and_without_stopwords.append([word for word in sentence if word not in stop_words])

        

    return tokens_custom_cleaned_and_without_stopwords



tokens_without_stopwords = removing_stopwords(tokenize_the_text(df.question_text))
vocab = create_a_vocab(tokens_without_stopwords)



print("Vocabulary size after removing stopwords:", len(vocab), "words")
def lemmatizing_the_tokens(tokens_custom_cleaned_and_without_stopwords):



    from nltk.stem.wordnet import WordNetLemmatizer 

    lem = WordNetLemmatizer()



    tokens_custom_cleaned_and_without_stopwords_and_lemmatized = []



    for sentence in tokens_custom_cleaned_and_without_stopwords:

        tokens_custom_cleaned_and_without_stopwords_and_lemmatized.append([lem.lemmatize(word, pos='v') for word in sentence])

        

    return tokens_custom_cleaned_and_without_stopwords_and_lemmatized





tokens_without_stopwords_and_lemmatized = lemmatizing_the_tokens(tokens_without_stopwords)
vocab = create_a_vocab(tokens_without_stopwords_and_lemmatized)



print("Vocabulary size after removing stopwords and lemmatizing the text:", len(vocab), "words")
### do the same and for the testSet





tokens_without_stopwords_test = removing_stopwords(tokenize_the_text(df_test.question_text))

tokens_without_stopwords_and_lemmatized_test = lemmatizing_the_tokens(tokens_without_stopwords_test)



vocab_test = create_a_vocab(tokens_without_stopwords_and_lemmatized_test)



print("Vocabulary size after removing stopwords and lemmatizing the text:", len(vocab_test), "words")
del tokens_without_stopwords_test

del tokens_without_stopwords

del vocab

del vocab_test
# Keras Libraries

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, SpatialDropout1D, Bidirectional, Activation,GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.models import load_model

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
def get_embeddings_dict():



    import numpy as np



    filename = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'



    glove_w2v_embeddings_index = dict()

    f = open(filename, "r", encoding='utf-8')

    for line in f:

        values = line.split(' ')

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        glove_w2v_embeddings_index[word] = coefs

    f.close()

    

    return glove_w2v_embeddings_index





glove_w2v_embeddings_index = get_embeddings_dict()
###

#tokens_uncleaned = tokenize_the_text(df.question_text.values)

#tokens_uncleaned_test = tokenize_the_text(df_test.question_text.values)

#vocab_all_corpus = create_a_vocab(tokenize_the_text(df.question_text.values) + tokenize_the_text(df_test.question_text.values))



#sentences = [' '.join(sent) for sent in tokens_uncleaned]

#sentences_test = [' '.join(sent) for sent in tokens_uncleaned_test]





all_corpus = [' '.join(sent) for sent in tokens_without_stopwords_and_lemmatized] + [' '.join(sent) for sent in tokens_without_stopwords_and_lemmatized_test]

max_len = max([len(elem.split()) for elem in all_corpus])

#print(max_len)

###





tokenizer = Tokenizer(lower=True, filters='')

tokenizer.fit_on_texts(all_corpus)



vocabulary_size = len(tokenizer.word_index) + 1

#print(vocabulary_size)







X = tokenizer.texts_to_sequences([' '.join(sent) for sent in tokens_without_stopwords_and_lemmatized])

y = df.target.values

X = pad_sequences(X, maxlen=max_len)





xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3, shuffle=True)
del X

del y

del tokens_without_stopwords_and_lemmatized

del all_corpus
import textblob



embedding_dim = 300



def get_embedding_matrix():

    

    embedding_matrix = np.zeros((vocabulary_size, embedding_dim + 2))



    for word, i in tokenizer.word_index.items():

        if i > vocabulary_size:

            continue

        embedding_vector = glove_w2v_embeddings_index.get(word)

        word_sentiment = textblob.TextBlob(word).sentiment

        if embedding_vector is not None:

            embedding_matrix[i] = np.append(embedding_vector, [word_sentiment.polarity, word_sentiment.subjectivity])

        else:

            embedding_matrix[i, -2:] = [word_sentiment.polarity, word_sentiment.subjectivity]

            

    return embedding_matrix



embedding_matrix = get_embedding_matrix()
del glove_w2v_embeddings_index
def plot_history(history):

    acc = history.history['categorical_accuracy']

    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
def build_dl_lstm_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs, filename):



    ## Network architecture

    from keras.utils import to_categorical

    from keras.callbacks import ModelCheckpoint

    from keras.callbacks import EarlyStopping

    from keras.layers import Masking

    from keras.initializers import Constant

    import time

    

    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    

    start_time = time.time()

    

    

    from numpy.random import seed

    seed(42)

    from tensorflow import set_random_seed

    set_random_seed(42)



    embedding_size= embedding_dim + 2

    batch_size = 512

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_size, input_length = max_len, 

                        weights=[embedding_matrix], trainable = False, mask_zero=True))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(LSTM(int((embedding_size/2)-50), recurrent_dropout=dropouts, dropout=dropouts, return_sequences=False))

    

    model.add(Dense(2, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop



    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=1, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time
def build_dl_lstm_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs, filename):



    ## Network architecture

    from keras.utils import to_categorical

    from keras.callbacks import ModelCheckpoint

    from keras.callbacks import EarlyStopping

    from keras.layers import Masking

    from keras.initializers import Constant

    import time

    

    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    

    start_time = time.time()

    

    

    from numpy.random import seed

    seed(42)

    from tensorflow import set_random_seed

    set_random_seed(42)



    embedding_size= embedding_dim + 2

    batch_size = 512

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_size, input_length = max_len, 

                        weights=[embedding_matrix], trainable = False, mask_zero=True))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(GRU(int((embedding_size/2)-50), recurrent_dropout=dropouts, dropout=dropouts, return_sequences=False))

    

    model.add(Dense(2, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop



    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=1, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time
def build_dl_lstm_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs, filename):



    ## Network architecture

    from keras.utils import to_categorical

    from keras.callbacks import ModelCheckpoint

    from keras.callbacks import EarlyStopping

    from keras.layers import Masking

    from keras.initializers import Constant

    import time

    

    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    

    start_time = time.time()

    

    

    from numpy.random import seed

    seed(42)

    from tensorflow import set_random_seed

    set_random_seed(42)



    embedding_size= embedding_dim + 2

    batch_size = 1024

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_size, input_length = max_len, 

                        weights=[embedding_matrix], trainable = False))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(GRU(units = 64, recurrent_dropout=dropouts, dropout=dropouts, return_sequences=True))

    

    model.add(Conv1D(128, kernel_size = 1, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(256, kernel_size = 3, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(512, kernel_size = 5, strides = 1,  padding='valid', activation='relu'))

    

    model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropouts))

    

    

    model.add(Dense(2, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop



    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=0, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time

def build_dl_gru_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs, filename):



    ## Network architecture

    from keras.utils import to_categorical

    from keras.callbacks import ModelCheckpoint

    from keras.callbacks import EarlyStopping

    from keras.layers import Masking

    from keras.initializers import Constant

    import time

    

    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    

    start_time = time.time()

    

    

    from numpy.random import seed

    seed(42)

    from tensorflow import set_random_seed

    set_random_seed(42)



    embedding_size= embedding_dim + 2

    batch_size = 1024

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_size, input_length = max_len, 

                        weights=[embedding_matrix], trainable = False))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(GRU(units = 84, recurrent_dropout=dropouts, dropout=dropouts, return_sequences=True))

    

    model.add(Conv1D(42, kernel_size = 1, strides = 1,  padding='valid', activation='relu'))

    #model.add(Conv1D(256, kernel_size = 3, strides = 1,  padding='valid', activation='relu'))

    #model.add(Conv1D(512, kernel_size = 5, strides = 1,  padding='valid', activation='relu'))

    

    model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropouts))

    

    model.add(Dense(26, activation="relu"))

    model.add(Dropout(dropouts))

        

    model.add(Dense(2, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop



    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=0, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time

def build_dl_bidirectional_gru_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs, filename):



    ## Network architecture

    from keras.utils import to_categorical

    from keras.callbacks import ModelCheckpoint

    from keras.callbacks import EarlyStopping

    from keras.layers import Masking

    from keras.initializers import Constant

    import time

    

    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    

    start_time = time.time()

    

    

    from numpy.random import seed

    seed(42)

    from tensorflow import set_random_seed

    set_random_seed(42)



    embedding_size= embedding_dim + 2

    batch_size = 1024

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_size, input_length = max_len, 

                        weights=[embedding_matrix], trainable = False))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(Bidirectional(GRU(units = 64, recurrent_dropout=dropouts, dropout=dropouts, return_sequences=True)))

    

    model.add(Conv1D(128, kernel_size = 1, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(256, kernel_size = 3, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(512, kernel_size = 5, strides = 1,  padding='valid', activation='relu'))

    

    model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropouts))

    

    model.add(Dense(16, activation="relu"))

    model.add(Dropout(dropouts))

    

    

    model.add(Dense(2, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop



    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=0, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time

'''

dl_lstm_model, history_lstm, elapsed_time = build_dl_lstm_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=30, filename="lstm")

print("Elapsed time in seconds:", elapsed_time)

'''
'''

dl_lstm_cnn_model, history_lstm_cnn, elapsed_time = build_dl_lstm_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=30, filename="lstm_cnn")

print("Elapsed time in seconds:", elapsed_time)

'''


dl_gru_cnn_model, history_gru_cnn, elapsed_time = build_dl_gru_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=30, filename="gru_cnn")

print("Elapsed time in seconds:", elapsed_time)

'''

dl_bidirectional_gru_cnn_model, history_bidirectional_gru_cnn, elapsed_time = build_dl_bidirectional_gru_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=30, filename="bidirectional_gru_cnn")

print("Elapsed time in seconds:", elapsed_time)

'''
'''

plot_history(history_lstm)





y_pred_lstm = dl_lstm_model.predict_classes(xvalid, verbose=1, batch_size = 256)

print()



print(classification_report(yvalid, y_pred_lstm))



print()

print("accuracy_score", accuracy_score(yvalid, y_pred_lstm))



print()

print("Weighted Averaged validation metrics")

print("precision_score", precision_score(yvalid, y_pred_lstm, average='weighted'))

print("recall_score", recall_score(yvalid, y_pred_lstm, average='weighted'))

print("f1_score", f1_score(yvalid, y_pred_lstm, average='weighted'))



print()

from sklearn.metrics import confusion_matrix

import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, y_pred_lstm)





print("elapsed time:", round(elapsed_time), "seconds")

'''
#plot_history(history_lstm_cnn)
plot_history(dl_gru_cnn_model)
xtest = tokenizer.texts_to_sequences([' '.join(sent) for sent in tokens_without_stopwords_and_lemmatized_test])

xtest = pad_sequences(xtest, maxlen=max_len)



'''

y_pred_test_lstm = dl_lstm_model.predict_classes(xtest, verbose=1, batch_size = 512)

submission = pd.DataFrame()

submission['qid'] = df_test.qid

submission['prediction'] = y_pred_test_lstm

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission.csv',index=False)

'''



'''

y_pred_test_lstm_cnn = dl_lstm_cnn_model.predict_classes(xtest, verbose=1, batch_size = 1024)

submission = pd.DataFrame()

submission['qid'] = df_test.qid

submission['prediction'] = y_pred_test_lstm_cnn

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission.csv',index=False)

'''





y_pred_test_gru_cnn = dl_gru_cnn_model.predict_classes(xtest, verbose=1, batch_size = 1024)

submission = pd.DataFrame()

submission['qid'] = df_test.qid

submission['prediction'] = y_pred_test_gru_cnn

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission.csv',index=False)



'''

y_pred_test_bidirectional_gru_cnn = dl_bidirectional_gru_cnn_model.predict_classes(xtest, verbose=1, batch_size = 1024)

submission = pd.DataFrame()

submission['qid'] = df_test.qid

submission['prediction'] = y_pred_test_bidirectional_gru_cnn

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission.csv',index=False)

'''