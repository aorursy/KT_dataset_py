import pandas as pd



import matplotlib

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import nltk
df = pd.read_csv("../input/train.tsv", sep="\t")



df_test = pd.read_csv("../input/test.tsv", sep="\t")
df.head(10)
df_test.head(10)
example = df[(df['PhraseId'] >= 0) & (df['PhraseId'] <= 2)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 517) & (df['PhraseId'] <= 518)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 68) & (df['PhraseId'] <= 69)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 10737) & (df['PhraseId'] <= 10738)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 22) & (df['PhraseId'] <= 24)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])



print()



print(example["Phrase"].values[2], " - Sentiment:", example["Sentiment"].values[2])
example = df[(df['PhraseId'] >= 46) & (df['PhraseId'] <= 47)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
# A function to tokenize phrases



def tokenize_the_text(phrases):

    

    from nltk.tokenize import word_tokenize

    from nltk.text import Text

    

    tokens = [word for word in phrases]

    tokens = [word.lower() for word in tokens]

    tokens = [word_tokenize(word) for word in tokens]

    

    return tokens



crude_tokens = tokenize_the_text(df.Phrase)

print(crude_tokens[0:10])
# a function to construct the vocabulary



def create_a_vocab(tokens):

    

    vocab = set()



    for sentence in tokens:

        for word in sentence:

            vocab.add(word)



    vocab = list(vocab)



    return vocab

    

vocab = create_a_vocab(crude_tokens)



print(len(vocab))
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



df_test = pd.read_csv("../input/test.tsv", sep="\t")



###

tokens_uncleaned = tokenize_the_text(df.Phrase.values)

tokens_uncleaned_test = tokenize_the_text(df_test.Phrase.values)

sentences = [' '.join(sent) for sent in tokens_uncleaned]

sentences_test = [' '.join(sent) for sent in tokens_uncleaned_test]

###





all_corpus = sentences + sentences_test



vocab_all_corpus = create_a_vocab(tokens_uncleaned + tokens_uncleaned_test)

max_len = max([len(elem.split()) for elem in all_corpus])

#print(max_len)





tokenizer = Tokenizer(lower=True, filters='')

tokenizer.fit_on_texts(all_corpus)



vocabulary_size = len(tokenizer.word_index) + 1

#print(vocabulary_size)



X = tokenizer.texts_to_sequences(sentences)

y = df.Sentiment.values

X = pad_sequences(X, maxlen=max_len)





xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2, shuffle=True)
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
dl_performance_metrics_df = pd.DataFrame(columns=['accuracy','F1-score','training-time'], index=['LSTM', 'BiLSTM', 'CNN', 'LSTM_CNN', 'BiLSTM_CNN'])
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

    



    embedding_size= 300

    batch_size = 128

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len, 

                        trainable = True, mask_zero=True))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(LSTM(int(embedding_size/2), recurrent_dropout=dropouts, dropout=dropouts, return_sequences=False))

    

    model.add(Dense(5, activation='softmax'))



    print(model.summary())



     

    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy'])

    

    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=3, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time


dl_lstm_model, history_lstm, elapsed_time = build_dl_lstm_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=30, filename="lstm")

print("Elapsed time in seconds:", elapsed_time)
from keras.utils.vis_utils import plot_model



plot_model(dl_lstm_model, to_file='dl_lstm_model.png', show_shapes=True, show_layer_names=True)



import matplotlib.pyplot as plt

import numpy as np

from PIL import Image



img = Image.open('dl_lstm_model.png')

plt.rcParams["figure.figsize"] = (10,10)

plt.axis('off')

plt.grid(False)

imgplot = plt.imshow(img)
plot_history(history_lstm)



y_pred_lstm = dl_lstm_model.predict_classes(xvalid, verbose=1)

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



dl_performance_metrics_df.loc['LSTM']['training-time'] = elapsed_time

dl_performance_metrics_df.loc['LSTM']['accuracy'] = accuracy_score(yvalid, y_pred_lstm)

dl_performance_metrics_df.loc['LSTM']['F1-score'] = f1_score(yvalid, y_pred_lstm, average='weighted')



print("elapsed time:", round(elapsed_time), "seconds")
def build_dl_bidirectional_lstm_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs, filename):



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



    

    embedding_size = 300

    batch_size = 256

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len, 

     trainable = False, mask_zero=True))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(Bidirectional(LSTM(int(embedding_size/2), recurrent_dropout=dropouts, dropout=dropouts, return_sequences=False)))

    

    model.add(Dense(5, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop



    

    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=1, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=3, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time


dl_bidirectional_lstm_model, history_bidirectional_lstm, elapsed_time = build_dl_bidirectional_lstm_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=50, filename="bidirectional_lstm")

print("Elapsed time in seconds:", elapsed_time)
from keras.utils.vis_utils import plot_model



plot_model(dl_bidirectional_lstm_model, to_file='dl_bidirectional_lstm_model.png', show_shapes=True, show_layer_names=True)



import matplotlib.pyplot as plt

import numpy as np

from PIL import Image



img = Image.open('dl_bidirectional_lstm_model.png')

plt.rcParams["figure.figsize"] = (10,10)

plt.axis('off')

plt.grid(False)

imgplot = plt.imshow(img)
plot_history(history_bidirectional_lstm)



y_pred_bidirectional_lstm = dl_bidirectional_lstm_model.predict_classes(xvalid, verbose=1)

print()



print(classification_report(yvalid, y_pred_bidirectional_lstm))



print()

print("accuracy_score", accuracy_score(yvalid, y_pred_bidirectional_lstm))



print()

print("Weighted Averaged validation metrics")

print("precision_score", precision_score(yvalid, y_pred_bidirectional_lstm, average='weighted'))

print("recall_score", recall_score(yvalid, y_pred_bidirectional_lstm, average='weighted'))

print("f1_score", f1_score(yvalid, y_pred_bidirectional_lstm, average='weighted'))



print()

from sklearn.metrics import confusion_matrix

import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, y_pred_bidirectional_lstm)





dl_performance_metrics_df.loc['BiLSTM']['training-time'] = elapsed_time

dl_performance_metrics_df.loc['BiLSTM']['accuracy'] = accuracy_score(yvalid, y_pred_bidirectional_lstm)

dl_performance_metrics_df.loc['BiLSTM']['F1-score'] = f1_score(yvalid, y_pred_bidirectional_lstm, average='weighted')



print("elapsed time:", round(elapsed_time), "seconds")
def build_dl_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs, filename):



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



    

    embedding_size = 300

    batch_size = 256

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len, trainable = True))

    

    model.add(SpatialDropout1D(dropouts))

    

    

    model.add(Conv1D(128, kernel_size = 1, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(256, kernel_size = 3, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(512, kernel_size = 5, strides = 1,  padding='valid', activation='relu'))

    

    model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropouts))



    model.add(Dense(int(embedding_size/2), activation="relu"))

    model.add(Dropout(dropouts))

    

    model.add(Dense(5, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop



    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=3, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time


dl_cnn_model, history_cnn, elapsed_time = build_dl_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=30, filename="cnn")

print("Elapsed time in seconds:", elapsed_time)
from keras.utils.vis_utils import plot_model



plot_model(dl_cnn_model, to_file='dl_cnn_model.png', show_shapes=True, show_layer_names=True)



import matplotlib.pyplot as plt

import numpy as np

from PIL import Image



img = Image.open('dl_cnn_model.png')

plt.rcParams["figure.figsize"] = (21,20)

plt.axis('off')

plt.grid(False)

imgplot = plt.imshow(img)
plot_history(history_cnn)



y_pred_cnn = dl_cnn_model.predict_classes(xvalid, verbose=1)

print()



print(classification_report(yvalid, y_pred_cnn))



print()

print("accuracy_score", accuracy_score(yvalid, y_pred_cnn))



print()

print("Weighted Averaged validation metrics")

print("precision_score", precision_score(yvalid, y_pred_cnn, average='weighted'))

print("recall_score", recall_score(yvalid, y_pred_cnn, average='weighted'))

print("f1_score", f1_score(yvalid, y_pred_cnn, average='weighted'))



print()

from sklearn.metrics import confusion_matrix

import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, y_pred_cnn)





dl_performance_metrics_df.loc['CNN']['training-time'] = elapsed_time

dl_performance_metrics_df.loc['CNN']['accuracy'] = accuracy_score(yvalid, y_pred_cnn)

dl_performance_metrics_df.loc['CNN']['F1-score'] = f1_score(yvalid, y_pred_cnn, average='weighted')



print("elapsed time:", round(elapsed_time), "seconds")
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



    

    embedding_size= 300

    batch_size = 128

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len, trainable = True))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(LSTM(int(embedding_size/2), recurrent_dropout=dropouts, dropout=dropouts, 

                   return_sequences=True))

    

    

    model.add(Conv1D(128, kernel_size = 1, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(256, kernel_size = 3, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(512, kernel_size = 5, strides = 1,  padding='valid', activation='relu'))

    

    model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropouts))



    model.add(Dense(int(embedding_size/2), activation="relu"))

    model.add(Dropout(dropouts))

    

    model.add(Dense(5, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop



    

    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=3, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time



dl_lstm_cnn_model, history_lstm_cnn, elapsed_time = build_dl_lstm_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=30, filename="lstm_cnn")

print("Elapsed time in seconds:", elapsed_time)
from keras.utils.vis_utils import plot_model



plot_model(dl_lstm_cnn_model, to_file='dl_lstm_cnn_model.png', show_shapes=True, show_layer_names=True)



import matplotlib.pyplot as plt

import numpy as np

from PIL import Image



img = Image.open('dl_lstm_cnn_model.png')

plt.rcParams["figure.figsize"] = (20,25)

plt.axis('off')

plt.grid(False)

imgplot = plt.imshow(img)
plot_history(history_lstm_cnn)



y_pred_lstm_cnn = dl_lstm_cnn_model.predict_classes(xvalid, verbose=1)

print()



print(classification_report(yvalid, y_pred_lstm_cnn))



print()

print("accuracy_score", accuracy_score(yvalid, y_pred_lstm_cnn))



print()

print("Weighted Averaged validation metrics")

print("precision_score", precision_score(yvalid, y_pred_lstm_cnn, average='weighted'))

print("recall_score", recall_score(yvalid, y_pred_lstm_cnn, average='weighted'))

print("f1_score", f1_score(yvalid, y_pred_lstm_cnn, average='weighted'))



print()

from sklearn.metrics import confusion_matrix

import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, y_pred_lstm_cnn)





dl_performance_metrics_df.loc['LSTM_CNN']['training-time'] = elapsed_time

dl_performance_metrics_df.loc['LSTM_CNN']['accuracy'] = accuracy_score(yvalid, y_pred_lstm_cnn)

dl_performance_metrics_df.loc['LSTM_CNN']['F1-score'] = f1_score(yvalid, y_pred_lstm_cnn, average='weighted')



print("elapsed time:", round(elapsed_time), "seconds")
def build_dl_bidirectional_lstm_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs, filename):



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

    



    embedding_size = 300

    batch_size = 256

    dropouts = 0.2

    epochs = num_of_epochs

    

    model=Sequential()

    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len, 

                        trainable = True))

    

    model.add(SpatialDropout1D(dropouts))

    

    model.add(Bidirectional(LSTM(int(embedding_size/2), recurrent_dropout=dropouts, dropout=dropouts, 

                   return_sequences=True)))

    

    model.add(Conv1D(128, kernel_size=2, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(256, kernel_size=3, strides = 1,  padding='valid', activation='relu'))

    model.add(Conv1D(512, kernel_size=5, strides = 1,  padding='valid', activation='relu'))

    

    model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropouts))

    

    model.add(Dense(int(embedding_size/2), activation="relu"))

    model.add(Dropout(dropouts))

    

    model.add(Dense(5, activation='softmax'))



    print(model.summary())



    model.compile(loss='categorical_crossentropy', optimizer = 'nadam', metrics=['categorical_accuracy']) # RMSprop





    '''

    saves the model weights after each epoch if the val_acc loss decreased

    '''

    checkpointer = ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', filepath=''+filename+'.hdf5', verbose=2, save_best_only=True)

    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=3, verbose=0, mode='max')



    history = model.fit(x = xtrain, y = to_categorical(ytrain), validation_data=(xvalid, to_categorical(yvalid)), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[checkpointer, earlyStopping])



    model = load_model(''+filename+'.hdf5')

    

    elapsed_time = time.time() - start_time

    

    return model, history, elapsed_time


dl_bidirectional_lstm_cnn_model, history_bidirectional_lstm_cnn, elapsed_time = build_dl_bidirectional_lstm_cnn_model(xtrain, ytrain, xvalid, yvalid, num_of_epochs=50, filename="bidirectional_lstm_cnn")

print("Elapsed time in seconds:", elapsed_time)
from keras.utils.vis_utils import plot_model



plot_model(dl_bidirectional_lstm_cnn_model, to_file='dl_bidirectional_lstm_cnn_model.png', show_shapes=True, show_layer_names=True)



import matplotlib.pyplot as plt

import numpy as np

from PIL import Image



img = Image.open('dl_bidirectional_lstm_cnn_model.png')

plt.rcParams["figure.figsize"] = (20,25)

plt.axis('off')

plt.grid(False)

imgplot = plt.imshow(img)
plot_history(history_bidirectional_lstm_cnn)



y_pred_bidirectional_lstm_cnn = dl_bidirectional_lstm_cnn_model.predict_classes(xvalid, verbose=1)

print()



print(classification_report(yvalid, y_pred_bidirectional_lstm_cnn))



print()

print("accuracy_score", accuracy_score(yvalid, y_pred_bidirectional_lstm_cnn))



print()

print("Weighted Averaged validation metrics")

print("precision_score", precision_score(yvalid, y_pred_bidirectional_lstm_cnn, average='weighted'))

print("recall_score", recall_score(yvalid, y_pred_bidirectional_lstm_cnn, average='weighted'))

print("f1_score", f1_score(yvalid, y_pred_bidirectional_lstm_cnn, average='weighted'))



print()

from sklearn.metrics import confusion_matrix

import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, y_pred_bidirectional_lstm_cnn)





dl_performance_metrics_df.loc['BiLSTM_CNN']['training-time'] = elapsed_time

dl_performance_metrics_df.loc['BiLSTM_CNN']['accuracy'] = accuracy_score(yvalid, y_pred_bidirectional_lstm_cnn)

dl_performance_metrics_df.loc['BiLSTM_CNN']['F1-score'] = f1_score(yvalid, y_pred_bidirectional_lstm_cnn, average='weighted')



print("elapsed time:", round(elapsed_time), "seconds")
dl_performance_metrics_df.sort_values(by="accuracy", ascending=False)
sns.set(rc={'figure.figsize':(15.27,6.27)})

dl_performance_metrics_df.sort_values(by="accuracy", ascending=False).accuracy.plot(kind="bar")
dl_performance_metrics_df.sort_values(by="F1-score", ascending=False)
sns.set(rc={'figure.figsize':(15.27,6.27)})

dl_performance_metrics_df.sort_values(by="F1-score", ascending=False)["F1-score"].plot(kind="bar")
dl_performance_metrics_df.sort_values(by="training-time", ascending=False)
sns.set(rc={'figure.figsize':(15.27,6.27)})

dl_performance_metrics_df.sort_values(by="training-time", ascending=False)["training-time"].plot(kind="bar")
y_pred_lstm = dl_lstm_model.predict_classes(xvalid, verbose=1)



y_pred_cnn = dl_cnn_model.predict_classes(xvalid, verbose=1)



y_pred_bidirectional_lstm = dl_bidirectional_lstm_model.predict_classes(xvalid, verbose=1)



y_pred_lstm_cnn = dl_lstm_cnn_model.predict_classes(xvalid, verbose=1)



y_pred_bidirectional_lstm_cnn =  dl_bidirectional_lstm_cnn_model.predict_classes(xvalid, verbose=1)

ensemble_all_dl_pred_df = pd.DataFrame({'model_lstm':y_pred_lstm,

                                                'model_bidirectional_lstm':y_pred_bidirectional_lstm,

                                                'model_cnn':y_pred_cnn,

                                                'model_bidirectional_lstm_cnn':y_pred_bidirectional_lstm_cnn,

                                                'model_lstm_cnn':y_pred_lstm_cnn,

                                                })





pred_mode = ensemble_all_dl_pred_df.agg('mode',axis=1)[0].values



print()

print(classification_report(yvalid, pred_mode))



print()

print("accuracy_score", accuracy_score(yvalid, pred_mode))



print()

print("Weighted Averaged validation metrics")

print("precision_score", precision_score(yvalid, pred_mode, average='weighted'))

print("recall_score", recall_score(yvalid, pred_mode, average='weighted'))

print("f1_score", f1_score(yvalid, pred_mode, average='weighted'))



print()

from sklearn.metrics import confusion_matrix

import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, pred_mode)

xtest = tokenizer.texts_to_sequences(df_test.Phrase.values)

xtest = pad_sequences(xtest, maxlen=max_len)



y_pred_test_lstm = dl_lstm_model.predict_classes(xtest, verbose=1)

submission = pd.DataFrame()

submission['PhraseId'] = df_test.PhraseId

submission['Sentiment'] = y_pred_test_lstm

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission_lstm.csv',index=False)





y_pred_test_bidirectional_lstm = dl_bidirectional_lstm_model.predict_classes(xtest, verbose=1)

submission = pd.DataFrame()

submission['PhraseId'] = df_test.PhraseId

submission['Sentiment'] = y_pred_test_bidirectional_lstm

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission_bidirectional_lstm.csv',index=False)





y_pred_test_cnn = dl_cnn_model.predict_classes(xtest, verbose=1)

submission = pd.DataFrame()

submission['PhraseId'] = df_test.PhraseId

submission['Sentiment'] = y_pred_test_cnn

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission_cnn.csv',index=False)





y_pred_test_lstm_cnn = dl_lstm_cnn_model.predict_classes(xtest, verbose=1)

submission = pd.DataFrame()

submission['PhraseId'] = df_test.PhraseId

submission['Sentiment'] = y_pred_test_lstm_cnn

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission_lstm_cnn.csv',index=False)





y_pred_test_bidirectional_lstm_cnn = dl_bidirectional_lstm_cnn_model.predict_classes(xtest, verbose=1)

submission = pd.DataFrame()

submission['PhraseId'] = df_test.PhraseId

submission['Sentiment'] = y_pred_test_bidirectional_lstm_cnn

#submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission_bidirectional_lstm_cnn.csv',index=False)
ensemble_all_dl_pred_test_df = pd.DataFrame({'model_lstm':y_pred_test_lstm,

                                                'model_bidirectional_lstm':y_pred_test_bidirectional_lstm,

                                                'model_cnn':y_pred_test_cnn,

                                                'model_lstm_cnn':y_pred_test_lstm_cnn,

                                                'model_bidirectional_lstm_cnn':y_pred_test_bidirectional_lstm_cnn})





pred_test_mode = ensemble_all_dl_pred_test_df.agg('mode',axis=1)[0].values

submission = pd.DataFrame()

submission['PhraseId'] = df_test.PhraseId

submission['Sentiment'] = pred_test_mode

submission['Sentiment'] = submission.Sentiment.astype(int)

submission.to_csv('submission_ensemble.csv',index=False)
