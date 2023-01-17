import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn import preprocessing

import nltk

import re

from tensorflow import keras

from sklearn.manifold import TSNE

from IPython.core.interactiveshell import InteractiveShell

from sklearn.metrics import accuracy_score

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", 50, "display.max_columns", 50)
data = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
label_encoder = preprocessing.LabelEncoder()

data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

data.head()
plt.figure(figsize=(6,3))

data.sentiment.value_counts().plot(kind='bar', rot=360)

plt.show()
print(data.review[0])
stopWords = nltk.corpus.stopwords.words('english')

snoStemmer = nltk.stem.SnowballStemmer('english')

wnlemmatizer = nltk.stem.WordNetLemmatizer()



def clean_html(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext



def clean_punc(word):

    cleaned = re.sub(r'[?|!|\'|#]', r'', word)

    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)

    return cleaned



def filtered_sents(data_frame):

    # Creating a list of filtered sentences:

    final_string = []

    s = ''

    for sentence in data_frame:

        filtered_sentence = []

        sentence = clean_html(sentence)

        for word in sentence.split():

            for cleaned_word in clean_punc(word).split():

                if (cleaned_word.isalpha() and (len(cleaned_word) > 2) and cleaned_word not in stopWords):

                    lemmatized_word = wnlemmatizer.lemmatize(cleaned_word.lower())

                    stemmed_word = snoStemmer.stem(lemmatized_word)

                    filtered_sentence.append(stemmed_word)

                else:

                    continue

        strl = ' '.join(filtered_sentence)

        final_string.append(strl)

    return final_string



data.cleaned_review = filtered_sents(data.review)
print(data.cleaned_review[0])
tokenizer = Tokenizer()

tokenizer.fit_on_texts(data.cleaned_review)

list_tokenized_data = tokenizer.texts_to_sequences(data.cleaned_review)

word_index = tokenizer.word_index

index_word = dict([(value, key) for (key, value) in word_index.items()])
print(pd.Series(word_index).head())

print('\n')

print(pd.Series(word_index).tail())
length_list = []

for i in list_tokenized_data:

    length_list.append(len(i))



f, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False)

pd.Series(length_list).hist(bins=100, ax = axes[0])

pd.Series(length_list).hist(bins=100, ax = axes[1])

plt.xlim(0,400)

plt.show()
MAX_LEN = 256

X = keras.preprocessing.sequence.pad_sequences(list_tokenized_data,

                                                        padding='post',

                                                        maxlen=MAX_LEN)

y = data.sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 0)

print(f'SHAPE: \n X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val:{y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')
vocab_size = max(np.max(X_train), np.max(X_test)) + 1

emb_size = 16

model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, emb_size, input_length = MAX_LEN, name = 'word_embedding'))

model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(emb_size, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))

model.add(keras.layers.Dropout(0.5, seed=0))

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
LEARNING_RATE = 1e-3

OPTIMIZER = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

model.compile(optimizer=OPTIMIZER,

              loss='binary_crossentropy',

              metrics=['acc'])



CALLBACKS = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

history = model.fit(X_train,

            y_train,

            epochs=100,

            batch_size=512,

            validation_data=(X_val, y_val),

            verbose=1,

            callbacks=CALLBACKS)
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
history_dict = history.history

metric_list = list(history_dict.keys())



loss = history_dict['loss']

val_loss = history_dict['val_loss']

# "bo" is for "blue dot"

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'g', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()



plt.clf()   # clear figure



if 'acc' in metric_list:

    acc = history_dict['acc']

    val_acc = history_dict['val_acc']

    plt.plot(epochs, acc, 'b', label='Training acc')

    plt.plot(epochs, val_acc, 'g', label='Validation acc')

    plt.ylabel('Accuracy')

    plt.title('Training and validation accuracy')

    

elif 'auc' in metric_list:

    auc = history_dict['auc']

    val_auc = history_dict['val_auc']

    plt.plot(epochs, auc, 'bo', label='Training auc')

    plt.plot(epochs, val_auc, 'b', label='Validation auc')

    plt.ylabel('AUC')

    plt.title('Training and validation auc')

    

elif 'precision' and 'recall' in list(metric_list):

    precision = history_dict['precision']

    val_precision = history_dict['val_precision']

    recall = history_dict['recall']

    val_recall = history_dict['val_recall']

    plt.plot(epochs, precision, 'bo', label='Training precision')

    plt.plot(epochs, val_precision, 'b', label='Validation precision')

    plt.plot(epochs, recall, 'ro', label='Training recall')

    plt.plot(epochs, val_recall, 'r', label='Validation recall')

    plt.ylabel('Precision and Recall')

    plt.title('Training and validation precision and recall')



plt.xlabel('Epochs')

plt.legend()

plt.show()
# Extract embeddings

word_layer = model.get_layer('word_embedding')

word_weights = word_layer.get_weights()[0]

word_weights.shape
word_weights = word_weights / np.linalg.norm(word_weights, axis = 1).reshape((-1, 1))

word_weights[0][:10]

np.sum(np.square(word_weights[0]))
%matplotlib inline

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 15



def find_similar(name, weights, index_name = 'word', n = 10, least = False, return_dist = False, plot = False):

    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""

    

    # Select index and reverse index

    if index_name == 'word':

        index = word_index

        rindex = index_word

    elif index_name == 'page':

        index = link_index

        rindex = index_link

    

    # Check to make sure `name` is in index

    try:

        # Calculate dot product between word and all others

        dists = np.dot(weights, weights[index[name]])

    except KeyError:

        print(f'{name} Not Found.')

        return

    

    # Sort distance indexes from smallest to largest

    sorted_dists = np.argsort(dists)

    

    # Plot results if specified

    if plot:

        

        # Find furthest and closest items

        furthest = sorted_dists[:(n // 2)]

        closest = sorted_dists[-n-1: len(dists) - 1]

        items = [rindex[c] for c in furthest]

        items.extend(rindex[c] for c in closest)

        

        # Find furthest and closets distances

        distances = [dists[c] for c in furthest]

        distances.extend(dists[c] for c in closest)

        

        colors = ['r' for _ in range(n //2)]

        colors.extend('g' for _ in range(n))

        

        data = pd.DataFrame({'distance': distances}, index = items)

        

        # Horizontal bar chart

        data['distance'].plot.barh(color = colors, figsize = (10, 8),

                                   edgecolor = 'k', linewidth = 2)

        plt.xlabel('Cosine Similarity');

        plt.axvline(x = 0, color = 'k');

        

        # Formatting for italicized title

        name_str = f'{index_name.capitalize()}s Most and Least Similar to'

        for word in name.split():

            # Title uses latex for italize

            name_str += ' $\it{' + word + '}$'

        plt.title(name_str, x = 0.2, size = 28, y = 1.05)

        

        return None

    

    # If specified, find the least similar

    if least:

        # Take the first n from sorted distances

        closest = sorted_dists[:n]

         

        print(f'{index_name.capitalize()}s furthest from {name}.\n')

        

    # Otherwise find the most similar

    else:

        # Take the last n sorted distances

        closest = sorted_dists[-n:]

        

        # Need distances later on

        if return_dist:

            return dists, closest

        

        

        #print(f'{index_name.capitalize()}s closest to {name}.\n')

        

    # Need distances later on

    if return_dist:

        return dists, closest

    

    

    # Print formatting

    max_width = max([len(rindex[c]) for c in closest])

    

    # Print the most similar and distances

    #for c in reversed(closest):

        #print(f'{index_name.capitalize()}: {rindex[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')

        

    return closest
find_similar('amaz', word_weights, least = True, n = 5, plot = True)
find_similar('livingroom', word_weights, least = True, n = 5, plot = True)
def reduce_dim(weights, components = 2, method = 'tsne'):

    """Reduce dimensions of embeddings"""

    if method == 'tsne':

        return TSNE(components, metric = 'cosine', random_state=0).fit_transform(weights)

    elif method == 'umap':

        # Might want to try different parameters for UMAP

        return UMAP(n_components=components, metric = 'cosine', 

                    init = 'random', n_neighbors = 5).fit_transform(weights)
word_r = reduce_dim(word_weights, components = 2, method = 'tsne')

word_r.shape
clustered_pos = find_similar('great', word_weights, n = 10)

clustered_neg = find_similar('bad', word_weights, n = 10)

clustered_neutral = find_similar('livingroom', word_weights, n = 10)

np.random.seed(seed=0)

clustered_pos = np.random.choice(clustered_pos, 5)

np.random.seed(seed=0)

clustered_neg = np.random.choice(clustered_neg, 5)

np.random.seed(seed=0)

clustered_neutral = np.random.choice(clustered_neutral, 5)

clustered_words = np.concatenate((clustered_pos, clustered_neg))

clustered_words = np.concatenate((clustered_words, clustered_neutral))



InteractiveShell.ast_node_interactivity = 'last' 

plt.figure(figsize = (14, 12))



#Plot all words

fig = plt.scatter(word_r[:, 0], word_r[:, 1], marker = '.', color = 'lightblue')



plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('TSNE Visualization of Word Embeddings');

np.random.seed(seed=0)

for index in clustered_words: 

    x, y = word_r[index, 0], word_r[index, 1];

    s = ''.join([' $\it{'+ i + '}$' for i in index_word[index].split()])

    plt.scatter(x, y, s = 250, color = 'r', marker = '*', edgecolor = 'k')

    plt.text(x + np.random.randint(-5,5), y + np.random.randint(-5,5), s, fontsize = 14);

plt.show()