import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



from sklearn.model_selection import train_test_split



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import json

from tqdm import tqdm



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
RECIPE_INGREDIENTS_PATH = '../input/recipe-ingredients-dataset/'

FASTTEXT_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
# Codes borrowed from [The Effect of Word Embeddings on Bias](https://www.kaggle.com/nholloway/the-effect-of-word-embeddings-on-bias/data)



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f, position=0))



    

def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in tqdm(word_index.items(), position=0):

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            embedding_matrix[i] = np.random.normal(-1, 1, (1, 300))

    return embedding_matrix
def build_dataset(data_path, cuisine=True):

    

    x_rid = []

    y_rid = []

    i_rid = []

    

    with open(data_path) as f:

        rid_list = json.load(f)

    

        for rid in tqdm(rid_list, position=0):

            x_rid.append([ing for ing in rid['ingredients']])

            if cuisine:

                y_rid.append(rid['cuisine'])

            i_rid.append(rid['id'])

        

    return i_rid, x_rid, y_rid
def n_gram_sequences(ingredients, tokenizer):

    tokenizer.fit_on_texts(ingredients)

    total_words = len(tokenizer.word_index) + 1

    

    x_sequences = []

    for items in ingredients:

        token_list = tokenizer.texts_to_sequences([items])[0]

        for i in range(1, len(token_list)):

            n_grams = token_list[: i + 1]

            x_sequences.append(n_grams)

    

    return x_sequences, total_words, tokenizer





def n_gram_padded(x_sequences):

    max_len = max([len(x) for x in x_sequences])

    x_sequences = np.array(pad_sequences(x_sequences, maxlen=max_len, padding='pre'))

    predictors, label = x_sequences[:, :-1], x_sequences[:, -1]

    return predictors, label, max_len
i_rid_train, x_rid_train, y_rid_train = build_dataset(RECIPE_INGREDIENTS_PATH + 'train.json')

i_rid_test, x_rid_test, _ = build_dataset(RECIPE_INGREDIENTS_PATH + 'test.json', cuisine=False)



print('Train:')

print('Number of recipes-ingredients %d' % (len(i_rid_train)))

print('Number of unique ingredients %d' % (len(list(set(x for l in x_rid_train for x in l)))))

print('Number of unique recipes %d' % (len(list(set(y_rid_train)))))



print()

print('Test')

print('Number of recipes-ingredients %d' % (len(i_rid_test)))

print('Number of unique ingredients %d' % (len(list(set(x for l in x_rid_test for x in l)))))
ingredients = x_rid_train + x_rid_test

ingredients = [' '.join([item.replace(' ', '-') for item in items]) for items in ingredients]



print('Number of recipes-ingredients %d' % len(ingredients))

print('Sample: %s' % ingredients[10])
tokenizer = Tokenizer()

x_sequences, total_words, tokenizer = n_gram_sequences(ingredients, tokenizer)

predictors, labels, max_len = n_gram_padded(x_sequences)



x_train, x_test, y_train, y_test = train_test_split(predictors, labels, test_size=0.05, random_state=42)



print('Predictors %s' % str(predictors.shape))

print('Labels %s' % str(labels.shape))

print('Train: %s, Test: %s' % (len(x_train), len(x_test)))

print('Max length of sequences %d' % max_len)
fasttext_matrix = build_matrix(tokenizer.word_index, FASTTEXT_PATH)

print("Number of unique words %d" % fasttext_matrix.shape[0])
print(x_train[0], y_train[0])
def create_model(max_len, rnn_units, total_words, embedding_matrix):

    inputs = keras.layers.Input(shape=(max_len,))

    x = keras.layers.Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inputs)

    # x = keras.layers.Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=True)(inputs)

    x = keras.layers.CuDNNGRU(rnn_units, name='gru_1')(x)

    outputs = tf.keras.layers.Dense(total_words, activation='softmax', name='output')(x)

    

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy'],

        optimizer='adam')

    

    return model
model = create_model(max_len - 1, 100, total_words, fasttext_matrix)

model.summary()
r = model.fit(x_train, y_train, validation_split=0.05, epochs=10, batch_size=256, verbose=1)
def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)



def generate_text(seed_text, next_words, max_len, model, tokenizer):

    idx2word = {idx: word for word, idx in tokenizer.word_index.items()}

    # Converting our start string to numbers (vectorizing)

    x_pred = tokenizer.texts_to_sequences([seed_text])[0]

    x_pred = np.array(pad_sequences([x_pred], maxlen=max_len - 1, padding='pre'))

    

    # Empty string to store our results

    text_generated = []

    

    # Low temperatures results in more predictable text.

    # Higher temperatures results in more surprising text.

    # Experiment to find the best setting.

    temperature = 1.0

    

    # Here batch size == 1

    model.reset_states()

    for i in range(next_words):

        predictions = model.predict(x_pred, verbose=0)[0]

        predicted_id = sample(predictions, temperature)

        text_generated.append(idx2word[predicted_id])

    

    return seed_text + ' ' + ' '.join(text_generated)
print(generate_text("pimentos sweet-pepper dried-oregano olive-oil", 5, max_len, model, tokenizer))