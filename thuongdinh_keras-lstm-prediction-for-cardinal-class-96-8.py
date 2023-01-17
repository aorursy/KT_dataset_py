import numpy as np

from numpy import argmax

import pandas as pd



# Keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import TimeDistributed

from keras.layers import RepeatVector

from keras.preprocessing.text import text_to_word_sequence
# Max columns for display

pd.options.display.max_columns = 999



# Disable warning message

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/en_train.csv")
# Refer to the ebook long_short_term_memory_networks_with_python



def integer_encode_X(X, vocabs, reverse_order=True, max_len=50, pad_default_char=' '):

    char_to_int = dict((c, i) for i, c in enumerate(vocabs))

    

    Xenc = list()

    for pattern in X:

        pattern = pattern[:max_len]

        pattern = pattern + (pad_default_char * (max_len - len(pattern)))

        if reverse_order:

            pattern = pattern[::-1]

        integer_encoded = [char_to_int[char] for char in pattern]

        Xenc.append(integer_encoded)

    return Xenc





def text_to_word_sequence_fixed(text):

    list_words = text_to_word_sequence(text)

    return list_words if len(list_words) > 0 else [text]





def integer_encode_Y(y, vocabs, max_len=50, pad_default_char='PAD'):

    idx_mapping = dict((c, i) for i, c in enumerate(vocabs))

    

    yenc = list()

    for pattern in y:

        pattern = text_to_word_sequence_fixed(pattern)[:max_len]

        pattern = pattern + ([pad_default_char] * (max_len - len(pattern)))

        integer_encoded = [idx_mapping[word] for word in pattern]

        yenc.append(integer_encoded)

    return yenc



def one_hot_encode(X, y, vec_size_x, vec_size_y):

    Xenc = list()

    for seq in X:

        pattern = list()

        for index in seq:

            vector = [0 for _ in range(vec_size_x)]

            vector[index] = 1

            pattern.append(vector)

        Xenc.append(pattern)

    yenc = list()

    for seq in y:

        pattern = list()

        for index in seq:

            vector = [0 for _ in range(vec_size_y)]

            vector[index] = 1

            pattern.append(vector)

        yenc.append(pattern)

    return np.array(Xenc), np.array(yenc)





def invert(seq, vocabs, join_char=''):

    idx_mapping = dict((i, c) for i, c in enumerate(vocabs))



    strings = list()

    for pattern in seq:

        string = idx_mapping[argmax(pattern)]

        strings.append(string)

    return join_char.join(strings)





def make_transform_train_data(

        df, X_vocabs, y_vocabs,

        n_in_seq_length, n_out_seq_length,

        n_in_terms, n_out_terms

    ):

    X_small = df['before']

    y_small = df['after']

    

    x_transformed = integer_encode_X(X_small, X_vocabs, max_len=n_in_seq_length)

    y_transformed = integer_encode_Y(y_small, y_vocabs, max_len=n_out_seq_length)

    

    return one_hot_encode(

        x_transformed,

        y_transformed,

        n_in_terms,

        n_out_terms

    )
# Only try out with the CARDINAL

df_filtered = df[df['class'] == 'CARDINAL']

df_filtered[['before', 'after']] = df_filtered[['before', 'after']].astype(str)
X = df_filtered['before']

y = df_filtered['after']



X_vocabs = set([' '])

for words in X:

    X_vocabs.update(list(words))

X_vocabs = [' '] + [X_vocab for X_vocab in list(X_vocabs) if X_vocab != ' ']



y_vocabs = set()

for words in y:

    y_vocabs.update(text_to_word_sequence_fixed(words))

y_vocabs = ['PAD'] + list(y_vocabs)
print('Index of empty word in X %s' % X_vocabs.index(' '))

print('Index of empty word in y %s' % y_vocabs.index('PAD'))
print('Size of vocabs X: %s' % len(X_vocabs))

print('Size of vocabs y: %s' % len(y_vocabs))
n_in_seq_length = np.min([50, len(X_vocabs)])

n_out_seq_length = np.min([50, len(y_vocabs)])



n_in_terms = len(X_vocabs)

n_out_terms = len(y_vocabs)
model = Sequential()

model.add(LSTM(75, input_shape=(n_in_seq_length, n_in_terms)))

model.add(RepeatVector(n_out_seq_length))

model.add(LSTM(50, return_sequences=True))

model.add(TimeDistributed(Dense(n_out_terms, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
%%time

x_transformed, y_transformed = make_transform_train_data(

    df_filtered, X_vocabs, y_vocabs,

    n_in_seq_length, n_out_seq_length,

    n_in_terms, n_out_terms

)
model.fit(x_transformed, y_transformed, epochs=1, batch_size=32)
# Get first 2000 rows for verify the model performance

x_transformed_test, y_transformed_test = make_transform_train_data(

    df_filtered.iloc[:2000], X_vocabs, y_vocabs,

    n_in_seq_length, n_out_seq_length,

    n_in_terms, n_out_terms

)
yhat = model.predict(x_transformed_test, verbose=0)
for idx, yh in enumerate(yhat[:50]):

    yh_inverted = invert(yh, vocabs=y_vocabs, join_char=' ').replace(' PAD', '')

    in_seq = invert(x_transformed_test[idx], X_vocabs).replace(' ', '')[::-1]

    out_seq = invert(y_transformed_test[idx], y_vocabs, join_char=' ').replace(' PAD', '')

    print('%s = %s (%s expect: %s)' % (in_seq, yh_inverted, ('TRUE' if out_seq == yh_inverted else "FALSE"), out_seq))