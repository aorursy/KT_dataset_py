import re



import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
filename = '/kaggle/input/shakespeare-plays/alllines.txt'

size_to_read = 500_000  # actual length of text file 4583798





def get_text_data(filename, size_to_read):

    with open(filename, 'r') as f:

        content = f.read(size_to_read)

    return content





text = get_text_data(filename, size_to_read)

print(text[:20])
corpus = text.lower().split('\n')

print(corpus[:10])
contractions = {

    "ain't": "am not / are not / is not / has not / have not",

    "aren't": "are not / am not",

    "can't": "cannot",

    "can't've": "cannot have",

    "'cause": "because",

    "could've": "could have",

    "couldn't": "could not",

    "couldn't've": "could not have",

    "didn't": "did not",

    "doesn't": "does not",

    "don't": "do not",

    "hadn't": "had not",

    "hadn't've": "had not have",

    "hasn't": "has not",

    "haven't": "have not",

    "he'd": "he had / he would",

    "he'd've": "he would have",

    "he'll": "he shall / he will",

    "he'll've": "he shall have / he will have",

    "he's": "he has / he is",

    "how'd": "how did",

    "how'd'y": "how do you",

    "how'll": "how will",

    "how's": "how has / how is / how does",

    "I'd": "I had / I would",

    "I'd've": "I would have",

    "I'll": "I shall / I will",

    "I'll've": "I shall have / I will have",

    "I'm": "I am",

    "I've": "I have",

    "isn't": "is not",

    "it'd": "it had / it would",

    "it'd've": "it would have",

    "it'll": "it shall / it will",

    "it'll've": "it shall have / it will have",

    "it's": "it has / it is",

    "let's": "let us",

    "ma'am": "madam",

    "mayn't": "may not",

    "might've": "might have",

    "mightn't": "might not",

    "mightn't've": "might not have",

    "must've": "must have",

    "mustn't": "must not",

    "mustn't've": "must not have",

    "needn't": "need not",

    "needn't've": "need not have",

    "o'clock": "of the clock",

    "oughtn't": "ought not",

    "oughtn't've": "ought not have",

    "shan't": "shall not",

    "sha'n't": "shall not",

    "shan't've": "shall not have",

    "she'd": "she had / she would",

    "she'd've": "she would have",

    "she'll": "she shall / she will",

    "she'll've": "she shall have / she will have",

    "she's": "she has / she is",

    "should've": "should have",

    "shouldn't": "should not",

    "shouldn't've": "should not have",

    "so've": "so have",

    "so's": "so as / so is",

    "that'd": "that would / that had",

    "that'd've": "that would have",

    "that's": "that has / that is",

    "there'd": "there had / there would",

    "there'd've": "there would have",

    "there's": "there has / there is",

    "they'd": "they had / they would",

    "they'd've": "they would have",

    "they'll": "they shall / they will",

    "they'll've": "they shall have / they will have",

    "they're": "they are",

    "they've": "they have",

    "to've": "to have",

    "wasn't": "was not",

    "we'd": "we had / we would",

    "we'd've": "we would have",

    "we'll": "we will",

    "we'll've": "we will have",

    "we're": "we are",

    "we've": "we have",

    "weren't": "were not",

    "what'll": "what shall / what will",

    "what'll've": "what shall have / what will have",

    "what're": "what are",

    "what's": "what has / what is",

    "what've": "what have",

    "when's": "when has / when is",

    "when've": "when have",

    "where'd": "where did",

    "where's": "where has / where is",

    "where've": "where have",

    "who'll": "who shall / who will",

    "who'll've": "who shall have / who will have",

    "who's": "who has / who is",

    "who've": "who have",

    "why's": "why has / why is",

    "why've": "why have",

    "will've": "will have",

    "won't": "will not",

    "won't've": "will not have",

    "would've": "would have",

    "wouldn't": "would not",

    "wouldn't've": "would not have",

    "y'all": "you all",

    "y'all'd": "you all would",

    "y'all'd've": "you all would have",

    "y'all're": "you all are",

    "y'all've": "you all have",

    "you'd": "you had / you would",

    "you'd've": "you would have",

    "you'll": "you shall / you will",

    "you'll've": "you shall have / you will have",

    "you're": "you are",

    "you've": "you have"

}





def clean_contractions(corpus, contractions):

    for i, text in enumerate(corpus):

        text_list = text.split()

        for j, word in enumerate(text_list):

            try:

                text_list[j] = contractions[word]

            except KeyError:

                continue

        corpus[i] = ' '.join(text_list)

    return corpus





corpus = clean_contractions(corpus, contractions)

print(corpus[:10])
def clean_text(text):

    text = re.sub("(\\t)", ' ', str(text)).lower()

    text = re.sub("(\\r)", ' ', str(text)).lower()

    text = re.sub("(\\n)", ' ', str(text)).lower()



    text = re.sub("(__+)", ' ', str(text)).lower()

    text = re.sub("(--+)", ' ', str(text)).lower()

    text = re.sub("(~~+)", ' ', str(text)).lower()

    text = re.sub("(\+\++)", ' ', str(text)).lower()

    text = re.sub("(\.\.+)", ' ', str(text)).lower()



    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(text)).lower()



    text = re.sub("(mailto:)", ' ', str(text)).lower()

    text = re.sub(r"(\\x9\d)", ' ', str(text)).lower()

    text = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(text)).lower()

    text = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)",

                  'CM_NUM', str(text)).lower()



    text = re.sub("(\.\s+)", ' ', str(text)).lower()

    text = re.sub("(\-\s+)", ' ', str(text)).lower()

    text = re.sub("(\:\s+)", ' ', str(text)).lower()

    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()



    try:

        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(text))

        repl_url = url.group(3)

        text = re.sub(

            r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(text))

    except:

        pass



    text = re.sub("(\s+)", ' ', str(text)).lower()

    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()



    text = ' '.join(text.split())



    return text





def clean_corpus(corpus):

    for i, text in enumerate(corpus):

        corpus[i] = clean_text(text)

    return corpus





corpus = clean_corpus(corpus)

print(corpus[:10])
oov_token = '<UNK>'



tokenizer = Tokenizer(oov_token=oov_token)

tokenizer.fit_on_texts(corpus)



tokenizer.word_index



total_words = len(tokenizer.word_index) + 1  # +1 for oov

print(total_words)
def get_input_sequences(corpus, tokenizer):

    input_sequences = []



    for line in corpus:

        tokens_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(tokens_list)):

            n_gram_sequence = tokens_list[:i+1]

            input_sequences.append(n_gram_sequence)



    return input_sequences





input_sequences = get_input_sequences(corpus, tokenizer)

print(input_sequences[1])
# getting the max len of among all sequences

max_sequence_len = max([len(x) for x in input_sequences])

print(max_sequence_len)
# padding the input sequence

padded_input_sequences = np.array(

    pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

)

print(padded_input_sequences[1])
# shuffling

np.random.shuffle(padded_input_sequences)
x = padded_input_sequences[:, :-1]

labels = padded_input_sequences[:, -1]

y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

print(x[1])

print(labels[1])

print(y[1])
lstm_units = 512

embedding_dim = 240





def build_model(total_words, max_sequence_len, lstm_units=128, embedding_dim=16):

    model = tf.keras.Sequential()

    

    model.add(Embedding(total_words, embedding_dim, input_length=max_sequence_len-1, trainable=True)) 

    model.add(Bidirectional(LSTM(lstm_units)))

    model.add(Dense(total_words, activation='softmax'))

    

    return model





model = build_model(total_words, max_sequence_len, lstm_units, embedding_dim)



model.summary()
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="loss", factor=0.1, patience=2, min_lr=0.000001, verbose=1),

]



optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)



model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(x, y, epochs=20, callbacks=callbacks, verbose=1)
# Accuracy



plt.plot(history.history['accuracy'][1:], label='train acc')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')
# Loss



plt.plot(history.history['loss'][1:], label='train loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc='lower right')
def predict_next(model, text, tokenizer, max_sequence_len, num_of_words=10):

    # predict next num_of_words for text

    for _ in range(num_of_words):

        input_sequences = tokenizer.texts_to_sequences([text])[0]

        padded_input_sequences = pad_sequences(

            [input_sequences], maxlen=max_sequence_len-1, padding='pre'

        )

        predicted = model.predict_classes(padded_input_sequences, verbose=0)

        output_word = ''



        for word, index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break



        text += ' ' + output_word



    return text
seed_text = 'The sky is'

print(predict_next(model, seed_text, tokenizer, max_sequence_len, num_of_words=50))
seed_text = 'Everything is fair in love and'

print(predict_next(model, seed_text, tokenizer, max_sequence_len, num_of_words=50))
seed_text = 'My life'

print(predict_next(model, seed_text, tokenizer, max_sequence_len, num_of_words=50))
seed_text = 'You are a type of guy that'

print(predict_next(model, seed_text, tokenizer, max_sequence_len, num_of_words=50))
model.save('model')       # SavedModel format

model.save('model.h5')    # HDF5 format