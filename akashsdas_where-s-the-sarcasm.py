import re



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer
filename = '/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'

filename2 = '/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json'
df1 = pd.read_json(filename, lines=True).drop('article_link', axis=1)

df1.head()
df2 = pd.read_json(filename2, lines=True).drop('article_link', axis=1)

df2.head()
df = pd.concat([df1, df2]).reset_index(drop=True)

df.head()
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





def clean_contractions(text, contractions):

    text_list = text.split()

    for idx, word in enumerate(text_list):

        try:

            text_list[idx] = contractions[word]

        except KeyError:

            continue

    return ' '.join(text_list)





df.headline = df.headline.apply(lambda x: clean_contractions(x, contractions))

df.head()
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



    return text



df.headline = df.headline.apply(lambda x: clean_text(x))

df.head()
vocab_size = 10000

oov_token = '<UNK>'
X_train, X_val, Y_train, Y_val = train_test_split(

    df[['headline']], df[['is_sarcastic']], test_size=0.1, random_state=0, shuffle=True

)



X_train.reset_index(drop=True, inplace=True)

X_val.reset_index(drop=True, inplace=True)

Y_train.reset_index(drop=True, inplace=True)

Y_val.reset_index(drop=True, inplace=True)
training_sentences = X_train.headline.tolist()

val_sentences = X_val.headline.tolist()
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
headline_word_count = [len(sentence.split()) for sentence in df.headline]

headline_word_count[:10]
# Analyze the distribution of sequences by looking at the length of the texts

pd.DataFrame({'articles': headline_word_count}).hist(

    bins=20, figsize=(20, 8), range=[0, 20]

)

plt.show()
max_length = 11

padding_type = 'post'

truncating_type = 'post'
training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_padded = pad_sequences(

    training_sequences, maxlen=max_length, padding=padding_type, truncating=truncating_type

)



val_sequences = tokenizer.texts_to_sequences(val_sentences)

val_padded = pad_sequences(

    val_sequences, maxlen=max_length, padding=padding_type, truncating=truncating_type

)



training_sequences[0]
embedding_dim = 128

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss = 'binary_crossentropy'

metrics = ['accuracy']

num_epochs = 10



callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.1, patience=1, min_lr=0.000001, verbose=1),

]
def build_model(vocab_size, embedding_dim):

    model = tf.keras.Sequential([

        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

        layers.GlobalAveragePooling1D(),

        layers.Dense(vocab_size, activation='relu'),

        layers.Dense(1, activation='sigmoid')

    ])

    return model





def build_compiled_model(vocab_size, embedding_dim, loss, optimizer, metrics):

    model = build_model(vocab_size, embedding_dim)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model



model = build_compiled_model(

    vocab_size, embedding_dim, loss, optimizer, metrics

)
history = model.fit(

    training_padded, Y_train, epochs=num_epochs,

    validation_data=(val_padded, Y_val), callbacks=callbacks,

    verbose=1

)
# Accuracy



plt.plot(history.history['accuracy'][1:], label='train acc')

plt.plot(history.history['val_accuracy'][1:], label='validation acc')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')
# Loss



plt.plot(history.history['loss'][1:], label='train loss')

plt.plot(history.history['val_loss'][1:], label='validation loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc='lower right')
# evaluating on validation data

print(model.evaluate(val_padded, Y_val))
predictions = model.predict_classes(val_padded)

for i in range(10):

    print(f'# {i+1} Headline: {X_val.iloc[i].values[0]}')

    print(f'Prediction: {predictions[i]}')

    print(f'Actual: { Y_val.iloc[i].values[0]}')

    print()
# Testing on unseen data



sentences = [

    'The whether today is bright and sunny.',

    'Not all men are annoying. Some are dead.',

    'yellowstone park attempts to increase ranger population with new mating program',

    'If you’re going to tell people the truth, be funny or they’ll kill you.',

    'He who laughs last didn’t get the joke'

]



sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(

    sequences, maxlen=max_length, padding=padding_type, truncating=truncating_type

)



predictions = model.predict_classes(padded)

print(predictions)
model.save('model')       # SavedModel format

model.save('model.h5')    # HDF5 format