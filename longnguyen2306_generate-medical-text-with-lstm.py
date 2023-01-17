import pandas as pd



df = pd.read_csv('../input/mtsamples.csv')



df.head()
from pandarallel import pandarallel



pandarallel.initialize()
del df["Unnamed: 0"]

del df["description"]

del df["medical_specialty"]

del df["sample_name"]

del df["keywords"]



df.to_csv("medical-text/transcription.csv") 

df.to_json("medical-text/transcription.json") 
def process_transcription(text):

    return len(str(text))



df["word_count"] = df["transcription"].parallel_apply(process_transcription)

df.head()
import seaborn as sns



sns.distplot(df["word_count"])
df_samples = df[df["word_count"] < 1000]

df_samples = df_samples[df_samples["word_count"] > 100]



del df_samples["word_count"]



print(len(df_samples))



df_samples.to_csv("medical-text/sampled_transcription.csv") 

df_samples.to_json("medical-text/sampled_transcription.json") 
stop_words = {

    ":": "",

    ",": " ",

    "_": "",

    '"': '',

    '#': '',

    "'": "",

    "-": " ",

    "(": "",

    ")": "",

    "+": "",

    "/": "",

    "*": "",

    "&": "",

    "?": "",

    "@": "",

    "!": "",

    "%": "",   

    "$": "",

    "<": "",

    ">": "",

    "[": "",

    ";": "",

    "=": "",

    "®": "",

    "©": "",

    "°": "",

    "º": "",

    "¼": "",

    "½": "",

    "è": "e",

    "é": "e",

    "ü": "",

    "…": "",

    '–': "",

    '’': "",

    '“': "",

    '”': "",

    "]": "",

    '{': "",

    '}': "",

    'µ': "",

    '·': ""

}



def process_transcription(text):

    text = str(text)

    text = text.lower()

    for k, v in stop_words.items():

        text = text.replace(k, v)

    return text



df["transcription"] = df["transcription"].parallel_apply(process_transcription)

df.to_csv("medical-text/transcription_1_processed.csv")

df.to_json("medical-text/transcription_1_processed.json")

df.head()
def process_transcription(text):

    text = str(text)

    for i in range(0, 10):

        text = text.replace(str(i), "")

        text = " ".join(text.split(" "))

        text = text.strip()

    return text



df["transcription"] = df["transcription"].parallel_apply(process_transcription)

df.to_csv("medical-text/transcription_2_processed.csv")

df.to_json("medical-text/transcription_2_processed.json")

df.head()
import nltk



stem = nltk.stem.WordNetLemmatizer()



def process_transcription(text):

    result = []

    for token in text.split():

        stemmed_token = stem.lemmatize(token)

        result.append(stemmed_token)

    return " ".join(result)



df["transcription"] = df["transcription"].parallel_apply(process_transcription)

df.to_csv("medical-text/transcription_3_processed.csv")

df.to_json("medical-text/transcription_3_processed.json")

df.head()
def process_transcription(text):

    text = text.replace(".", "  .  ")

    return text



df["transcription"] = df["transcription"].parallel_apply(process_transcription)

df.to_csv("medical-text/transcription_4_processed.csv")

df.to_json("medical-text/transcription_4_processed.json")

df.head()
corpus = df.transcription.str.cat()

corpus = " ".join(corpus.split())

corpus[:100]
unique_chars = sorted(list(set(corpus)))

unique_chars
encoded_chars = dict()

decoded_chars = dict()



for index, char in enumerate(unique_chars):

    encoded_chars[char] = index

    decoded_chars[index] = char



with open("medical-text/encoded-chars.json", "w") as f:

    json.dump(encoded_chars, f)



with open("medical-text/decoded_chars.json", "w") as f:

    json.dump(decoded_chars, f)



print(f"There are {len(unique_chars)} unique chars in the corpus")
encoded_corpus = []



for token in corpus:

    encoded_token = encoded_chars[token]

    encoded_corpus.append(encoded_token)



encoded_corpus[:10]
maxlen = 30



step = 3



sentences = []



next_chars = []



for i in range(0, len(encoded_corpus) - maxlen, step):

    sentences.append(encoded_corpus[i:i + maxlen])

    next_chars.append(encoded_corpus[i + maxlen])



print('Number of sequences: ', len(sentences))
from tensorflow.keras.utils import to_categorical

import numpy as np 



x = to_categorical(np.array(sentences))

y = to_categorical(next_chars)



x.shape, y.shape
import tensorflow

from tensorflow.keras import layers



model = tensorflow.keras.models.Sequential()



model.add(layers.LSTM(128, input_shape=(x.shape[1], x.shape[2])))



model.add(layers.Dense(len(encoded_chars), activation='softmax'))



optimizer = tensorflow.keras.optimizers.Adam(lr=0.0001)



model.compile(loss='categorical_crossentropy', optimizer=optimizer)



model.summary()
model.fit(x, y, batch_size=512, epochs=10)
import random

import sys



def sample(preds, temperature = 1.0):

    preds = np.asarray(preds).astype(np.float64)

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)



start_index = random.randint(0, len(corpus) - maxlen - 1)

generated_text = corpus[start_index: start_index + maxlen]

for temperature in [0.95, 0.96, 0.97, 0.98, 0.99, 1]:

    print('\nTemperature ', temperature)

    print(generated_text)

    for i in range(400):

        sampled = np.zeros((1, maxlen, len(encoded_chars)))

        for t, char in enumerate(generated_text):

            sampled[0, t, encoded_chars[char]] = 1.

        preds = model.predict(sampled, verbose=0)[0]

        next_index = sample(preds, temperature)

        next_char = decoded_chars[next_index]



        generated_text += next_char

        generated_text = generated_text[1:]



        sys.stdout.write(next_char)
def process_transcription(text):

    text = " ".join(str(text).split())

    return text



df["transcription"] = df["transcription"].parallel_apply(process_transcription)

df.to_csv("medical-text/transcription_4_processed.csv")

df.to_json("medical-text/transcription_4_processed.json")

df.head()
df_sampled = pd.DataFrame.sample(df, n=100, random_state=42)
corpus = df_sampled.transcription.str.cat()

corpus = " ".join(corpus.split())

corpus = corpus.split()

corpus[:10]
unique_words = sorted(list(set(corpus)))

unique_words[:20]
encoded_words = dict()

decoded_words = dict()



for index, word in enumerate(unique_words):

    encoded_words[word] = index

    decoded_words[index] = word



with open("medical-text/encoded_words.json", "w") as f:

    json.dump(encoded_words, f)



with open("medical-text/decoded_words.json", "w") as f:

    json.dump(decoded_words, f)



print(f"There are {len(unique_words)} unique words in the corpus")
encoded_corpus = []



for token in corpus:

    encoded_token = encoded_words[token]

    encoded_corpus.append(encoded_token)



encoded_corpus[:10]
maxlen = 30



step = 1



sentences = []



next_chars = []



for i in range(0, len(encoded_corpus) - maxlen, step):

    sentences.append(encoded_corpus[i:i + maxlen])

    next_chars.append(encoded_corpus[i + maxlen])



print('Number of sequences: ', len(sentences))
from tensorflow.keras.utils import to_categorical

import numpy as np



x = to_categorical(np.array(sentences))

y = to_categorical(next_chars)



x.shape, y.shape
import tensorflow

from tensorflow.keras import layers



model = tensorflow.keras.models.Sequential()



model.add(layers.LSTM(128, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))



model.add(layers.LSTM(128, input_shape=(x.shape[1], x.shape[2])))



model.add(layers.Dense(len(encoded_words), activation='softmax'))



optimizer = tensorflow.keras.optimizers.Adam(lr=0.0001)



model.compile(loss='categorical_crossentropy', optimizer=optimizer)



model.summary()
model.fit(x, y, batch_size=32, epochs=30)
import random

import sys



def sample(preds, temperature = 1.0):

    preds = np.asarray(preds).astype(np.float64)

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)



def print_sentences(text):

    for word in text:

        sys.stdout.write(" " + word)

        



start_index = random.randint(0, len(corpus) - maxlen - 1)

generated_text = corpus[start_index: start_index + maxlen]



for temperature in [1]:

    print('\nTemperature ', temperature)

    print_sentences(generated_text)

    for i in range(50):

        sampled = np.zeros((1, maxlen, len(encoded_words)))

        for index, char in enumerate(generated_text):

            sampled[0, index, encoded_words[char]] = 1.

            

        preds = model.predict(sampled, verbose=0)[0]

        next_index = sample(preds, temperature)

        next_word = decoded_words[next_index]



        generated_text.append(next_word)

        generated_text = generated_text[1:]



        sys.stdout.write(next_word + " ")