import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

submission = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
data = train



texts = data['text']

test_data = data[:1400]

train_data = data[1400:]

test_data.sample(4)

trainX = train_data['text']

trainY = train_data['target']



testX = train_data['text']

testY = train_data['target']



testY.sample(5)
import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = 12800, oov_token = "<OOV/>")

tokenizer.fit_on_texts(trainX)

word_index = tokenizer.word_index

len(word_index)
train_sequences = tokenizer.texts_to_sequences(trainX)

train_padded = pad_sequences(train_sequences)



test_sequences = tokenizer.texts_to_sequences(testX)

test_padded = pad_sequences(test_sequences)



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(12800, 128),

    #tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(36, activation="relu"),

    tf.keras.layers.Dense(12, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")

   

])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



history = model.fit(np.array(train_padded), np.array(trainY), 

                    epochs = 10, 

                    validation_data=(np.array(test_padded), np.array(testY)))
to_predict = ["@sakuma_en If you pretend to feel a certain way the feeling can become genuine all by accident. -Hei (Darker than Black) #manga #anime",

             "I collapsed in the bathroom bcuz of Michael.", 

             "We had a room full of people lifting up Collide in prayer!! We are so excited for Friday night!! http://t.co/645dYNAMy8",

              "Motorcyclist bicyclist injured in Denver collision on Broadway: At least two people were taken to a localÂ‰Ã›_ http://t.co/PMv8ZDFnmr",

              "Traffic Collision - Ambulance Enroute: Florin Rd at Franklin Blvd South Sac http://t.co/dYEl9nMQ0A",            

              "MicrosoftÂ‰Ã›Âªs Nokia acquisition was an even bigger disaster than we ever imagined https://t.co/4MneTInGXl",

                "Not an electric debut for Severino but not a disaster either. Looking forward to see what adjustments he makes for start #2.",

                "This Palestinian family was lucky but hundreds are feared dead in the latest boat disaster in the Mediterranean",

            "https://t.co/cT5v3LcNKD",

            "Heat wave adding to the misery of internally-displaced Gazans http://t.co/jW3hN9ewFT via @PressTV http://t.co/NYWrkRQ7Kn"



]

padded_prediction = pad_sequences(tokenizer.texts_to_sequences(to_predict))

model.predict(padded_prediction)
testX = submission['text']



testX.sample(11)

test_padded = pad_sequences(tokenizer.texts_to_sequences(testX))

submission['target'] = np.round(model.predict(test_padded)).astype(int)



#print(submission[['text', 'target']].sample(22))

   



final = submission[['id', 'target']]



        
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(final)



# create a link to download the dataframe

create_download_link(df)