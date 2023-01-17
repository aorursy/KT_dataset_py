!wget https://zenodo.org/record/841984/files/wili-2018.zip
!unzip wili-2018.zip
import numpy as np

import pandas as pd

import re

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing import sequence

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
with open('x_train.txt') as f:

  data = f.read()



data_x_train = data.split('\n')

print(len(data_x_train))



with open('x_test.txt') as f:

  data = f.read()



data_x_test = data.split('\n')

print(len(data_x_test))



with open('y_train.txt') as f:

  data = f.read()



data_y_train = data.split('\n')

print(len(data_y_train))



with open('y_test.txt') as f:

  data = f.read()



data_y_test = data.split('\n')

print(len(data_y_test))
data_x_train.pop(-1)

data_x_test.pop(-1)

data_y_train.pop(-1)

data_y_test.pop(-1)
print(len(data_x_train))

print(len(data_x_test))

print(len(data_y_train))

print(len(data_y_test))
data_x_train  = pd.DataFrame(data_x_train,columns=['sentence'])

data_y_train  = pd.DataFrame(data_y_train,columns=['language'])



data_x_test   = pd.DataFrame(data_x_test,columns=['sentence'])

data_y_test   = pd.DataFrame(data_y_test,columns=['language'])

def process_sentence(sentence):

    '''Removes all special characters from sentence. It will also strip out

    extra whitespace and makes the string lowercase.

    '''

    

    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|]', '', sentence.lower().strip())
x_train = data_x_train['sentence'].apply(process_sentence)

y_train = data_y_train['language']



x_test = data_x_test['sentence'].apply(process_sentence)

y_test = data_y_test['language']
print(len(x_train))

print(len(y_train))
elements =  set((''.join([element for element in x_train])).split())
languages = set(y_train)
print("Total Words       :    " + str(len(elements)))

print("Total Languages   :    " + str(len(languages)) )
def create_lookup_table(text):

    """Create lookup tables for vocabulary

    """

    

    text_to_int = { word : i for i, word in enumerate(text)}

    int_to_text = {   v  : k for k, v in text_to_int.items()}

    

    return text_to_int, int_to_text
elements.add("<UNK>")
elements_to_int, int_to_elements = create_lookup_table(elements)

languages_to_int, int_to_languages = create_lookup_table(languages)
def encode_to_int(data, data_to_int):

    """Converts all our text to integers

    """

    encoded_items = []

    for sentence in data: 

        encoded_items.append([data_to_int[word] if word in data_to_int else data_to_int["<UNK>"] for word in sentence.split()])

    

    return encoded_items
x_train_encoded = encode_to_int(x_train, elements_to_int)

x_test_encoded = encode_to_int(x_test, elements_to_int)
print(elements_to_int['<UNK>'])
print(x_test_encoded[1111])
y_train_encoded = OneHotEncoder().fit_transform(encode_to_int(y_train, languages_to_int)).toarray()

y_test_encoded = OneHotEncoder().fit_transform(encode_to_int(y_test, languages_to_int)).toarray()
max_sentence_length = 200

embedding_vector_length = 300

dropout = 0.5

x_train_pad = sequence.pad_sequences(x_train_encoded, maxlen=max_sentence_length)

x_test_pad = sequence.pad_sequences(x_test_encoded, maxlen=max_sentence_length)
print(x_test_pad[0])
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():

    model = Sequential()



    model.add(Embedding(len(elements_to_int), embedding_vector_length, input_length=max_sentence_length))

    model.add(LSTM(256, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))

    model.add(LSTM(256, dropout=dropout, recurrent_dropout=dropout))

    model.add(Dense(len(languages), activation='softmax'))



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    #checkpointer = ModelCheckpoint(filepath='language_model.hdf5', monitor = "val_loss", verbose=1, save_best_only=True, mode = 'auto')



    print(model.summary())
print(len(x_train_encoded))

print(len(y_train_encoded))
scores = []





for i in range(0,15):

  print("\tEpoch  :  "+str(i))

  score = model.fit(x_train_pad, y_train_encoded, epochs=1, batch_size=256, validation_data=(x_test_pad, y_test_encoded ))

  scores.append(score)

  



train_loss = []

validation_loss = []



for history in scores:

  train_loss.append(history.history['loss'])

  validation_loss.append(history.history['val_loss'])





plt.plot(train_loss)

plt.plot(validation_loss)

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



del train_loss

del validation_loss
model
def predict_sentence(sentence):

    """Converts the text and sends it to the model for classification

    """

    

    # Clean the sentence

    sentence = process_sentence(sentence)

    # Transform and pad it before using the model to predict

    x = np.array(encode_to_int([sentence], elements_to_int))

    #print(x)

    x = sequence.pad_sequences(x, maxlen=max_sentence_length)

    

    #print(x.shape)

    

    prediction = model.predict(x)

    #print(prediction)

    

    # Get the highest prediction

    lang_index = np.argmax(prediction)

    #print(lang_index)

    #print(prediction[0][lang_index])

    #print(prediction[0][93])

    return int_to_languages[lang_index]
predict_sentence('बांग्लादेश के मुख्य न्यायाधीश का पद, बांग्लादेश सर्वोच्च न्यायिक पद है। इस पद पर विराजमान होने वाले पहले पदाधिकारी न्यायमूर्ति अब सादात मोहम्मह खान सयम थे, जोकि १६ दिसंबर १९७२ से नवंबर १९७५ तक इस पद पर रहे थे। तत्पश्चात, जनवरी २०१५ तक इस पद पर कुल २१ लोग विराजमान हो चुके हैं। वर्तमान मुख्य न्यायाधीश सुरेन्द्र कुमार सिन्हा इस पद पर १७ जनवरी २०१५ विराजमान हैं। वे हिन्दू धर्म के अनुयायी हैं, तथा बिष्णुप्रिय मणिपुरी समुदाय से आते हैं, तथा वे बांग्लादेश में किसी भी अल्पसंख्यक जातीय समूहों से नियुक्त पहली मुख्य न्यायाधीश है। न्यायमूर्ति भावनी प्रसाद सिन्हा भी एक ही समुदाय से हैं। न्यायमूर्ति महौदय नाज़मन आरा सुल्ताना इस पद को सुशोभित करने वाली पहली महिला न्यायाधीश थीं, और मैडम जस्टिस कृष्णा देबनाथ बांग्लादेश की पहली महिला हिंदू न्यायाधीश है। वर्तमान में सुप्रीम कोर्ट में छह महिला न्यायाधीशों रहे हैं।')
predict_sentence('Ne l fin de l seclo XIX l Japon era inda çconhecido i sótico pa l mundo oucidental. Cula antroduçon de la stética japonesa, particularmente na Sposiçon Ounibersal de 1900, an Paris, l Oucidente adquiriu un apetite ansaciable pul Japon i Heiarn se tornou mundialmente coincido pula perfundidade, ouriginalidade i sinceridade de ls sous cuntos. An sous radadeiros anhos, alguns críticos, cumo George Orwell, acusórun Heiarn de trasferir sou nacionalismo i fazer l Japon parecer mais sótico, mas, cumo loufereciu al Oucidente alguns de sous purmeiros lampeijos de l Japon pré-andustrial i de l Período Meiji, sou trabalho inda ye balioso até hoije.')