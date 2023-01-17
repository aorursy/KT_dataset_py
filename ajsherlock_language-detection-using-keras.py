from keras.models import load_model

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

import keras.optimizers

from ipywidgets import interact_manual

from ipywidgets import widgets

import pickle

import numpy as np

import re
import os

print(os.listdir("../input"))
def define_alphabet():

    base_en = 'abcdefghijklmnopqrstuvwxyz'

    special_chars = ' !?¿¡'

    german = 'äöüß'

    italian = 'àèéìíòóùú'

    french = 'àâæçéèêêîïôœùûüÿ'

    spanish = 'áéíóúüñ'

    czech = 'áčďéěíjňóřšťúůýž'

    slovak = 'áäčďdzdžéíĺľňóôŕšťúýž'

    all_lang_chars = base_en + german +  italian + french + spanish + czech + slovak

    small_chars = list(set(list(all_lang_chars)))

    small_chars.sort() 

    big_chars = list(set(list(all_lang_chars.upper())))

    big_chars.sort()

    small_chars += special_chars

    letters_string = ''

    letters = small_chars + big_chars

    for letter in letters:

        letters_string += letter

    return small_chars,big_chars,letters_string
def get_sample_text(file_content,start_index,sample_size):



    while not (file_content[start_index].isspace()):

        start_index += 1

    while file_content[start_index].isspace():

        start_index += 1

    end_index = start_index+sample_size 

    while not (file_content[end_index].isspace()):

        end_index -= 1

    return file_content[start_index:end_index]
def get_input_row(content,start_index,sample_size, alphabet):

    sample_text = get_sample_text(content,start_index,sample_size)

    counted_chars_all = count_chars(sample_text.lower(), alphabet[0])

    counted_chars_big = count_chars(sample_text, alphabet[1])

    all_parts = counted_chars_all + counted_chars_big

    return all_parts
def remove_xml(text):

    return re.sub(r'<[^<]+?>', '', text)



def remove_newlines(text):

    return text.replace('\n', ' ') 

    



def remove_manyspaces(text):

    return re.sub(r'\s+', ' ', text)



def clean_text(text):

    text = remove_xml(text)

    text = remove_newlines(text)

    text = remove_manyspaces(text)

    return text
def count_chars(text, alphabet):

    alphabet_counts = []

    for letter in alphabet:

        count = text.count(letter)

        alphabet_counts.append(count)

    return alphabet_counts
# Load the Alphabet

alphabet = define_alphabet()

LANGUAGES_DICT = {'en':0,'fr':1,'es':2,'it':3,'de':4,'sk':5,'cs':6}

LABELS =  list(LANGUAGES_DICT.keys())

# Length of cleaned text used for training and prediction - 140 chars

MAX_LEN = 140



# number of language samples per language that we will extract from source files

NUM_SAMPLES = 250000





with open("../input/input_size.sav", "rb") as file_obj:

    input_size = pickle.load(file_obj)



with open("../input/standard_scaler.sav", "rb") as file_obj:

    standard_scaler = pickle.load(file_obj)

    

def get_prediction(TEXT):

    #if len(TEXT) < MAX_LEN:

    #    print("Text has to be at least {} chars long, but it is {}/{}".format(MAX_LEN, len(TEXT), MAX_LEN))

    #    return(-1)

    # Data cleaning

    cleaned_text = clean_text(TEXT)

    temp_text=cleaned_text.split(' ')

    if len(temp_text)<MAX_LEN:

        count=MAX_LEN-len(temp_text)

        for i in range(0,count):

            temp_text.append(" unk ")

        cleaned_text=' '.join(temp_text)

    

    # Get the MAX_LEN char

    input_row = get_input_row(cleaned_text, 0, MAX_LEN, alphabet)

    

    # Data preprocessing (Standardization)

    test_array = standard_scaler.transform([input_row])

    

    raw_score = model.predict(test_array)

    pred_idx= np.argmax(raw_score, axis=1)[0]

    score = raw_score[0][pred_idx]*100

    

    # Prediction

    prediction = LABELS[model.predict_classes(test_array)[0]]

    print('TEXT:', TEXT, '\nPREDICTION:', prediction.upper(), '\nSCORE:', score)







model = Sequential()

# Note: glorot_uniform is the Xavier uniform initializer.



model.add(Dense(500,input_dim=input_size, kernel_initializer="glorot_uniform", activation="sigmoid"))

model.add(Dropout(0.5))

model.add(Dense(300, kernel_initializer="glorot_uniform", activation="sigmoid"))

model.add(Dropout(0.5))

model.add(Dense(100, kernel_initializer="glorot_uniform", activation="sigmoid"))

model.add(Dropout(0.5))

model.add(Dense(len(LANGUAGES_DICT), kernel_initializer="glorot_uniform", activation="softmax"))

model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',

              optimizer=model_optimizer,

              metrics=['accuracy'])



model.summary()

model.load_weights('../input/lang_identification_weights.h5')





print(get_prediction("This is a sample text in English"))

print(get_prediction("Ceci est un exemple de texte en anglais"))

print(get_prediction("Dies ist ein Beispieltext in Englisch"))
