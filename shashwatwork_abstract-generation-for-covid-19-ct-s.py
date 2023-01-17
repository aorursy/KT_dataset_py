%pip install stylecloud -q
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping

from keras.models import Sequential

import keras.utils as ku 

from keras.utils.vis_utils import plot_model





# set seeds for reproducability

from numpy.random import seed

seed(1)



import pandas as pd

import numpy as np

import string, os 



import warnings

import stylecloud

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)

curr_dir = '../input/../input/covid19-publicationsdatasets-clinical-trials/'

ct_df = pd.read_csv(curr_dir + 'Clinical_Trails.csv')

ct_df.head()
all_trails = []

all_trails.extend(list(ct_df.Abstract.values))

all_trails = [h for h in all_trails if h != "Unknown"]

len(all_trails)
all_trails
def clean_text(txt):

    txt = "".join(v for v in txt if v not in string.punctuation).lower()

    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt 



corpus = [clean_text(x) for x in all_trails]

corpus[:10]
unique_hashtags=(" ").join(corpus)

stylecloud.gen_stylecloud(text = unique_hashtags,

                          icon_name="fas fa-head-side-mask",

                          background_color='white',

                          gradient='horizontal')

from IPython.display import Image

Image(filename='stylecloud.png',width = 500, height = 600) 
tokenizer = Tokenizer()



def get_sequence_of_tokens(corpus):

    ## tokenization

    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    

    ## convert data to sequence of tokens 

    input_sequences = []

    for line in corpus:

        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):

            n_gram_sequence = token_list[:i+1]

            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words



inp_sequences, total_words = get_sequence_of_tokens(corpus)

inp_sequences[:5]
def generate_padded_sequences(input_sequences):

    max_sequence_len = max([len(x) for x in input_sequences])

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len



predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
def create_model(max_sequence_len, total_words):

    input_len = max_sequence_len - 1

    model = Sequential()

    

    model.add(Embedding(total_words, 10, input_length=input_len))



    

    model.add(LSTM(100))



    model.add(Dropout(0.1))

    

    model.add(Dense(total_words, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='adam')

    

    return model



model = create_model(max_sequence_len, total_words)

model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.fit(predictors, label, epochs=100, verbose=1)
def generate_text(seed_text, next_words, model, max_sequence_len):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = model.predict_classes(token_list, verbose=0)

        

        output_word = ""

        for word,index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break

        seed_text += " "+output_word

    return seed_text.title()
print (generate_text("united states", 20, model, max_sequence_len))

print('*' * 100)

print (generate_text("clinical trails", 20, model, max_sequence_len))

print('*' * 100)



print (generate_text("donald trump", 20, model, max_sequence_len))

print('*' * 100)



print (generate_text("india and china", 20, model, max_sequence_len))

print('*' * 100)



print (generate_text("virus", 20, model, max_sequence_len))

print('*' * 100)



print (generate_text("covid", 20, model, max_sequence_len))

print('*' * 100)



print (generate_text("treatment", 20, model, max_sequence_len))

print('*' * 100)



print (generate_text("coronavirus", 20, model, max_sequence_len))

print('*' * 100)