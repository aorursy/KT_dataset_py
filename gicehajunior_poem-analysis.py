import tensorflow

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow import keras

from tensorflow.keras import layers

import numpy

import os





file_path = "/kaggle/input/collection-of-180-poems/Poems.txt"

models_filepath_to_save = "kaggle/output/working/models"



def read_file(file_path):

    corpus = []

    with open(file_path,"r") as file:

        for line in file.readlines():

            line = line.lower().split('\n')

            

            corpus.append(line)

               

        return corpus

        

        

#tokenization of the corpus

def tokenize_texts(corpus):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(corpus)

    

    return tokenizer



def create_sequences(corpus, tokenizer, total_words):

    input_sequences = []

    # Remove any empty lines

    corpus = [line for line in corpus if line != '']

    for line in corpus:

        #create token list

        token_list = tokenizer.texts_to_sequences([line])[0]

        

        for i in range(1, len(token_list)):

            #create an Ngram sequence

            ngram_sequences = token_list[:i+1]



            input_sequences.append(ngram_sequences)



            maximum_sequence_len = max([len(input_sequence) for input_sequence in input_sequences])

            padded_sequences = numpy.array(pad_sequences(input_sequences, maxlen=maximum_sequence_len, padding='pre'))

            input_sequences = padded_sequences[:,:-1]

            labels = padded_sequences[:,-1]

            hot_encoded_labels = tensorflow.keras.utils.to_categorical(labels, num_classes=total_words)

            return [input_sequences, hot_encoded_labels, maximum_sequence_len]

            #return input_sequences



corpus = read_file(file_path)



tokenizer = tokenize_texts(corpus)



tokenized_corpus = tokenizer.word_index



total_words = len(tokenized_corpus)+1

print(tokenized_corpus)



print(total_words)



padded_sequences = create_sequences(corpus, tokenizer, total_words)                                                

#print(padded_sequences[0])

#print(padded_sequences[1])

print(padded_sequences)











        

        
def create_model():

    model = keras.Sequential()

    model.add(layers.Embedding(total_words, 240, input_length=padded_sequences[2]-1))

    model.add(layers.Bidirectional(layers.LSTM(150)))

    model.add(layers.Dense(total_words, activation='softmax'))

    

    return model



def model_history(model):

    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    #compile the model

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



    #create a fit

    history = model.fit(padded_sequences[0], padded_sequences[1], epochs=100, verbose=1)

    return [history, model]



def save_model(model_path, model):

    if not os.path.exists(model_path):

        os.mkdir(model_path)

    else:

        model.save(model_path)

    



create_model = create_model()



#display the model archtecture

create_model.summary()



history, model = model_history(create_model)

#print(history)

#print(model)



#saving the model

#save_model(models_filepath_to_save, model)

def predict_next_sentiment(model, tokenizer, maximum_sequence_len):

    print("Enter a sample text to test")

    seed_text = "In love with your girl"



    next_words = 30



    for _ in range(next_words):

        # do some tokenization

        token_list = tokenizer.texts_to_sequences([seed_text])[0]



        # do some padding

        token_list = pad_sequences([token_list], maxlen=maximum_sequence_len-1, padding='pre')



        # predicted classes

        predicted = model.predict_classes(token_list, verbose=0)





        output_word = ""

        seed_text = tokenizer.word_index.items()

        for word, index in seed_text:

            if index == predicted:

                output_word = word



                break

            seed_text += " " + output_word

    return seed_text

    

seed_text = predict_next_sentiment(model, tokenizer, padded_sequences[2])

print(seed_text)