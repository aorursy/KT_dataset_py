import pandas as pd
import gc
f=open("../input/review.txt",'r',encoding='latin-1')
#f.seek()
s=f.read()
del f
gc.collect()
print("number of character ",len(s))
no_review=s.split("\n")
del s
gc.collect()
print("number of review",len(no_review))
train=no_review[0:100]
#test=no_review[700:1000]
del no_review
gc.collect()
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
tokenizer = Tokenizer()

def dataset_preparation(data):
	#corpus = data.lower().split("\n")    
	tokenizer.fit_on_texts(data)
	total_words = len(tokenizer.word_index) + 1

	input_sequences = []
	for line in data:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)

	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
	label = ku.to_categorical(label, num_classes=total_words)
	return predictors, label, max_sequence_len, total_words

def create_model(predictors, label, max_sequence_len, total_words):
	input_len = max_sequence_len - 1
	model = Sequential()
	model.add(Embedding(total_words, 10, input_length=input_len))
	model.add(LSTM(100))
	model.add(Dropout(0.1))
	model.add(Dense(total_words, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.fit(predictors, label, epochs=50, verbose=1)
	return model

def generate_text(seed_text, next_words, max_sequence_len, model):
	for j in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen= 
							 max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text

def create_train_and_test_data(file):
    with open(file, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        num_lines = len(lines)
        training_len = int(0.8*num_lines)
        training_data = ''.join(lines[:training_len])
        testing_data = lines[training_len:]
    return training_data, testing_data

pred,labels,max_len,total_words=dataset_preparation(train)
print(max_len,total_words)
del train 
gc.collect()
model=create_model(pred, labels, max_len,total_words)
model.save("lstm_model_1k.h5")
