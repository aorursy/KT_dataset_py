import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding # Import layers from Keras
from keras.models import Sequential

import os
print(os.listdir("../input"))
raw_data = pd.read_csv('../input/train.csv', encoding='latin-1') # Read the data as a DataFrame using Pandas
raw_test_data = pd.read_csv('../input/test.csv', encoding='latin-1')

print(raw_data.shape) # Print the dimensions of train DataFrame
print(raw_data.columns) # Print the column names of the DataFrame
print('\n')
raw_data.head(5) # Print the top few records
# Print the unique classes and their counts/frequencies
classes = np.unique(raw_data['target'], return_counts=True) # np.unique returns a tuple with class names and counts
print(classes[0]) #Print the list of unique classes
print(classes[1]) #Print the list of frequencies of the above classes02155
max_num_words = 10000
seq_len = 150
embedding_size = 100

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=max_num_words) #Tokenizer is used to tokenize text
tokenizer.fit_on_texts(raw_data.question_text) #Fit this to our corpus

x_train = tokenizer.texts_to_sequences(raw_data.question_text) #'text to sequences converts the text to a list of indices
x_train = pad_sequences(x_train, maxlen=150) #pad_sequences makes every sequence a fixed size list by padding with 0s 


x_test = tokenizer.texts_to_sequences(raw_test_data.question_text) 
x_test = pad_sequences(x_test, maxlen=150)

x_train.shape, x_test.shape # Check the dimensions of x_train and x_test  
unique_labels = list(raw_data.target.unique())
print(unique_labels)
# Building an LSTM model
model = Sequential() # Call Sequential to initialize a network
model.add(Embedding(input_dim = max_num_words, 
                    input_length = seq_len, 
                    output_dim = embedding_size)) # Add an embedding layer which represents each unique token as a vector
model.add(LSTM(10, return_sequences=False)) # Add an LSTM layer ( will not return output at each step)
model.add(Dense(1, activation='sigmoid')) # Add an ouput layer. Since classification, 3 nodes for 3 classes.
model.summary()
from keras.optimizers import Adam
adam = Adam(lr=0.001)
# Mention the optimizer, Loss function and metrics to be computed
model.compile(optimizer=adam,                  # 'Adam' is a variant of gradient descent technique
              loss='binary_crossentropy', # categorical_crossentropy for multi-class classification
              metrics=['accuracy'])            # These metrics are computed for evaluating and stored in history
y_train = raw_data['target']
model.fit(x_train, y_train, epochs=1, validation_split=0.25)
preds = model.predict(x_test)
pred_test_y = (preds>0.35).astype(int)

# Read the submission file
submission=pd.read_csv("../input/sample_submission.csv")

# Fill the is_pass variable with the predictions
submission['prediction']= pd.DataFrame(pred_test_y)

# Converting the submission file to csv format
submission.to_csv('submission.csv', index=False)
