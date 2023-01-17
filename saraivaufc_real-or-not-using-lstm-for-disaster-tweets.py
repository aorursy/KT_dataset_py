! pip3 install textaugment
import re

import pandas as pd
import numpy as np
import csv
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
train_file = '/kaggle/input/nlp-getting-started/train.csv'
max_sequence_length = 32
max_words = 3000
embedding_size = 32
model_file = '/kaggle/working/model.h5'
tokenizer_file = '/kaggle/working/tokenizer.pickle'
num_classes = 2
def clean_str(string):
    string = re.sub(r'http\S+', 'link', string) # replace links by generic text link
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    cleanr = re.compile('<.*?>')

    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')

    return string.strip().lower()


cleaned_str = clean_str('Horrible Accident | Man Died In Wings of AirplaneåÊ(29-07-2015) http://t.co/wq3wJsgPHL')
cleaned_str
stop_words = set(stopwords.words('english'))

def remove_stopwords(word_list):
    no_stop_words = [w for w in word_list if not w in stop_words]
    return no_stop_words


remove_stopwords(cleaned_str.split(" "))
# Import and Create an EDA object
from textaugment import EDA

t = EDA()
for i in range(3):
    print(t.random_deletion('The pastor was not in the scene of the accident... who was the owner of the range rover?', p=0.2))
for i in range(3):
    print(t.random_swap('The pastor was not in the scene of the accident... who was the owner of the range rover?'))
for i in range(3):
    print(t.synonym_replacement('The pastor was not in the scene of the accident... who was the owner of the range rover?'))
for i in range(3):
    print(t.random_insertion('The pastor was not in the scene of the accident... who was the owner of the range rover?'))
data = pd.read_csv(train_file, sep=',', header=0, quotechar='"')

data = data[['text', 'target']]
data
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data.target)
plt.xlabel('Target')
plt.title('Number of disaster tweets')
# text cleaning
data['text'] = data['text'].apply(lambda x: clean_str(x))

sequences = []
targets = []

for index, row in data.iterrows():
    seqs = []
    text = row['text']

    # if empty text, skipping to next row data
    if not text:
        continue

    seqs.append(text)
    
    # apply data augmentation
    
    # random deletion
    seq2 = t.random_deletion(text, p=0.2)
    if type(seq2) == type([]):
        seqs.append(seq2[0])
    else:
        seqs.append(seq2)

    # random swap
    if len(text) > 1:
        seqs.append(t.random_swap(text))

    # synonym replacement and random insertion
    for i in range(2):
        seqs.append(t.synonym_replacement(text))    
        try:
            seqs.append(t.random_insertion(text))
        except:
            pass

    
    """
    All sequence variations created in the data augmentation process are grouped in bags. 
    This is important to avoid that in the process of splitting the data, variations of 
    the same sequence are allocated in different sets. For example, an X variation of 
    sequence A falls in the training set and an Y variation of sequence A falls in the test set.
    """
    sequence_group = []
    target_group = []

    target = row['target']

    for sequence in seqs: 
        word_list = text_to_word_sequence(sequence)
        
        # remove stop words
        no_stop_words = remove_stopwords(word_list)
        
        if not no_stop_words:
            continue

        sequence_group.append(" ".join(no_stop_words))
        target_group.append(target)

    sequences.append(sequence_group)
    targets.append(target_group)


X = sequences
Y = np.array(targets)

print("{bags_count} bags".format(bags_count=len(X)))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,
                                                    random_state=42)

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.25,
                                                    random_state=42)

X_train = [item for sublist in X_train for item in sublist]
Y_train = [item for sublist in Y_train for item in sublist]

X_validation = [item for sublist in X_validation for item in sublist]
Y_validation = [item for sublist in Y_validation for item in sublist]

X_test = [item for sublist in X_test for item in sublist]
Y_test = [item for sublist in Y_test for item in sublist]

print("Train: {train_size}\nValidation: {validation_size}\nTest: {test_size}\n".format(train_size=len(X_train), validation_size=len(X_validation), test_size=len(X_test)))
# tokenizer
tokenizer = Tokenizer(num_words=max_words)  



# Updates internal vocabulary based on a list of texts. 
# This method creates the vocabulary index based on word frequency. 
tokenizer.fit_on_texts(X_train)


# Transforms each row from texts to a sequence of integers. 
# So it basically takes each word in the text and replaces it 
# with its corresponding integer value from the
X_train = tokenizer.texts_to_sequences(X_train)
X_validation = tokenizer.texts_to_sequences(X_validation)
X_test = tokenizer.texts_to_sequences(X_test)


# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_sequence_length, dtype='int32', value=0)
X_validation = pad_sequences(X_validation, maxlen=max_sequence_length, dtype='int32', value=0)
X_test = pad_sequences(X_test, maxlen=max_sequence_length, dtype='int32', value=0)


word_index = tokenizer.word_index

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_validation = np.array(X_validation)
Y_validation = np.array(Y_validation)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2

l2_reg = l2(0.001)

def model_fn():
    model = Sequential()

    model.add(Embedding(max_words, embedding_size, input_length=max_sequence_length, embeddings_regularizer=l2_reg))
    
    model.add(SpatialDropout1D(0.5))
    
    model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg, bias_regularizer=l2_reg))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    return model

!rm /kaggle/working/*
import os
import numpy as np
import pickle

# epochs
epochs = 10

# number of samples to use for each gradient update
batch_size = 128

# saving tokenizer
with open(tokenizer_file, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = model_fn()

# loadin saved model
if os.path.exists(model_file):
    model.load_weights(model_file)

history = model.fit(X_train, Y_train,
          validation_data=(X_validation, Y_validation),
          epochs=epochs,
          batch_size=batch_size,
          shuffle=True,
          verbose=1)

# saving model
model.save_weights(model_file)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show();
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show();
# evaluate model
scores = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Acc: %.2f%%" % (scores[1] * 100))
# Read the test data to create the submission.csv file

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

# Clear text
test['text'] = test['text'].apply(lambda x: clean_str(x))

# Remove stop words
test['text'] = test['text'].apply(lambda x: " ".join(remove_stopwords(x.split(" "))))

# Get text
test_X = list(test["text"])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(test_X)

# Pad sequences
sequences = pad_sequences(sequences,
                             maxlen=max_sequence_length,
                             dtype='int32',
                             value=0)

# Predict sequences
predicted = model.predict(sequences)

binary_predicted = np.array(predicted) >= 0.5
targets = binary_predicted.astype(int).reshape((len(binary_predicted)))

my_submission = pd.DataFrame({'id': test.id, 'target': targets})
my_submission.to_csv('submission.csv', index=False)

print("Submission file created!")