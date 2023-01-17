!ln -s /kaggle/input/projectdata/ data
!ln -s /kaggle/input/updateddata/ data2
!ls /kaggle/input/nyttxt/
republic = '/kaggle/input/republic/republic.txt'
rf = '/kaggle/input/train_test_strings/RFs.txt'
nyt = '/kaggle/input/nyttxt/nyt.txt'
import multiprocessing, string
multiprocessing.cpu_count()

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# def tokenize(text):
#     tokens = []; word = ""
#     for char in text.lower():
#         if (char in string.whitespace) or (char in string.punctuation):
#             if word:
#                 tokens.append(word.strip(' '))
#                 word = ""
#         if char in string.punctuation:
#             tokens.append(char)
#         else:
#             word += char
#     return [word for word in tokens]

# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', '')
    # split into tokens by white space
    tokens = doc.split()
    #tokens = tokenize(doc)
    # remove punctuation from each token
    #table = str.maketrans('', '', string.punctuation)
    #tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    #tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    #tokens = [word.lower() for word in tokens]
    return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
# load document
#in_filename = republic
in_filename = nyt
doc = load_doc(in_filename)
print(doc[:200])

# clean document
tokens = clean_doc(doc)
print(tokens[:50])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# bad_words = ['satan','hitler','suicide',
#             's', 't','m']
# for profanity in bad_words:
#     tokens = list(filter((profanity).__ne__, tokens))
# organize into sequences of tokens
length = 50
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))
# save sequences to file
#out_filename = 'republic_sequences.txt'
out_filename = 'sequences.txt'
save_doc(sequences, out_filename)
!head sequences.txt -n 2
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Lambda
from keras import backend as K
K.clear_session()
# load
in_filename = 'sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
lines[:5]
# integer encode sequences of words
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(lines)


sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
vocab_size
# separate into input and output
sequences = array(sequences)
sequences.shape
type(sequences)
len(sequences[0]), len(sequences[1]),len(sequences[4])
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
y
X.shape
seq_length = X.shape[1]
seq_length
# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=250)
 
# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

# load cleaned text sequences
in_filename = 'sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')

# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)
from keras.utils import plot_model
plot_model(model,'model.png')