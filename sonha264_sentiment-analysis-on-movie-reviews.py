import numpy as np 
import pandas as pd 
from os import listdir
from nltk.corpus import stopwords
import string
import re
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
def load_doc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close
    return(text)

def clean_doc(doc):
    tokens = doc.split()
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # remove all punc, either in a word (eg. what's) or independent (normal ",")
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('',w) for w in tokens] 
    # remove remaining tokens that are not alphabetic (blank words, etc)
    tokens = [w for w in tokens if w.isalpha()]
    # filter out short tokens
    tokens = [w for w in tokens if len(w)>1]
    return(tokens)

def add_doc_to_vocab(filename,vocab): # vocab here is a counter object
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)
    
def doc_to_lines(filename,vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def create_vocab(directory,vocab):
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)

def process_docs(directory,vocab,is_train):
    lines = list()
    for filename in listdir(directory):
    # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + "/" + filename
        line = doc_to_lines(path,vocab)
        lines.append(line)
    return(lines)

def load_clean_dataset(vocab,is_train):
    neg = process_docs('../input/movie-review-polarity/txt_sentoken/neg',vocab,is_train)
    pos = process_docs('../input/movie-review-polarity/txt_sentoken/pos',vocab,is_train)
    docs = neg + pos
    labels = [0 for i in range(len(neg))] + [1 for i in range(len(neg))]
    return(docs,labels)

def save_list(lines,filename):
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()
    
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def predict_sentiment(review,vocab,tokenizer,model):
    tokens = clean_doc(review)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    yhat = model.predict(encoded, verbose=0)
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'

# Load data and create a 'bag-of-words'
vocab = Counter()
create_vocab('../input/movie-review-polarity/txt_sentoken/neg',vocab)
create_vocab('../input/movie-review-polarity/txt_sentoken/pos',vocab)
# Remove words with low freq occurance dut to weak predictive power
min_occurance = 2
tokens = [k for k,c in vocab.items() if c >= min_occurance]
# Save bag of words for future use
save_list(tokens,'vocab.txt')

# Load that vocab just saved 
vocab_filename = './vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

# Load the dataset
train_docs,y_train = load_clean_dataset(vocab,is_train=True)
test_docs,y_test = load_clean_dataset(vocab,is_train=False)

# Create the tokenizer
tokenizer = create_tokenizer(train_docs)

# encode data
X_train = tokenizer.texts_to_matrix(train_docs,mode='binary')
X_test = tokenizer.texts_to_matrix(test_docs,mode='binary')

# binary: given the bag of words, for each paragraph, encode 1 for word appear in that para, 0 otherwise. 
# Return vector of same length as bag of words.
# freq: same thing but encode the frequency in each paragraph of that word 
# Training time!

X_train = np.array(X_train) #.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)
X_test = np.array(X_test) 
y_test = np.array(y_test) # Weird tf bug

X_tr, X_val, y_tr, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=1)

# Define
model = Sequential()
model.add(Dense(50, input_shape = (25798,),activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])
model.summary()

# Fit
model.fit(X_tr, y_tr, epochs=10, verbose=2,
         validation_data = (X_val,y_val))

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))
text = "Animation is spectacular but the dialogue and plot of the movie is terrible"
percent, sentiment = predict_sentiment(text, vocab, tokenizer, model)
print(percent)
print(sentiment)