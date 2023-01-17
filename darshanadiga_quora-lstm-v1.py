import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))
print(os.listdir("../input/embeddings/wiki-news-300d-1M"))
# Check if GPU is enabled
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from IPython.display import display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import pickle
from keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
# Import the data
input_dir = '../input/'
train_file_path = input_dir + 'train.csv'
test_file_path = input_dir + 'test.csv'

SUBMISSION_FILE_PATH = 'submission.csv'
EVALUATION_FILE_PATH = 'model/evaluation.txt'
MAX_SEQ_SIZE=150
TRAIN_FRACTION = 0.9

if not os.path.isdir('model'):
    os.mkdir('model')
TOKENIZER_FILE_PATH='model/token_model.pickle'
MODEL_FILE_PATH='model/lstm_model'

#WIKI_VECTORS_FILE = input_dir + 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
#PARAGRAM_VECTORS_FILE = input_dir + 'embeddings/paragram_300_sl999/paragram_300_sl999.txt'
GLOVE_VECTOR_FILE = input_dir + 'embeddings/glove.840B.300d/glove.840B.300d.txt'

WORD_EMB_FILE = GLOVE_VECTOR_FILE
VECTOR_LEN = 300
# Clean the text
import re
import nltk
#Looks like already downloaded: nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from nltk.stem import SnowballStemmer
import string

#Initialization
stop_word_set = set(stopwords.words("english"))
#stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Cleans and returns the given text/sentence by removing stop-words and stemming"""
    ## Remove puncuation
    text = text.translate(string.punctuation)
    ##> Replace with spaces
    text = re.sub(r"[^A-Za-z0-9^,.\/'+-=]", " ", text)
    ##> Convert words to lower case
    text = text.lower()
    ##> Clean the text
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    
    ##> Stop word removal
    ##split them
    text = text.split()
    ## Remove stop words
    text = [w for w in text if not w in stop_word_set]
    
    ##> Stemming
    #words = [stemmer.stem(word) for word in text]
    ##> Lemmatize
    words = [wordnet_lemmatizer.lemmatize(word) for word in text]
    text = " ".join(words)
    return text

#Test
clean_text("This isn't likes of you doing! at alll?")
import random
def balance_train_data(train_df, sincere_reduce_ratio=0.8):
    """
    Remove the sincere training data from the train_df to make it more balanced
    sincere_reduce_ratio=0.5 The sincere documents will be reduced by given percentage(approximately)
    """
    print('-------Balancing---------')
    tot_len = train_df.shape[0]
    sincere_idx = train_df.index[train_df['target'] == 0].tolist()
    print('Sincere questions before balancing:',len(sincere_idx))
    rand_idx_size_approx = int(len(sincere_idx) * sincere_reduce_ratio)
    rand_idx = list(set(random.choices(sincere_idx, k=rand_idx_size_approx)))
    rand_idx.extend(list(set(random.choices(sincere_idx, k=rand_idx_size_approx))))
    
    print('Going to drop {} sincere questions'.format(len(rand_idx)))
    train_df = train_df.drop(rand_idx)
    train_df = train_df.reset_index(drop=True)
    ln = train_df[train_df['target'] == 0].shape[0]
    print('Sincere questions after balancing:',ln)
    print('-------Balancing---------')
    return train_df

# Train and Test Data
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

print('Original Train shape:', train_df.shape)
# Take the evaluation data first
evaluation_df = train_df.tail(5000)

#Take the sample data
train_df = balance_train_data(train_df, sincere_reduce_ratio=0.5)
print('Re-balanced Train shape:', train_df.shape)

#Small data
#train_df = train_df.head(10000)

#Clean the dataframes before using them
train_df['question_text'] = train_df['question_text'].map(lambda t: clean_text(t))
test_df['question_text'] = test_df['question_text'].map(lambda t: clean_text(t))
evaluation_df['question_text'] = evaluation_df['question_text'].map(lambda t: clean_text(t))

print('Evaluation data:')
display(evaluation_df.groupby(['target']).count())

print('Train Shape:', train_df.shape)
display(train_df.groupby(['target']).count())
display(train_df.head())

# Prepare features and labels
qstns_np = train_df.question_text
target_labels = train_df.target
# Vectorization
#Features and lables
def vectorize_text(qstns_np, tokenizer_file_path, fit=True):
    """Tokenizes and then converts the them into sequences
    qstns_np               List of strings
    fit                    If true, the tokenizer will be fit on the given qstns_np
    tokenizer_file_path    The file path where the tokenizer weights will be saved to or restored from
    
    Returns word_to_idx, idx_to_word, vocab_size, qstns_seqs"""
    
    if fit:
        tokenizer = Tokenizer(filters='"#$%&*+/:;<=>@[\\]^_`{|}~\t\n')
        # Fit and then save
        tokenizer.fit_on_texts(qstns_np)
        with open(tokenizer_file_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved the tokenizer model at', tokenizer_file_path)
    else:
        # Load the tokenizer
        with open(tokenizer_file_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print('Loaded the tokenizer model from', tokenizer_file_path)

    qstns_seqs = tokenizer.texts_to_sequences(qstns_np)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer.word_index, tokenizer.index_word, vocab_size, qstns_seqs    
    
## Test case
_,_,_,_ = vectorize_text(qstns_np, TOKENIZER_FILE_PATH)
wi,iw,vs,sq = vectorize_text(['Why is this common in india?', 'test'], TOKENIZER_FILE_PATH, fit=False)
print('Vocab Size:', vs)
print('Test seq:', sq)
def preprocess_sequences(qstns_seqs, max_seq_size=MAX_SEQ_SIZE):
    """
    Trim down the token sequences qstns_seqs to max_seq_size
    qstns_seqs         List of tokenized sequences
    max_seq_size=MAX_SEQ_SIZE   Maximum length of the tokenized sequences(to trim down or to pad)

    Returns the updated token sequences of shape (len(qstns_seqs), max_seq_size)
    """
    #print('Before trim', np.array(qstns_seqs).shape)
    new_qstns_seqs = []
    for q in qstns_seqs:
        # Zero Padding
        if len(q) < max_seq_size:
            q.extend(np.zeros(max_seq_size - len(q)))
        # Trim down
        if len(q) > max_seq_size:
            q = q[:max_seq_size]
        new_qstns_seqs.append(q)
    #print('After trim', np.array(new_qstns_seqs).shape)
    return new_qstns_seqs

def prepare_train_data(qstns_seqs, target_labels, train_ratio=TRAIN_FRACTION):
    """
    qstns_seqs         List of tokenized sequences
    target_labels      List of 0/1 labels
    train_ratio=TRAIN_FRACTION    Percentage of random tokenized sequences to put in training set
    
    Returns train_x,train_y,test_x,test_y
    """
    # Sequence preprocessing
    qstns_seqs = preprocess_sequences(qstns_seqs, MAX_SEQ_SIZE)
            
    # Debug logs
    s = 0
    print('Sample x len', len(qstns_seqs[s]))
    print('Sample x', qstns_seqs[s])
    print('Sample y', target_labels[s])
    print('---------------')
        
    # Select random train_ration samples
    qstns_seqs, target_labels = shuffle(qstns_seqs, target_labels)
    train_size = int(len(target_labels) * train_ratio)
    train_x = np.array(qstns_seqs[:train_size])
    test_x = np.array(qstns_seqs[train_size:])
    
    train_y = np.array(target_labels[:train_size])
    test_y = np.array(target_labels[train_size:])
    
    #OneHot encoding of target labels
    #onehot_enc = OneHotEncoder(categories='auto')
    onehot_enc = OneHotEncoder()
    onehot_enc.fit(train_y.reshape(-1,1))
    train_y = onehot_enc.transform(train_y.reshape(-1,1)).toarray()
    test_y = onehot_enc.transform(test_y.reshape(-1,1)).toarray()
    
    return train_x, train_y, test_x, test_y

## Test case
_,_,_,qstns_seqs = vectorize_text(qstns_np, TOKENIZER_FILE_PATH, fit=False)
train_x, train_y, test_x, test_y = prepare_train_data(qstns_seqs, target_labels, train_ratio=TRAIN_FRACTION)
print('Train X shape:', train_x.shape)
print('Train Y shape:', train_y.shape)
print('Test X shape:', test_x.shape)
print('Test Y shape:', test_y.shape)
## Word embeddings
# Load in Embedding file
def load_word_emb(word_emb_file):
    """
    Loads the word-embeddings from the given file path into a dictionary {word: embedding vector} and returns it.
    """
    print('Loading word embedding:', word_emb_file)
    model = {}
    errors = 0
    with open(word_emb_file,'r', errors='surrogateescape') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            try:
                embedding = np.array([float(val) for val in splitLine[1:]])
            except:
                errors += 1
            model[word] = embedding
    print('Done', len(model), 'words loaded!')
    print('Errorneous lines:', errors)
    return model

def prepare_embedding_matrix(word_lookup, word_to_index, vocab_size, vector_len):
    """
    Prepares the embedding_matrix of shape (vocab_size, vector_len)
    word_lookup    Dictionary of {word: embeddings of length vector_len}
    word_to_index  Dictionary of {word: token index in the tokenized vocabulary}
    vocab_size     Number of unique words in the tokenized vocabulary
    vector_len     Length of embedding vector
    
    Returns the embedding_matrix of shape (vocab_size, vector_len)
    """
    embedding_matrix = np.zeros((vocab_size, vector_len))
    word_not_found = 0
    for i, word in enumerate(word_to_index.keys()):
        vector = word_lookup.get(word, None)
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
        else:
            word_not_found += 1
    
    print('WARNING: There were {0} words without pre-trained embeddings!'.format(word_not_found))
    return embedding_matrix

##>COSTLY
word_lookup = load_word_emb(word_emb_file=WORD_EMB_FILE)
print('Loaded the ', len(word_lookup), ' word embeddings')

# Test case
embedding_matrix = prepare_embedding_matrix(word_lookup, wi, vs, vector_len=VECTOR_LEN)
print('Embedding matrix shape:', embedding_matrix.shape)
## Below cells build the actual network and training flow
# Keras imports
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam
from keras.utils import plot_model

# Build the LSTM network
def build_lstm_network(vocab_size,
                       output_size,
                       embedding_matrix,
                       lstm_cells=64,
                       trainable=False,
                       lstm_layers=1,
                       bi_direc=False,
                       dropout=0.1):
    """
    Builds a LSTM network that uses a pretrained embeddings with lstm_layers layers each containing lstm_cells cells and returns it.
    vocab_size         Unique words in the tokenized vocabulary of training data
    output_size        Output vector size produced by the network
    embedding_matrix   Embedding matrix of shape (vocab_size, embedding length)
    lstm_cells=64      Number of cells in each LSTM layer
    trainable=False    Whether to train the pre-trained word embedding weights or not
    lstm_layers=1      Number of LSTM layers in the network
    bi_direc=False     Whether to add bi-directional layers(both past and future contexts)
    dropout=0.1        Dropout value at LSTM layers
    """
    lstm_model = Sequential()
    
    # Trainable embeddings or not
    if trainable:
        lstm_model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True))
    else:
        lstm_model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
        lstm_model.add(Masking())
        
    # Adding initial set of LSTM layers if lstm_layers is more than 1
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            lstm_model.add(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=dropout))
    
    # Adding the final LSTM layer
    if bi_direc:
        lstm_model.add(
            Bidirectional(
                LSTM(
                lstm_cells,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=dropout)))
    else:
        lstm_model.add(
            LSTM(
                lstm_cells,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=dropout))
        
    # Rest of the network after LSTM layers
    # Dense layer
    lstm_model.add(Dense(64, activation='relu'))
    
    # Dropout of regularization
    lstm_model.add(Dropout(0.5))
    
    # Output layer
    lstm_model.add(Dense(output_size, activation='softmax'))
    
    #Compile the model
    lstm_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    return lstm_model

print('LSTM network')
# Training callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

def initialize_callbacks(model_file_path):
    """
    Initialize a list of keras callbacks to be used during training process.
    model_file_path    File path where the model file needs to be saved/checkpointed
    """
    # Early stopping callback
    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    
    # Model saving callback
    callbacks.append(
        ModelCheckpoint(
            '{0}.h5'.format(model_file_path),
            save_best_only=True,
            save_weights_only=False))
        
    return callbacks

# Test case
callbacks = initialize_callbacks(model_file_path=MODEL_FILE_PATH)
print('Initialized the callbacks')
# Start the training
EPOCHS = 70
BATCH_SIZE = 1500
SAVE_MODEL = True
VERBOSE = 1

# Build the model
lstm_model = build_lstm_network(vocab_size=vs,
                   output_size=2,
                   embedding_matrix=embedding_matrix,
                   lstm_cells=64,
                   trainable=True,
                   lstm_layers=2,
                   bi_direc=True,
                   dropout=0.1)
lstm_model.summary()

# Train the model
train_history = lstm_model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=VERBOSE,
    callbacks=callbacks,
    validation_data=(test_x, test_y))
train_history
# Run the submission data
def load_lstm_model(model_file_path):
    """
    Load the trained model and return it
    """
    lstm_model = load_model('{0}.h5'.format(model_file_path))
    return lstm_model
    
# Test case
lstm_model = load_lstm_model(model_file_path=MODEL_FILE_PATH)
# Save evaluation results
from sklearn.metrics import classification_report, confusion_matrix
#TODO predict and output evaluation results evaluation_df

# Sequence preprocessing
_,_,_,eval_q_seqs = vectorize_text(evaluation_df.question_text, TOKENIZER_FILE_PATH, fit=False)
eval_y = evaluation_df.target
eval_q_seqs = preprocess_sequences(eval_q_seqs, MAX_SEQ_SIZE)
eval_x = np.array(eval_q_seqs)
eval_y = np.array(eval_y)

onehot_enc = OneHotEncoder()
onehot_enc.fit(eval_y.reshape(-1,1))
eval_y = onehot_enc.transform(eval_y.reshape(-1,1)).toarray()

print('Eval x:', eval_x.shape)
#display(eval_x[:5])
print('Eval y:', eval_y.shape)
#display(eval_y[:5])

y_eval_pred = lstm_model.predict(eval_x)
y_eval_pred_bin = np.argmax(y_eval_pred, axis=1)
#OneHot to 0s and 1s
y_eval_actual = [0 if t[0]==1 else 1 for t in eval_y]
#print(confusion_matrix(y_true, y_pred))
print(classification_report(y_eval_actual, y_eval_pred_bin, target_names=['sincere', 'insincere']))

evaluation_df['predicted'] = y_eval_pred_bin
display(evaluation_df.sample(10))
evaluation_df.to_csv(EVALUATION_FILE_PATH, index=False)
# Model performance metrics
from sklearn.metrics import classification_report, confusion_matrix

y_pred = lstm_model.predict(test_x)
y_pred = np.argmax(y_pred, axis=1)
#OneHot to 0s and 1s
y_true = [0 if t[0]==1 else 1 for t in test_y]
#print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=['sincere', 'insincere']))
# Submission test data
print('Sample test data')
display(test_df.head())
def submission_results():
    test_questions = test_df.question_text
    test_qids = test_df.qid
    
    _,_,_,test_seq = vectorize_text(test_questions, TOKENIZER_FILE_PATH, fit=False)
    submission_x = preprocess_sequences(test_seq, MAX_SEQ_SIZE)
    submission_x = np.array(submission_x)
    print('Submission X shape:', submission_x.shape)
    
    # Run the prediction
    lstm_model = load_lstm_model(model_file_path=MODEL_FILE_PATH)
    submission_y = lstm_model.predict(submission_x)
    submission_y = np.argmax(submission_y, axis=1)
    submission_df = test_df.copy(deep=True)
    submission_df['prediction'] = submission_y
    return submission_df

##>COSTLY 
submission_df = submission_results()
display(submission_df.head())
submission_df.loc[:,['qid','prediction']].to_csv(SUBMISSION_FILE_PATH, index=False)
print('----------####---------')
print('Saved the submission data to ', SUBMISSION_FILE_PATH)
