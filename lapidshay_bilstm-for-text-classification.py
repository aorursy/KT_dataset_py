import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns



from copy import deepcopy

import gc



import nltk

from nltk.tokenize import TweetTokenizer  # Twitter-aware tokenizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords



import torch

import torch.nn as nn



from gensim.models.word2vec import Word2Vec

import gensim.downloader as api



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



%matplotlib inline
def _normalize_tweet(text):

    """Returns a normalized versions of text."""



    # change hyperlinks to '<url>' tokens

    output = re.sub(r'http[s]{0,1}://t.co/[a-zA-Z0-9]+\b', '<url>', text)

    

    # separate all '#' signs from following word with one whitespace

    output = re.sub(r'#(\w+)', r'# \1', output)



    return output
def _tokenize(tokenizer, string):

    """Tokenizes a sentence, but leave hastags (#) and users (@)"""

    

    tokenized = tokenizer.tokenize(string)

    return tokenized
def _numbers_to_number_tokens(tokenized_string, num_token='<number>'):

    """Returns the tokenized string (list) with numbers replaced by a numbet token."""

    

    # create a list of (word, POS-tags) tuples

    pos_tagged = nltk.pos_tag(tokenized_string)

    

    # find indices of number POS tags

    num_indices = [idx for idx in range(len(pos_tagged)) if pos_tagged[idx][1] == 'CD']

    

    # replace numbers by token

    for idx in num_indices:

        tokenized_string[idx] = num_token

        

    return tokenized_string    
def preprocess_text(tokenizer, string):

    """Executes all text cleaning functions."""

    

    return _numbers_to_number_tokens(_tokenize(tokenizer, _normalize_tweet(string)))
def preprocess_keyword(keyword):

    """Returns a clean, tokenized keyword."""

    

    # return None if keywors is np.nan

    if type(keyword) == np.float and np.isnan(keyword):

        return

    

    # replace '%20' with whitespace, lower, and tokenize

    output = re.sub(r'%20', ' ', keyword)

    output = output.lower()

    output = output.split()

    return output
def count_all_caps(text):

    """Returns an integer denoting number of ALL-CAPS words (e.g. 'CANADA', 'WELCOME')."""



    return len([word for word in text.split() if word.isupper()])
def count_capitalized(text):

    """Returns an integer denoting number of capitalized words (e.g. 'Beer', 'Obama')."""



    return len([word for word in text.split() if word.istitle()])
def count_words(text):

    """Returns an integer denoting number of words in tweet (before normalizing)."""



    return len(text.split())
def sentiment_analyze_df(df, column):

    """Adds 4 columns of sentiment analysis scores to input DataFrame. changes occur inplace."""



    # instantiate a sentiment anlayzer

    sid = SentimentIntensityAnalyzer()

    

    # instantiate a matrix and populate it with scores of each of df[column]

    output_values = np.zeros((len(df), 4))

    for tup in df.itertuples():

        output_values[tup.Index, :] = list(sid.polarity_scores(' '.join(getattr(tup, column))).values())

    

    # adding column to input DataFrame

    for idx, col in enumerate(['sent_neg', 'sent_neu', 'sent_pos', 'sent_compound']):

        df[col] = output_values[:, idx]
def _get_word_vec(embedding_model, use_norm, word):

    """

    Returns a normalized embedding vector of input word.

    

    Takes care of special cases.

    <url> tokens are already taken care of in normalization.

    """



    if word[0] == '@':

        return embedding_model.word_vec('<user>', use_norm=use_norm)

        

    elif word == '#':

        return embedding_model.word_vec('<hashtag>', use_norm=use_norm)



    elif word in embedding_model.vocab:

        return embedding_model.word_vec(word, use_norm=use_norm)



    else:

        return embedding_model.word_vec('<UNK>', use_norm=use_norm)
def _text_to_vectors(embedding_model, use_norm, tokenized_text):

    """Returns tweet's words' embedding vector.s"""



    vectors = [_get_word_vec(embedding_model, use_norm, word) for word in tokenized_text]

    vectors = np.array(vectors)

    

    return vectors
def _trim_and_pad_vectors(text_vectors, embedding_dimension, seq_len):

    """Returns a padded matrix of text embedding vectors with dimensions (seq_len, embedding dimensions)."""



    # instantiate 0's matrix

    output = np.zeros((seq_len, embedding_dimension))



    # trim long tweets to be seq_len long

    trimmed_vectors = text_vectors[:seq_len]



    # determine index of end of padding and beginning of tweet embedding

    end_of_padding_index = seq_len - trimmed_vectors.shape[0]



    # pad if needed, by replacing last rows with the tweet's words' embedding vectors

    output[end_of_padding_index:] = trimmed_vectors



    return output
def embedding_preprocess(embedding_model, use_norm, seq_len, tokenized_text):

    """Returns an embedding representation of input tokenized text, by executing text embedding functions."""

    

    # get matrix of tweet's words' embedding vectors (tweet length, embedding_dimension)

    text_vectors = _text_to_vectors(embedding_model, use_norm, tokenized_text)

    

    output = _trim_and_pad_vectors(text_vectors, embedding_model.vector_size, seq_len)

    

    return output
def keyword_to_avg_vector(embedding_model, use_norm, tokenized_keyword):

    """Returns keyword(s') average embedding vector."""

    

    # return a vector of zeros if tokenized_keyword is None

    if tokenized_keyword is None:

        return np.zeros((1, embedding_model.vector_size))

    

    # otherwise, calculate average embedding vector

    vectors = [_get_word_vec(embedding_model, use_norm, word) for word in tokenized_keyword]

    vectors = np.array(vectors)

    avg_vector = np.mean(vectors, axis=0)

    avg_vector = avg_vector.reshape((1, embedding_model.vector_size))

    return avg_vector
# load a pre-trained model, which was trained on twitter

model_glove_twitter = api.load("glove-twitter-100")
# create a random vector, to represent <UNK> token (unseen word)

random_vec_for_unk = np.random.uniform(-1, 1, size=model_glove_twitter.vector_size).astype('float32')

random_vec_for_unk = random_vec_for_unk.reshape(1,model_glove_twitter.vector_size)
# add the random vector to model

model_glove_twitter.add(['<UNK>'], random_vec_for_unk, replace=True)
# compute noramlized vectors, and replace originals

model_glove_twitter.init_sims(replace=True)
TRAIN_SET_PATH = '../input/nlp-getting-started/train.csv'
train_df = pd.read_csv(TRAIN_SET_PATH)

train_df.head()
# create a tokenizer which lowercases, reduces length of and preserves user handles ('@user')

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False) 
# normalize and tokenize texts

train_df['tok_norm_text'] = [preprocess_text(tokenizer, text) for text in train_df['text']]
train_df['keyword'] = train_df['keyword'].apply(preprocess_keyword)
train_df['num_all_caps'] = train_df['text'].apply(count_all_caps)

train_df['num_caps'] = train_df['text'].apply(count_capitalized)

train_df['num_words'] = train_df['text'].apply(count_words)
# create a scaler to make all features be in range [-1, 1], thus suitable for a newural network model

scaler = MinMaxScaler(feature_range=(-1, 1))



columns_to_scale = ['num_all_caps', 'num_caps', 'num_words']

scaler.fit(train_df[columns_to_scale])

train_df[columns_to_scale] = scaler.transform(train_df[columns_to_scale])
# create sentiment analysis feautres

sentiment_analyze_df(train_df, 'tok_norm_text')
train_df.head()
sns.distplot([len(tok) for tok in train_df['tok_norm_text']])
sequence_max_length = 30
train_df['text_embedding'] = [

    embedding_preprocess(

        embedding_model=model_glove_twitter, use_norm=True, seq_len=sequence_max_length, tokenized_text=text)

    for text in train_df['tok_norm_text']

]
train_df['keyword_embedding'] = [

    keyword_to_avg_vector(embedding_model=model_glove_twitter, use_norm=True, tokenized_keyword=keyword)

    for keyword in train_df['keyword']

]
train_df.head()
def _single_values_repeat(seq_len, static_single_values):

    """Returns a numpy array containing seq_len-repeated values."""

    

    output = static_single_values.reshape((1, len(static_single_values)))

    output = np.repeat(output, seq_len, axis=0)

    return output
def _static_embedding_repeat(seq_len, static_embedding_values):

    """Return a numpy array os stacked static embedding vectors."""

    

    horizontally_stacked = np.hstack(static_embedding_values)

    output = np.repeat(horizontally_stacked, seq_len, axis=0)

    return output
def concatenate_embeddings(df, embedding_model, seq_len, sequence_embedding_col, static_embedding_cols, static_singles_cols):

    """Returns one embedding representation of all features - main sequence, static embedded featues, and single values."""

    

    emb_dim = embedding_model.vector_size

    

    # instantiate output matrix

    output = np.zeros((len(df), seq_len, len(static_singles_cols) + len(static_embedding_cols) * emb_dim + emb_dim))

    

    for idx, row in df.iterrows():

        

        single_vals = _single_values_repeat(seq_len, row[static_singles_cols].values)

        static_emb_vals = _static_embedding_repeat(seq_len, row[static_embedding_cols])

        seq_emb_vals = row[sequence_embedding_col]



        # horizontally stack embeddings and features

        row_embedding = np.hstack((single_vals, static_emb_vals, seq_emb_vals))



        output[idx, :, :] = row_embedding

        

    return output
# Create one embedding representation of all chosen features

embedding_matrix = concatenate_embeddings(

    df=train_df, embedding_model=model_glove_twitter, seq_len=sequence_max_length,

    sequence_embedding_col='text_embedding',

    static_embedding_cols=['keyword_embedding'],

    static_singles_cols=['num_all_caps', 'num_caps', 'num_words', 'sent_neg', 'sent_neu', 'sent_pos', 'sent_compound'])
embedding_matrix.shape
class BiLSTM(nn.Module):

    """A pyTorch Bi-Directional LSTM RNN implementation"""



    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, batch_size, dropout, device):

        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.num_layers = num_layers



        self.dropout = nn.Dropout(p=dropout)



        self.lstm = nn.LSTM(

            input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,

            batch_first=True, dropout=dropout, bidirectional=True)

        

        self.fc = nn.Linear(hidden_dim * 2, num_classes)



        self.device = device

        

        # instantiate lists for evaluating and plotting

        self.train_loss = []

        self.train_acc = []

        self.val_loss = []

        self.val_acc = []

        

        # an attribute to hold model's best weights (used for evaluating)

        self.best_weights = deepcopy(self.state_dict())



    def _init_hidden(self, current_batch_size):

        """Sets initial hidden and cell states (for LSTM)."""



        h0 = torch.zeros(self.num_layers * 2, current_batch_size, self.hidden_dim).to(self.device)

        c0 = torch.zeros(self.num_layers * 2, current_batch_size, self.hidden_dim).to(self.device)

        return h0, c0



    def forward(self, x):

        """Forward step."""



        # Forward propagate LSTM

        h, c = self._init_hidden(current_batch_size=x.size(0))

        out, _ = self.lstm(x, (h, c))



        # dropout

        out = self.dropout(out)



        # Decode the hidden state of the last time step

        out = self.fc(out[:, -1, :])



        return out

    

    def predict(self, x: torch.tensor):

        """Return a tensor of predictions of tensor x."""



        class_predictions = self(x).data

        _, predicted = torch.max(class_predictions, dim=1)

        return predicted



    def _train_evaluate(self, X_train, y_train, X_val, y_val, criterion):

        """Evaluates model during training time, and returns train_loss, train_acc, val_loss, val_acc."""



        # set model to evaluation mode

        self.eval()



        # calculate accuracy and loss of train set and append to lists

        epoch_train_acc = (self.predict(X_train) == y_train).sum().item() / y_train.shape[0]

        epoch_train_loss = criterion(self(X_train), y_train).item()

        self.train_acc.append(epoch_train_acc)

        self.train_loss.append(epoch_train_loss)



        # calculate accuracy and loss of validation set, and append to lists

        if X_val is not None and y_val is not None:

            epoch_val_acc = (self.predict(X_val) == y_val).sum().item() / y_val.shape[0]

            epoch_val_loss = criterion(self(X_val), y_val).item()

            self.val_acc.append(epoch_val_acc)

            self.val_loss.append(epoch_val_loss)



            # return all loss and accuracy values

            return epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc



        # return train set loss and accuracy values, if there is no validation set

        return epoch_train_loss, epoch_train_acc, None, None

    

    @staticmethod

    def _print_progress(epoch, train_loss, train_acc, val_loss, val_acc, improved, verbose=False):

        """Prints the training progress."""



        output = f'Epoch {str(epoch + 1).zfill(3)}:'

        output += f'\n\t Training   Loss: {str(train_loss)[:5]} | Accuracy: {str(train_acc)[:5]}.'



        if val_loss is not None and val_acc is not None:

            output += f'\n\t Validation Loss: {str(val_loss)[:5]} | Accuracy: {str(val_acc)[:5]}.'



        if improved:

            output += f' Improvement!'



        if verbose:

            print(output)



    def fit(self, X_train, y_train, X_val, y_val, epoch_num, criterion, optimizer, verbose=False):

        """Trains the model."""



        # a variable to determine whether to update best weights (and report progress)

        best_acc = 0.0



        # split dataset to batches

        X_train_tensor_batches = torch.split(X_train, self.batch_size)

        y_train_tensor_batches = torch.split(y_train, self.batch_size)



        for epoch in range(epoch_num):



            # set model to train mode

            self.train()



            for i, (X_batch, y_batch) in enumerate(zip(X_train_tensor_batches, y_train_tensor_batches)):



                # Forward pass

                outputs = self(X_batch)

                loss = criterion(outputs, y_batch)



                # Backward and optimize

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



            # calculate accuracy and loss of train and validation set (if validation set is None, values are None)

            train_loss, train_acc, val_loss, val_acc = self._train_evaluate(X_train, y_train, X_val, y_val, criterion)



            # a boolean to determine the correct accuracy to consider for progress (Validation ot Training)

            if X_val is not None and y_val is not None:

                accuracy = val_acc

            else:

                accuracy = train_acc



            # if accuracy outperforms previous best accuracy, print and update best accuracy and model's best weights

            if accuracy > best_acc:

                self._print_progress(epoch, train_loss, train_acc, val_loss, val_acc, improved=True, verbose=verbose)

                best_acc = accuracy

                self.best_weights = deepcopy(self.state_dict())



            # else, print

            else:

                self._print_progress(epoch, train_loss, train_acc, val_loss, val_acc, improved=False, verbose=verbose)



        gc.collect()
def plot_graphs(model):

    plt.figure(figsize=(6, 12))



    plt.subplot(311)

    plt.title("Accuracy")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.plot(range(1, len(model.train_acc)+1), model.train_acc, label="Train")

    plt.plot(range(1, len(model.val_acc)+1), model.val_acc, label="Validation")



    plt.xticks(np.arange(0, len(model.train_acc)+1, 5))

    plt.legend()



    plt.subplot(312)

    plt.title("Loss")

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.plot(range(1, len(model.train_loss)+1), model.train_loss, label="Train")

    plt.plot(range(1, len(model.val_loss)+1), model.val_loss, label="Validation")



    plt.xticks(np.arange(0, len(model.train_acc)+1, 5))

    plt.legend()



    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show()
X_train_val, X_held_out_set, y_train_val, y_held_out_set = train_test_split(

    embedding_matrix, train_df['target'].values, test_size=0.1)
# determines which device to mount the model to

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {device}')
# convert above LSTM arrays to tensors to be used in BiLSTM Neural Network

X_train_val = torch.from_numpy(X_train_val).float().to(device)

X_held_out_set = torch.from_numpy(X_held_out_set).float().to(device)

y_train_val = torch.from_numpy(y_train_val).long().to(device)

y_held_out_set = torch.from_numpy(y_held_out_set).long().to(device)
# network hyprer-parameters

embedding_dim = embedding_matrix.shape[2]

hidden_size = 50

num_layers = 2

num_classes = 2

batch_size = 256

dropout = 0.3
# learning hyprer-parameters

num_epochs = 20

learning_rate = 0.0005

weight_decay = 0.0005
# instantiate Model, Loss and Optimizer

bilstm = BiLSTM(embedding_dim, hidden_size, num_layers, num_classes, batch_size, dropout, device).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(bilstm.parameters(), lr=learning_rate, weight_decay=weight_decay)
# train the model

bilstm.fit(

    X_train=X_train_val, y_train=y_train_val, X_val=X_held_out_set, y_val=y_held_out_set,

    epoch_num=num_epochs, criterion=criterion, optimizer=optimizer, verbose=True)
plot_graphs(bilstm)
del(bilstm)
X_train = torch.from_numpy(embedding_matrix).float().to(device)

y_train = torch.from_numpy(train_df['target'].values).long().to(device)
# instantiate Model, Loss and Optimizer

bilstm = BiLSTM(embedding_dim, hidden_size, num_layers, num_classes, batch_size, dropout, device).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(bilstm.parameters(), lr=learning_rate, weight_decay=weight_decay)
# train the model

bilstm.fit(

    X_train=X_train, y_train=y_train, X_val=None, y_val=None,

    epoch_num=num_epochs, criterion=criterion, optimizer=optimizer, verbose=True)
plot_graphs(bilstm)
TEST_SET_PATH = '../input/nlp-getting-started/test.csv'
test_df = pd.read_csv(TEST_SET_PATH)

test_df.head()
# normalize and tokenize texts and keywords

test_df['tok_norm_text'] = [preprocess_text(tokenizer, text) for text in test_df['text']]

test_df['keyword'] = test_df['keyword'].apply(preprocess_keyword)
# feature extraction

test_df['num_all_caps'] = test_df['text'].apply(count_all_caps)

test_df['num_caps'] = test_df['text'].apply(count_capitalized)

test_df['num_words'] = test_df['text'].apply(count_words)



test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])



sentiment_analyze_df(test_df, 'tok_norm_text')
#text embedding

test_df['text_embedding'] = [

    embedding_preprocess(

        embedding_model=model_glove_twitter, use_norm=True, seq_len=sequence_max_length, tokenized_text=text)

    for text in test_df['tok_norm_text']

]
#keyword embedding

test_df['keyword_embedding'] = [

    keyword_to_avg_vector(embedding_model=model_glove_twitter, use_norm=True, tokenized_keyword=keyword)

    for keyword in test_df['keyword']

]
test_df.head()
# Create one embedding representation of all chosen features

test_embedding_matrix = concatenate_embeddings(

    df=test_df, embedding_model=model_glove_twitter, seq_len=sequence_max_length,

    sequence_embedding_col='text_embedding',

    static_embedding_cols=['keyword_embedding'],

    static_singles_cols=['num_all_caps', 'num_caps', 'num_words', 'sent_neg', 'sent_neu', 'sent_pos', 'sent_compound'])
X_test = torch.from_numpy(test_embedding_matrix).float().to(device)
# predict

preds = bilstm.predict(X_test)
# put predictions and id's in DataFrame

final_preds = preds.cpu().numpy().reshape(-1,1)

ids = test_df['id'].values.reshape(-1,1)

data = np.hstack((ids, final_preds))



submission_df = pd.DataFrame(data=data,columns = ['id', 'target'])
submission_df.to_csv('submission.csv', index=False)