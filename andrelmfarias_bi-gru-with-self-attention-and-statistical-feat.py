import pandas as pd

import numpy as np

import re



from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader



import spacy

from gensim.models import KeyedVectors

from nltk.stem import PorterStemmer, SnowballStemmer

from nltk.stem.lancaster import LancasterStemmer



import time

import gc

from tqdm import tqdm
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



labels = np.array(train_df.target, dtype=int)
# Functions to extract statistical features



def n_upper(sentence):

    return len(re.findall(r'[A-Z]',sentence))



def n_unique_words(sentence):

    return len(set(sentence.split()))



def n_question_mark(sentence):

    return len(re.findall(r'[?]',sentence))



def n_exclamation_mark(sentence):

    return len(re.findall(r'[!]',sentence))



def n_asterisk(sentence):

    return len(re.findall(r'[*]',sentence))



def n_parentheses(sentence):

    return len(re.findall(r'[()]',sentence))



def n_brackets(sentence):

    return len(re.findall(r'[\[\]]',sentence))



def n_braces(sentence):

    return len(re.findall(r'[{}]',sentence))



def n_quotes(sentence):

    return len(re.findall(r'["]',sentence))



def n_ampersand(sentence):

    return len(re.findall(r'[&]',sentence))



def n_dash(sentence):

    return len(re.findall(r'[-]',sentence))



n_stats = 11 # Number of statistical features excluding sequence length



def get_stat(questions_list):

    ''' 

    Function that builds matrix of statistical features

    '''

    stat_feat = np.zeros((len(questions_list), n_stats), dtype=int)

    for i,question in tqdm(enumerate(questions_list)):

        stat_feat[i,0] = n_upper(question)

        stat_feat[i,1] = n_unique_words(question)

        stat_feat[i,2] = n_question_mark(question)

        stat_feat[i,3] = n_exclamation_mark(question)

        stat_feat[i,4] = n_asterisk(question)

        stat_feat[i,5] = n_parentheses(question)

        stat_feat[i,6] = n_brackets(question)

        stat_feat[i,7] = n_braces(question)

        stat_feat[i,8] = n_quotes(question)

        stat_feat[i,9] = n_ampersand(question)

        stat_feat[i,10] = n_dash(question)

    

    return stat_feat    
train_stat = get_stat(train_df.question_text)

test_stat = get_stat(test_df.question_text)
# Lowering all the text and storing the lists

train_list = list(train_df.question_text.apply(lambda s: s.lower()))

test_list = list(test_df.question_text.apply(lambda s: s.lower()))



# Getting all text in both samples

train_text = ' '.join(train_list)

test_text = ' '.join(test_list)
# Using spacy parser and tokenizer

nlp = spacy.load("en", disable=['tagger','parser','ner','textcat'])
# Creating the vocabulary and tokenizing datasets

vocab = {}

lemma_vocab = {} # lemmatizing vocabulary

word_idx = 1 # start from 1 as we use 0 for padding



train_tokens = []

for doc in tqdm(nlp.pipe(train_list)):

    curr_tokens = []

    for token in doc:

        if token.text not in vocab:

            vocab[token.text] = word_idx

            lemma_vocab[token.text] = token.lemma_

            word_idx += 1

        curr_tokens.append(vocab[token.text])

    train_tokens.append(np.array(curr_tokens, dtype=int))



test_tokens = []

for doc in tqdm(nlp.pipe(test_list)):

    curr_tokens = []

    for token in doc:

        if token.text not in vocab:

            vocab[token.text] = word_idx

            lemma_vocab[token.text] = token.lemma_

            word_idx += 1

        curr_tokens.append(vocab[token.text])

    test_tokens.append(np.array(curr_tokens, dtype=int))
def pad(questions, seq_length):

    '''

    This function pad the questions fed as list of tokens with 0 at left

    and returns a numpy array, and the length of the sentence at the position 0 of the array.

    This length will be useful in order to change the sequences length for each batch during 

    training and to be used as statistical feature.

    '''

    

    features = np.zeros((len(questions), seq_length+1), dtype=int)

    for i, sentence in enumerate(questions):

        if len(sentence)==0: # dealing with empty sentences

            continue

        features[i, 0] = len(sentence)

        features[i, -len(sentence):] = sentence

    

    return features
# We are going to use a sequence length that does not truncates any of the samples

seq_length = max(max(map(len, train_tokens)), max(map(len, test_tokens))) 
# Applying padding

train_tokens = pad(train_tokens, seq_length)

test_tokens = pad(test_tokens, seq_length)
def get_embeddings(file):

    embeddings = {}

    with open(file, encoding="utf8", errors='ignore') as f:

        for line in tqdm(f):

            line_list = line.split(" ")

            if len(line_list) > 100:

                embeddings[line_list[0]] = np.array(line_list[1:], dtype='float32')

    return embeddings



def get_embeddings_matrix(vocab, lemma_vocab, embeddings, keyedVector=False):

    

    # Stemmers

    ps = PorterStemmer()

    lc = LancasterStemmer()

    sb = SnowballStemmer("english")

    

    n_words = len(vocab)

    if keyedVector:

        emb_size = embeddings.vector_size

    else:

        emb_size = next(iter(embeddings.values())).shape[0]

        

    # If word2vec, convert it to dict for simplicity and compatibility with the others vectors

    if keyedVector:

        emb_dict = {}

        for word in vocab:

            try:

                emb_dict[word] = embeddings.get_vector(word)

            except:

                continue

        embeddings = emb_dict

    

    embedding_matrix = np.zeros((n_words+1, emb_size), dtype=np.float32)

    unknown_vec = np.zeros((emb_size,), dtype=np.float32) - 1 # (-1, -1, ..., -1)

    unknown_words = 0  # unknown words counter  

    for word in tqdm(vocab):

        emb_vec = embeddings.get(word)

        if emb_vec is not None:

            embedding_matrix[vocab[word]] = emb_vec

            continue

            

        # Lemmatizing

        emb_vec = embeddings.get(lemma_vocab[word])

        if emb_vec is not None:

            embedding_matrix[vocab[word]] = emb_vec

            continue

            

        # Stemming

        emb_vec = embeddings.get(ps.stem(word))

        if emb_vec is not None:

            embedding_matrix[vocab[word]] = emb_vec

            continue

        emb_vec = embeddings.get(lc.stem(word))

        if emb_vec is not None:

            embedding_matrix[vocab[word]] = emb_vec

            continue    

        emb_vec = embeddings.get(sb.stem(word))

        if emb_vec is not None:

            embedding_matrix[vocab[word]] = emb_vec

            continue

        

        # If word vector not found

        embedding_matrix[vocab[word]] = unknown_vec

        unknown_words += 1

        

    print('% known words: {:.2%}'.format(1 - unknown_words/n_words))

            

    return embedding_matrix
# Getting embeddings from file

glove_file = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

glove_emb = get_embeddings(glove_file)



glove_emb_matrix = get_embeddings_matrix(vocab, lemma_vocab, glove_emb)



# Cleaning up memory

del glove_emb

gc.collect()
# Getting embeddings from file

fasttext_file = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

fasttext_emb = get_embeddings(fasttext_file)



# Building embedding matrix

fasttext_emb_matrix = get_embeddings_matrix(vocab, lemma_vocab, fasttext_emb)



# Cleaning up memory

del fasttext_emb

gc.collect()

# Getting embeddings from file

word2vec_file = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

word2vec_emb = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)



# Building embedding matrix

word2vec_emb_matrix = get_embeddings_matrix(vocab, lemma_vocab, word2vec_emb, keyedVector=True)



# Cleaning up memory

del word2vec_emb

gc.collect()
# Getting embeddings from file

paragram_file = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

paragram_emb = get_embeddings(paragram_file)



# Building embedding matrix

paragram_emb_matrix = get_embeddings_matrix(vocab, lemma_vocab, paragram_emb)



# Cleaning up memory

del paragram_emb

gc.collect()
emb_matrix = np.concatenate((glove_emb_matrix, 

                             paragram_emb_matrix), axis=1)



del glove_emb_matrix, fasttext_emb_matrix, word2vec_emb_matrix, paragram_emb_matrix

gc.collect()
# Concatenating tokens and statistical features

train_feat = np.concatenate((train_stat, train_tokens), axis=1)

test_feat = np.concatenate((test_stat, test_tokens), axis=1)
x_train, x_val, label_train, label_val = train_test_split(train_feat, labels, test_size=0.1, random_state=0) 



# Create Tensor datasets

train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(label_train))

valid_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(label_val))

test_data = TensorDataset(torch.from_numpy(test_feat))



# Create Dataloaders

batch_size = 64



train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
# Checking if GPU is available

train_on_gpu=torch.cuda.is_available()



if train_on_gpu:

    print('Training on GPU.')

else:

    print('No GPU available, training on CPU.')
def init_emb_layer(self, embedding_matrix):

    '''

    Function to help in the creation of the embedding layer

    '''

    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    num_emb, emb_size = embedding_matrix.size()

    emb_layer = nn.Embedding.from_pretrained(embedding_matrix)

    return emb_layer



class SelfAttention(nn.Module):

    '''

    Class that implements a Self-Attention module that will be applied on the outputs of the GRU layer

    '''

    

    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):

        super(SelfAttention, self).__init__()



        self.batch_first = batch_first

        self.attention_weights = nn.Parameter(torch.FloatTensor(attention_size))

        self.softmax = nn.Softmax(dim=-1)



        if non_linearity == "relu":

            self.non_linearity = nn.ReLU()

        else:

            self.non_linearity = nn.Tanh()



        nn.init.uniform(self.attention_weights.data, -0.005, 0.005)



    def get_mask(self, attentions, lengths):

        """

        Construct mask for padded itemsteps, based on lengths

        """

        max_len = max(lengths.data)

        mask = torch.autograd.Variable(torch.ones(attentions.size())).detach()



        if attentions.data.is_cuda:

            mask = mask.cuda()



        for i, l in enumerate(lengths.data):  # skip the first sentence

            if l < max_len:

                mask[i, :-l] = 0

        return mask



    def forward(self, inputs, lengths):



        # STEP 1 - perform dot product of the attention vector and each hidden state

        

        # inputs is a 3D Tensor: batch, len, hidden_size

        # scores is a 2D Tensor: batch, len

        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        scores = self.softmax(scores)



        # Step 2 - Masking



        # construct a mask, based on the sentence lengths

        mask = self.get_mask(scores, lengths)



        # apply the mask - zero out masked timesteps

        masked_scores = scores * mask



        # re-normalize the masked scores

        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row

        scores = masked_scores.div(_sums)  # divide by row sum



        # Step 3 - Weighted sum of hidden states, by the attention scores



        # multiply each hidden state with the attention weights

        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))



        # sum the hidden states

        representations = weighted.sum(1).squeeze()



        return representations



class Quora_model(nn.Module):

    def __init__(self, hidden_layer_dim, embedding_matrix, hidden_dim, gru_layers, stat_layers, drop_prob=0.5):

        """

        Quora model with bi-directional GRU and self-attention merged with Dense layers

        """

        super(Quora_model, self).__init__()

        

        self.hidden_layer_dim = hidden_layer_dim

        self.gru_layers = gru_layers

        self.emb_dim = embedding_matrix.shape[1]

        self.hidden_dim = hidden_dim   

        self.stat_layers = stat_layers

        

        # Dense layers for statistical features

        stat_in_dim = n_stats + 1 # including sequence length

        modules = []

        for out_dim in self.stat_layers:

            modules.append(nn.Linear(stat_in_dim, out_dim))

            modules.append(nn.ReLU())

            stat_in_dim = out_dim

        

        self.stat_dense = nn.Sequential(*modules)

        

        # Embedding layer

        self.embedding = init_emb_layer(self, embedding_matrix)

        

        # Bidirectional GRU layer

        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, self.gru_layers, 

                          batch_first=True, bidirectional=True, dropout = drop_prob)

        

        # Attention layer

        self.attention = SelfAttention(self.hidden_dim*2, batch_first=True)

        

        # Final dense --- merger of text and statistical features

        self.final_dense = nn.Sequential(

            nn.Dropout(p=drop_prob),

            nn.Linear(self.hidden_dim*2 + out_dim, self.hidden_layer_dim),

            nn.ReLU(),

            nn.Dropout(p=drop_prob),

            nn.Linear(self.hidden_layer_dim, 1),

            nn.Sigmoid()

        )

        

    def forward(self, x, hidden):

        

        batch_size, _ = x.size()

        

        # Deal with cases were the current batch_size is different from general batch_size

        # It occurrs at the end of iteration with the Dataloaders

        if hidden.size(1) != batch_size:

            hidden = hidden[:, :batch_size, :].contiguous()

            

        # Lengths of sequences

        lengths = x[:,n_stats].cpu().numpy().astype(int)

        

        # Adapting seq_len for the current batch

        seq_len = max(lengths) 

        x_text = x[:, -seq_len:] # input to gru layer

        x_stat = x[:, :n_stats+1].type(torch.FloatTensor) # include sequence length as statistical feature

        if train_on_gpu:

            x_stat = x_stat.cuda()

        

        # Apply embedding

        x_text = self.embedding(x_text)

        

        # GRU Layer

        out_gru, _ = self.gru(x_text, hidden)

        

        # Apply attention

        out_att = self.attention(out_gru, lengths)

        

        # Dense layer for statistical features

        out_stat = self.stat_dense(x_stat)

        

        # Concatenate output of the RNN with output from statistical features

        out = torch.cat((out_att, out_stat), dim=1)

        

        # Final dense_layer

        out = self.final_dense(out)

        

        return out

    

    def init_hidden(self, batch_size):

        ''' Initializes hidden state '''

        # Create a new tensor with sizes n_layers x batch_size x hidden_dim,

        # initialized to zero

        

        weight = next(self.parameters()).data

        

        if train_on_gpu:

            hidden = weight.new(self.gru_layers*2, batch_size, self.hidden_dim).zero_().cuda()

            

        else:

            hidden = weight.new(self.gru_layers*2, batch_size, self.hidden_dim).zero_()

        

        return hidden
hidden_dim = 256

gru_layers = 1

dropout = 0.1

stat_layers_dim = [16, 8] 

hidden_layer_dim = 64



# Initiating the model

model = Quora_model(hidden_layer_dim, emb_matrix, hidden_dim, gru_layers, stat_layers_dim, dropout)

model
# Training parameters



epochs = 4



print_every = 1000

early_stop = 20

clip = 5 # gradient clipping - to avoid gradient explosion



lr=0.001



# Defining loss and optimization functions



criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def train_model(model, train_loader, valid_loader, batch_size, epochs, 

                optimizer, criterion, clip, print_every, early_stop):

    

    # move model to GPU, if available

    if(train_on_gpu):

        model.cuda()

    

    counter = 0

    

    # Model in training mode

    model.train()

    breaker = False

    for e in range(epochs):



        # Batch loop

        for inputs, labels in train_loader:

            counter += 1



            # move data to GPU, if available

            if(train_on_gpu):

                inputs, labels = inputs.cuda(), labels.cuda()



            # Initialize hidden state

            h = model.init_hidden(batch_size)



            # Setting accumulated gradients to zero before backward step

            model.zero_grad()



            # Output from the model

            output = model(inputs, h)



            # Calculate the loss and do backprop step

            loss = criterion(output.squeeze(), labels.float())

            loss.backward()



            # Clipping the gradient to avoid explosion

            nn.utils.clip_grad_norm_(model.parameters(), clip)



            # Backpropagation step

            optimizer.step()



            # Validation stats

            if counter % print_every == 0:



                with torch.no_grad():



                    # Get validation loss and F1-score on validation set



                    val_losses = []

                    all_val_labels = []

                    all_val_preds = []

                    all_val_probs = []



                    # Model in evaluation mode

                    model.eval()

                    for inputs, labels in valid_loader:



                        all_val_labels += list(labels)



                        # Sending data to GPU

                        if(train_on_gpu):

                            inputs, labels = inputs.cuda(), labels.cuda()



                        # Initiating hidden state for the validation set

                        val_h = model.init_hidden(batch_size)



                        output = model(inputs, val_h)



                        # Computing validation loss

                        val_loss = criterion(output.squeeze(), labels.float())



                        val_losses.append(val_loss.item())



                        # Computing validation F1-score for threshold 0.5



                        preds = torch.round(output.squeeze())  # 1 if output probability >= 0.5

                        preds = np.squeeze(preds.cpu().numpy())

                        all_val_preds += list(preds)

                        

                        output = np.squeeze(output.cpu().detach().numpy())

                        all_val_probs += list(output)



                current_loss = np.mean(val_losses)

                

                print("Epoch: {}/{}...".format(e+1, epochs),

                      "Step: {}...".format(counter),

                      "Loss: {:.6f}...".format(loss.item()),

                      "Val Loss: {:.6f}...".format(current_loss),

                      "F1-score (threshold=0.5): {:.3%}".format(f1_score(all_val_labels, all_val_preds)))

                

                # Saving the best model and stopping if there is no improvement after "early_stop" evaluations

                    

                if  counter == print_every or current_loss < best_loss: # first evaluation or improvement

                    best_loss = current_loss

                    best_val_labels = all_val_labels

                    best_probs = all_val_probs

                    torch.save(model.state_dict(), 'checkpoint.pth')

                    counter_eval = 0 

                    

                counter_eval += 1

                if counter_eval == early_stop:

                    breaker = True

                    break



                # Put model back to training mode

                model.train()

        

        # breaking outer loop on epochs

        if breaker:

            break

    

    # Loading best model

    state_dict = torch.load('checkpoint.pth')

    model.load_state_dict(state_dict)

    

    return best_probs, best_val_labels
t0 = time.time()

all_val_probs, all_val_labels = train_model(model, train_loader, valid_loader, batch_size, epochs, 

                                            optimizer, criterion, clip, print_every, early_stop)

tf = time.time()

print("\nExecution time: {:.2f}min".format((tf-t0)/60))
best_score = 0

for thr in np.arange(0.0, 0.5, 0.005):

    pred = np.array(all_val_probs > thr, dtype=int)

    score = f1_score(all_val_labels, pred)

    print("Threshold: {:.3f}... F1-score {:.3%}".format(thr, score))

    if score > best_score:

        best_score = score

        best_thr = thr

print("\nBest threshold: {:.3f}... F1-score {:.3%}".format(best_thr, best_score))
# Model in evaluation mode

model.eval()



with torch.no_grad():

    all_test_preds = []



    for inputs in test_loader:

        inputs = inputs[0]

        

        # Sending data to GPU

        if(train_on_gpu):

            inputs = inputs.cuda()

            

        test_h = model.init_hidden(batch_size)

        

        output = model(inputs, test_h)

        

        preds = (output.squeeze() > best_thr).type(torch.IntTensor)

        preds = np.squeeze(preds.cpu().numpy())

        all_test_preds += list(preds.astype(int))
sub = pd.DataFrame({

    'qid': test_df.qid,

    'prediction': all_test_preds

})



# Make sure the columns are in the correct order

sub = sub[['qid', 'prediction']]



sub.to_csv('submission.csv', index=False, sep=',')