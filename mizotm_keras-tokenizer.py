import time

import numpy as np

import pandas as pd 

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import spacy

import time

import gc
batch_size = 512

maxlen = 100

max_features = int(120e3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



train_y = train_df['target'].values

len_train = len(train_df)

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)



train_text = train_df['question_text']

test_text = test_df['question_text']

text_list = pd.concat([train_text, test_text])
start_time = time.time()

print("Spacy NLP ...")

nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])

nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

word_dict = {}

word_index = 1

lemma_dict = {}

docs = nlp.pipe(text_list, n_threads = 2)

word_sequences = []

for doc in tqdm(docs):

    word_seq = []

    for token in doc:

        if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):

            word_dict[token.text] = word_index

            word_index += 1

            lemma_dict[token.text] = token.lemma_

        if token.pos_ is not "PUNCT":

            word_seq.append(word_dict[token.text])

    word_sequences.append(word_seq)



del docs, train_text, test_text, text_list, train_df, test_df



import gc

gc.collect()



print("--- %s seconds ---" % (time.time() - start_time))
train_word_sequences = word_sequences[:len_train]

test_word_sequences = word_sequences[len_train:]



## Pad the sentences 

train_word_sequences = pad_sequences(train_word_sequences, maxlen=maxlen)

test_word_sequences = pad_sequences(test_word_sequences, maxlen=maxlen)
# borrowed from https://www.kaggle.com/wowfattie/3rd-place

from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.stem.lancaster import LancasterStemmer

lc = LancasterStemmer()

from nltk.stem import SnowballStemmer

sb = SnowballStemmer("english")



def load_glove(word_dict, lemma_dict = None):

    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    embed_size = 300

    nb_words = len(word_dict)+1

    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)

    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.

    print(unknown_vector[:5])

    for key in tqdm(word_dict):

        word = key

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.lower()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.upper()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.capitalize()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = ps.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lc.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = sb.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        if(lemma_dict is not None):    

            word = lemma_dict[key]

            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:

                embedding_matrix[word_dict[key]] = embedding_vector

                continue

        embedding_matrix[word_dict[key]] = unknown_vector                    

    return embedding_matrix, nb_words 



def load_fasttext(word_dict, lemma_dict = None):

    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    embed_size = 300

    nb_words = len(word_dict)+1

    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)

    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.

    print(unknown_vector[:5])

    for key in tqdm(word_dict):

        word = key

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.lower()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.upper()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.capitalize()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = ps.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lc.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = sb.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        if(lemma_dict is not None):    

            word = lemma_dict[key]

            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:

                embedding_matrix[word_dict[key]] = embedding_vector

                continue

        embedding_matrix[word_dict[key]] = unknown_vector                    

    return embedding_matrix, nb_words 



def load_para(word_dict, lemma_dict = None):

    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    embed_size = 300

    nb_words = len(word_dict)+1

    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)

    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.

    print(unknown_vector[:5])

    for key in tqdm(word_dict):

        word = key

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.lower()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.upper()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.capitalize()

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = ps.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lc.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = sb.stem(key)

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        if(lemma_dict is not None):    

            word = lemma_dict[key]

            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:

                embedding_matrix[word_dict[key]] = embedding_vector

                continue

        embedding_matrix[word_dict[key]] = unknown_vector                    

    return embedding_matrix, nb_words 
class CustomDataset(Dataset):



    def __init__(self, x, y):

        super(CustomDataset, self).__init__()

        self.x = torch.tensor(x, dtype = torch.long)

        self.y = torch.tensor(y, dtype = torch.float32)

    

    def __len__(self):

        return self.x.shape[0]



    def __getitem__(self, idx):

        x = self.x[idx, :]

        y = self.y[idx, :]

        return x, y



#train_ds = CustomDataset(train_word_sequences, train_y.reshape((len(train_df), 1)))



#train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

x_test = torch.tensor(test_word_sequences, dtype=torch.long)

test = torch.utils.data.TensorDataset(x_test)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
start_time = time.time()

print("Loading embedding matrix ...")



# For spacy tokenizer 

#embedding_matrix_glove, _ = load_glove(word_dict, lemma_dict)

#embedding_matrix_fasttext, _ = load_fasttext(word_dict, lemma_dict)

#embedding_matrix_para, _ = load_para(word_dict, lemma_dict)



# For keras tokenizer 

embedding_matrix_glove, _ = load_glove(word_dict)

embedding_matrix_fasttext, _ = load_fasttext(word_dict)

embedding_matrix_para, _ = load_para(word_dict)



embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_fasttext, embedding_matrix_para), axis=1)



del embedding_matrix_glove, embedding_matrix_fasttext, embedding_matrix_para

gc.collect()



print("--- %s seconds ---" % (time.time() - start_time))
x_batch = next(iter(test_loader))[0]

embed_size = 300

embed = nn.Embedding(max_features, 300)

x_batch.shape, embed(x_batch).shape

hidden_size = 10

lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

o, (h, c) = lstm(embed(x_batch))

o.shape, h.shape, c.shape



avg_pool = torch.mean(o, 1,)

max_pool, _ = torch.max(o, 1)

conc = torch.cat((avg_pool, max_pool), 1)

conc.shape, avg_pool.shape, max_pool.shape, c.shape, h.view(512, 2*hidden_size).shape
class SimpleNet(nn.Module):

    def __init__(self, embedding_matrix = None):

        super(SimpleNet, self).__init__()

        

        self.hidden_size = 128

        

        self.embed_size = 300 if(embedding_matrix is None) else embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, self.embed_size)

        if(embedding_matrix is not None):

            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

            self.embedding.weight.requires_grad = False

        

        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True, batch_first=True)

        

        self.linear = nn.Linear(self.hidden_size * 3*2, 32)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        self.out = nn.Linear(32, 1)

        

    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        

        h_lstm, (h, c) = self.lstm(h_embedding)

        

        avg_pool = torch.mean(h_lstm, 1)

        max_pool, _ = torch.max(h_lstm, 1)

        #print(avg_pool.shape, max_pool.shape, c.shape, h.view(batch_size, 2*self.hidden_size).shape)

        conc = torch.cat((avg_pool, max_pool, h.view(-1, 2*self.hidden_size)), 1)

        conc = self.relu(self.linear(conc))

        #conc = torch.cat((self.relu(self.linear(conc)), conc), 1)

        conc = self.dropout(conc)

        out = torch.sigmoid(self.out(conc))

        

        return out

    

def unfreeze_embedding(model):

    model.embedding.weight.requires_grad = True
class Attention(nn.Module):

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):

        super(Attention, self).__init__(**kwargs)

        

        self.supports_masking = True



        self.bias = bias

        self.feature_dim = feature_dim

        self.step_dim = step_dim

        self.features_dim = 0

        

        weight = torch.zeros(feature_dim, 1)

        nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

        

        if bias:

            self.b = nn.Parameter(torch.zeros(step_dim))

        

    def forward(self, x, mask=None):

        feature_dim = self.feature_dim

        step_dim = self.step_dim



        eij = torch.mm(

            x.contiguous().view(-1, feature_dim), 

            self.weight

        ).view(-1, step_dim)

        

        if self.bias:

            eij = eij + self.b

            

        eij = torch.tanh(eij)

        a = torch.exp(eij)

        

        if mask is not None:

            a = a * mask



        a = a / torch.sum(a, 1, keepdim=True) + 1e-10



        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1)

    

class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix):

        super(NeuralNet, self).__init__()

        

        hidden_size = 128

        

        self.embed_size = 300 if(embedding_matrix is None) else embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, self.embed_size)

        if(embedding_matrix is not None):

            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

            self.embedding.weight.requires_grad = False

        

        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(self.embed_size, hidden_size, bidirectional=True, batch_first=True)

        self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)

        

        self.lstm_attention = Attention(hidden_size*2, maxlen)

        self.gru_attention = Attention(hidden_size*2, maxlen)

        

        self.linear = nn.Linear(1024, 16)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        self.out = nn.Linear(16, 1)

        

    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        

        h_lstm, _ = self.lstm(h_embedding)

        h_gru, _ = self.gru(h_lstm)

        

        h_lstm_atten = self.lstm_attention(h_lstm)

        h_gru_atten = self.gru_attention(h_gru)

        

        avg_pool = torch.mean(h_gru, 1)

        max_pool, _ = torch.max(h_gru, 1)

        

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)

        conc = self.relu(self.linear(conc))

        conc = self.dropout(conc)

        out  = torch.sigmoid(self.out(conc))

        

        return out
def train_model(model, x_train, y_train, x_val, y_val, n_epochs, optimizer = None, validate=True):

    

    if(optimizer is None):

        optimizer = torch.optim.Adam(model.parameters())

    

    train_ds = CustomDataset(x_train, y_train)

    valid_ds  = CustomDataset(x_val, y_val)



    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    

    loss_fn = torch.nn.BCELoss(reduction='mean').cuda()

    best_score = -np.inf

    

    for epoch in range(n_epochs):

        start_time = time.time()

        model.train()

        avg_loss = 0.

        

        for x_batch, y_batch in tqdm(train_loader):

            

            x_batch = x_batch.to(device)

            y_batch = y_batch.to(device)

            

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)



            optimizer.zero_grad()



            loss.backward()



            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

            

        model.eval()

        

        valid_preds = np.zeros((x_val_fold.shape[0]))

        

        if validate:

            avg_val_loss = 0.

            for i, (x_batch, y_batch) in tqdm(enumerate(valid_loader)):

                x_batch = x_batch.to(device)

                y_batch = y_batch.to(device)

                y_pred = model(x_batch).detach()



                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

                valid_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]

            search_result = threshold_search(y_val, valid_preds)



            val_f1, val_threshold = search_result['f1'], search_result['threshold']

            elapsed_time = time.time() - start_time

            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} best_t={:.2f} \t time={:.2f}s'.format(

                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_f1, val_threshold, elapsed_time))

        else:

            elapsed_time = time.time() - start_time

            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(

                epoch + 1, n_epochs, avg_loss, elapsed_time))

    

    valid_preds = np.zeros((x_val_fold.shape[0]))

    

    avg_val_loss = 0.

    for i, (x_batch, y_batch) in tqdm(enumerate(valid_loader)):

        x_batch = x_batch.to(device)

        y_batch = y_batch.to(device)

        y_pred = model(x_batch).detach()



        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

        valid_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]



    print('Validation loss: ', avg_val_loss)



    test_preds = np.zeros((len(test_loader.dataset)))

    

    for i, (x_batch,) in tqdm(enumerate(test_loader)):

        

        x_batch = x_batch.to(device)

        y_pred = model(x_batch).detach()



        test_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]

    

    return valid_preds, test_preds, optimizer



def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result



def seed_everything(seed=1234):

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()
from sklearn.model_selection import StratifiedKFold



splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=10).split(train_word_sequences, train_y))



test_preds = np.zeros((len(test_word_sequences), len(splits)))

train_preds = np.zeros(len(train_word_sequences))



n_epochs = 4

from sklearn.metrics import f1_score



seed = 10



for i, (train_idx, valid_idx) in enumerate(splits):    

    #i = 0

    train_idx, valid_idx =  splits[i]

    x_train_fold = train_word_sequences[train_idx]

    y_train_fold = train_y[train_idx, np.newaxis]

    x_val_fold = train_word_sequences[valid_idx]

    y_val_fold = train_y[valid_idx, np.newaxis]

    print('Fold {}'.format(i+1))



    seed_everything(seed + i)

    model = SimpleNet(embedding_matrix)

    #model = NeuralNet(embedding_matrix)

    model.to(device)



    valid_preds_fold, test_preds_fold, opt = train_model(model,

                                                    x_train_fold, 

                                                    y_train_fold, 

                                                    x_val_fold, 

                                                    y_val_fold, 

                                                    n_epochs,

                                                    validate=True)



    test_preds[:, i] = test_preds_fold

    train_preds[valid_idx] = valid_preds_fold
unfreeze_embedding(model)



for param_group in opt.param_groups:

        param_group['lr'] = 1e-4



valid_preds_fold, test_preds_fold, opt = train_model(model,

                                                x_train_fold, 

                                                y_train_fold, 

                                                x_val_fold, 

                                                y_val_fold, 

                                                1,

                                                opt,

                                                validate=True)
sub = pd.read_csv('../input/sample_submission.csv')



search_result = threshold_search(train_y, train_preds)

sub['prediction'] = test_preds.mean(1) > search_result['threshold']

sub.to_csv("submission.csv", index=False)