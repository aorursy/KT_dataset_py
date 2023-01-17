import gc

import os

import time

import numpy as np

import pandas as pd

from tqdm import tqdm



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import torch

import torch.nn as nn

import torch.utils.data



tqdm.pandas()



print(os.listdir("../input"))

TEXT_COL = 'comment_text'

EMB_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', index_col='id')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



def load_embeddings(embed_dir=EMB_PATH):

    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))

    return embedding_index



def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):

    embedding_matrix = np.zeros((max_features, 300))

    for word, i in tqdm(word_index.items(),disable = not verbose):

        if lower:

            word = word.lower()

        if i >= max_features: continue

        try:

            embedding_vector = embeddings_index[word]

        except:

            embedding_vector = embeddings_index["unknown"]

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    return embedding_matrix



def build_matrix(word_index, embeddings_index):

    embedding_matrix = np.zeros((len(word_index) + 1,300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embeddings_index[word]

        except:

            embedding_matrix[i] = embeddings_index["unknown"]

    return embedding_matrix
maxlen = 220

max_features = 100000

embed_size = 300

tokenizer = Tokenizer(num_words=max_features, lower=True) #filters = ''

#tokenizer = text.Tokenizer(num_words=max_features)

print('fitting tokenizer')

tokenizer.fit_on_texts(list(train[TEXT_COL]) + list(test[TEXT_COL]))

word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(list(train[TEXT_COL]))

train['target'] = train['target'].apply(lambda x: 1 if x > 0.5 else 0)

y_train = train['target'].values

X_test = tokenizer.texts_to_sequences(list(test[TEXT_COL]))



X_train = pad_sequences(X_train, maxlen=maxlen)

X_test = pad_sequences(X_test, maxlen=maxlen)





del tokenizer

gc.collect()
embeddings_index = load_embeddings()
embedding_matrix = build_matrix(word_index, embeddings_index)
del embeddings_index

gc.collect()
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
# Refactored based on reasonable remarks

# of @ddanevskyi https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/79911

class NeuralNet(nn.Module):

    def __init__(self):

        super(NeuralNet, self).__init__()

        

        hidden_size = 64

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        

        self.embedding_dropout = nn.Dropout2d(0.2) 

        self.lstm = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)

        

        self.lstm_attention = Attention(hidden_size*2, maxlen)

        

        self.out = nn.Linear(384, 1)



        

    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = self.embedding_dropout(h_embedding.transpose(1,2).unsqueeze(-1)).squeeze().transpose(1,2)



        h_lstm, _ = self.lstm(h_embedding)

        h_lstm_atten = self.lstm_attention(h_lstm)



        avg_pool = torch.mean(h_lstm, 1)

        max_pool, _ = torch.max(h_lstm, 1)

        

        conc = torch.cat((h_lstm_atten, avg_pool, max_pool), 1)

        out = self.out(conc)

        

        return out

    

def sigmoid(x):

    return 1 / (1 + np.exp(-x))
# Stolen from https://github.com/Bjarten/early-stopping-pytorch

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf



    def __call__(self, val_loss, model):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), 'checkpoint.pt')

        self.val_loss_min = val_loss
from sklearn.model_selection import KFold

splits = list(KFold(n_splits=5).split(X_train, y_train))
BATCH_SIZE = 2048

NUM_EPOCHS = 100



train_preds = np.zeros((len(X_train)))

test_preds = np.zeros((len(X_test)))



x_test_cuda = torch.tensor(X_test, dtype=torch.long).cuda()

test = torch.utils.data.TensorDataset(x_test_cuda)

test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)



for i, (train_idx, valid_idx) in enumerate(splits):

    x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()

    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()

    x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()

    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()

    

    model = NeuralNet()

    model.cuda()

    

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    optimizer = torch.optim.Adam(model.parameters())

    

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    

    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)

    

    early_stopping = EarlyStopping(patience=3, verbose=True)

    

    print(f'Fold {i + 1}')

    

    for epoch in range(NUM_EPOCHS):

        start_time = time.time()

        

        model.train()

        avg_loss = 0.

        for x_batch, y_batch in tqdm(train_loader, disable=True):

            optimizer.zero_grad()

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)

            loss.backward()

            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

        

        model.eval()

        valid_preds_fold = np.zeros((x_val_fold.size(0)))

        test_preds_fold = np.zeros(len(X_test))

        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(valid_loader):

            y_pred = model(x_batch).detach()

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

            valid_preds_fold[i * BATCH_SIZE:(i+1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]

        

        elapsed_time = time.time() - start_time 

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(

            epoch + 1, NUM_EPOCHS, avg_loss, avg_val_loss, elapsed_time))

        

        early_stopping(avg_val_loss, model)

        

        if early_stopping.early_stop:

            print("Early stopping")

            break

        

        

    # load the last checkpoint with the best model

    model.load_state_dict(torch.load('checkpoint.pt'))

    

    for i, (x_batch,) in enumerate(test_loader):

        y_pred = model(x_batch).detach()



        test_preds_fold[i * BATCH_SIZE:(i+1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]



    train_preds[valid_idx] = valid_preds_fold

    test_preds += test_preds_fold / len(splits)    
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train>0.5, train_preds)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

submission['prediction'] = test_preds

submission.reset_index(drop=False, inplace=True)

submission.head()
submission.to_csv('submission.csv', index=False)