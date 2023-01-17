import numpy as np

import pandas as pd

import os, time, gc, pickle, random

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from keras.preprocessing import text, sequence

import torch

from torch import nn

from torch.utils import data

from torch.nn import functional as F
# disable progress bars when submitting

def is_interactive():

    return 'SHLVL' not in os.environ



if not is_interactive():

    def nop(it, *a, **k):

        return it



    tqdm = nop
def seed_everything(seed=1234):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()
CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'

GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'



NUM_MODELS = 2

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

MAX_LEN = 220
def build_matrix(word_index, emb_path, unknown_token='unknown'):

    with open(emb_path, 'rb') as fp:

        embedding_index = pickle.load(fp)

    

    # TODO: Build random token instead of using unknown

    unknown_token = embedding_index[unknown_token].copy()

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word].copy()

        except KeyError:

            embedding_matrix[i] = unknown_token

            unknown_words.append(word)

            

    del embedding_index; gc.collect()

    return embedding_matrix, unknown_words
class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)    # (N, T, 1, K)

        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)

        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked

        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)

        x = x.squeeze(2)  # (N, T, K)

        return x

    

class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix, num_aux_targets):

        super(NeuralNet, self).__init__()

        embed_size = embedding_matrix.shape[1]

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout(0.3)

        

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

    

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)

        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

        

    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = self.embedding_dropout(h_embedding)

        

        h_lstm1, _ = self.lstm1(h_embedding)

        h_lstm2, _ = self.lstm2(h_lstm1)

        

        # global average pooling

        avg_pool = torch.mean(h_lstm2, 1)

        # global max pooling

        max_pool, _ = torch.max(h_lstm2, 1)

        

        h_conc = torch.cat((max_pool, avg_pool), 1)

        h_conc_linear1  = F.relu(self.linear1(h_conc))

        h_conc_linear2  = F.relu(self.linear2(h_conc))

        

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        

        result = self.linear_out(hidden)

        aux_result = self.linear_aux_out(hidden)

        out = torch.cat([result, aux_result], 1)

        

        return out
def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



x_test = preprocess(test['comment_text'])

x_train = preprocess(train['comment_text'])



y_train = np.where(train['target'] >= 0.5, 1, 0)

y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test  = tokenizer.texts_to_sequences(x_test)

x_train_lens = [len(i) for i in x_train]

x_test_lens  = [len(i) for i in x_test]
max_features = None

max_features = max_features or len(tokenizer.word_index) + 1

max_features
crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, emb_path=CRAWL_EMBEDDING_PATH, unknown_token='unknown')

print('n unknown words (crawl): ', len(unknown_words_crawl))
glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, emb_path=GLOVE_EMBEDDING_PATH, unknown_token='unknown')

print('n unknown words (glove): ', len(unknown_words_glove))
embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

embedding_matrix.shape



del crawl_matrix

del glove_matrix

gc.collect()
class TextDataset(data.Dataset):

    def __init__(self, text, lens, y=None):

        self.text = text

        self.lens = lens

        self.y = y



    def __len__(self):

        return len(self.lens)



    def __getitem__(self, idx):

        if self.y is None:

            return self.text[idx], self.lens[idx]

        return self.text[idx], self.lens[idx], self.y[idx]
class Collator(object):

    def __init__(self,test=False,percentile=100):

        self.test = test

        self.percentile = percentile

        

    def __call__(self, batch):

        global MAX_LEN

        

        if self.test:

            texts, lens = zip(*batch)

        else:

            texts, lens, target = zip(*batch)



        lens = np.array(lens)

        max_len = min(int(np.percentile(lens, self.percentile)), MAX_LEN)

        texts = torch.tensor(sequence.pad_sequences(texts, maxlen=max_len), dtype=torch.long).cuda()

        

        if self.test:

            return texts

        

        return texts, torch.tensor(target, dtype=torch.float32).cuda()
final_y_train = np.hstack([y_train[:, np.newaxis], y_aux_train])



train_collate = Collator(percentile=96)

train_dataset = TextDataset(x_train, x_train_lens, final_y_train)

train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=train_collate)



test_collate = Collator(test=True)

test_dataset = TextDataset(x_test, x_test_lens)

test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False , collate_fn=test_collate)



# del y_train, y_aux_train; gc.collect()
def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def train_model(model, train_loader, test_loader, loss_fn, output_dim, lr=0.001,

                batch_size=512, n_epochs=4,

                enable_checkpoint_ensemble=True):

    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]

    optimizer = torch.optim.Adam(param_lrs, lr=lr)



    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    

    all_test_preds = []

    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

    

    for epoch in range(n_epochs):

        start_time = time.time()

        

        scheduler.step()

        

        model.train()

        avg_loss = 0.

        

        for step, (seq_batch, y_batch) in enumerate(tqdm(train_loader, disable=False)):

            y_pred = model(seq_batch)            

            loss = loss_fn(y_pred, y_batch)



            optimizer.zero_grad()

            loss.backward()



            optimizer.step()

            avg_loss += loss.item() #/ len(train_loader)

            

            if step > 0 and step % 100 == 0:

                print(step, avg_loss / step)

            

        model.eval()

        test_preds = np.zeros((len(test), output_dim))

    

        for step, seq_batch in enumerate(test_loader):

            y_pred = sigmoid(model(seq_batch).detach().cpu().numpy())

            test_preds[step * batch_size:step * batch_size + y_pred.shape[0], :] = y_pred[:,:1]



        all_test_preds.append(test_preds)

        elapsed_time = time.time() - start_time

        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(

              epoch + 1, n_epochs, avg_loss / len(train_loader), elapsed_time))



    if enable_checkpoint_ensemble:

        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    

    else:

        test_preds = all_test_preds[-1]

        

    return test_preds
aux_size = final_y_train.shape[-1] - 1  # targets



all_test_preds = []

for model_idx in range(NUM_MODELS):

    print('Model ', model_idx)

    seed_everything(1234 + model_idx)

    

    model = NeuralNet(embedding_matrix, aux_size)

    model.cuda()

    

    test_preds = train_model(model, train_loader, test_loader, output_dim=1, loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))

    all_test_preds.append(test_preds)

    print()
submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': np.mean(all_test_preds, axis=0)[:, 0]

})



submission.to_csv('submission.csv', index=False)