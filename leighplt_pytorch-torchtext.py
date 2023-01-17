import numpy as np 

import pandas as pd 

import os, random, sys, time



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



import torchtext

from torchtext import vocab, data



from scipy.stats import spearmanr

from sklearn.model_selection import StratifiedKFold, KFold



import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '../input/glove-reddit-comments/')

from clean_text import RegExCleaner
DATA_PATH = '../input/google-quest-challenge'

EMB_PATH = '../input/embeddings-glove-crawl-torch-cached'

EMB_FILENAME = 'GloVe.Reddit.120B.512D.txt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_SPLITS = 4

BATCH_SIZE = 128
df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

subm = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'), index_col='qa_id')

subm.loc[:, :] = 0
# Just simple tokenizer

tknz = RegExCleaner.reddits()

def tokenizer(text):

    text = tknz(text).split()

    return text
## define the columns that we want to process and how to process



## text field, We have 2 text field, so we must fixed by length

txt_field = data.Field(sequential=True, tokenize=tokenizer, fix_length=125,

                       batch_first=False, include_lengths=False,  use_vocab=True)

## Numeric fields

num_field = data.Field(sequential=False, dtype=torch.float,  use_vocab=False)

idx_field = data.Field(sequential=False, dtype=torch.int,  use_vocab=False)

## Fields which don't need preprocessing

raw_field = data.RawField()



labels = df.columns[11:]

num_columns = list(zip(labels, [num_field]*len(labels)))

basic_columns = [

    ('qa_id', idx_field),

    ('question_title', raw_field),

    ('question_body', txt_field),

    ('question_user_name', raw_field),

    ('question_user_page', raw_field),

    ('answer', txt_field),

    ('answer_user_name', raw_field),

    ('answer_user_page', raw_field),

    ('url', raw_field),

    ('category', raw_field),

    ('host', raw_field),

]

train_fields = basic_columns + num_columns

test_fields = basic_columns
# Loading csv file

train_ds = data.TabularDataset(path=os.path.join(DATA_PATH, 'train.csv'), 

                           format='csv',

                           fields=train_fields, 

                           skip_header=True)



test_ds = data.TabularDataset(path=os.path.join(DATA_PATH, 'test.csv'), 

                           format='csv',

                           fields=test_fields, 

                           skip_header=True)
# Example

print('Chunk of answer: '+' '.join(train_ds.examples[1].answer[:10]))
# specify the path to the localy saved vectors

vec = vocab.Vectors(os.path.join(EMB_PATH, EMB_FILENAME), cache=EMB_PATH)

# build the vocabulary using train and validation dataset and assign the vectors

txt_field.build_vocab(train_ds, test_ds, max_size=300000, vectors=vec)



embs_vocab = train_ds.fields['question_body'].vocab.vectors

print('Embedding vocab size: ', embs_vocab.size()[0])

vocab_size = embs_vocab.size()[0]
# Wrapper for loaders, which structured fields

class BatchWrapper:

      def __init__(self, dataloader, mode='train'):

            self.dataloader, self.mode = dataloader, mode

     

      def __iter__(self):

            if self.mode =='test':

                for batch in self.dataloader:

                    yield (batch.qa_id, batch.question_body, batch.answer)

            else:

                for batch in self.dataloader:

                    target = torch.stack([getattr(batch, label) for label in labels], dim=-1)

                    yield (batch.question_body,  batch.answer, target)

  

      def __len__(self):

            return len(self.dl)



def wrapper(ds, mode='train', **kwargs):

    dataloader = data.BucketIterator(ds, device=DEVICE, **kwargs)

    return BatchWrapper(dataloader, mode)



def splits_cv(dataset, cv, y=None, batch_size=BATCH_SIZE):

    """

        Split dataset to train and validation used cross-validator and wrap loader

    """

    for indices in cv.split(range(len(dataset)), y):

        (train_data, valid_data) = tuple([dataset.examples[i] for i in index] for index in indices)

        yield tuple(wrapper(data.Dataset(d, dataset.fields), batch_size=batch_size) for d in (train_data, valid_data) if d)

        

cv = KFold(n_splits=N_SPLITS, random_state=6699)
test_loader = wrapper(test_ds, batch_size=BATCH_SIZE, shuffle=False, repeat=False, mode='test')
class RNN_QA(nn.Module):

    def __init__(self, embs_vocab, hidden_size=64, layers=1,

                 dropout=0., bidirectional=False, num_classes=30):

        super().__init__()



        coef = 2 if bidirectional else 1

        dropout = dropout if layers > 1 else 0

        self.emb = nn.Embedding.from_pretrained(embs_vocab, freeze=True)

                

        self.question = nn.LSTM(embs_vocab.size(1), hidden_size,

                            num_layers=layers, bidirectional=bidirectional, dropout=dropout)

        

        self.answer = nn.LSTM(embs_vocab.size(1), hidden_size,

                            num_layers=layers, bidirectional=bidirectional, dropout=dropout)

        

        self.classifier = nn.Sequential(

                nn.Linear(2*hidden_size*coef, 64),

                nn.ReLU(inplace=True),

                nn.Linear(64, num_classes)

            )

                

    def forward(self, q, a):

        

        q = self.emb(q)

        a = self.emb(a)

        

        q_rnn, _ = self.question(q)

        a_rnn, _ = self.answer(a)

        

        q_rnn, _ = q_rnn.max(dim=0, keepdim=False) 

        a_rnn, _ = a_rnn.max(dim=0, keepdim=False) 

        

        out = torch.cat([q_rnn, a_rnn], dim=-1)

        out = self.classifier(out).sigmoid()

        return out
def metric_fn(p, t):

    score = 0

    for i in range(p.shape[1]):

        score += np.nan_to_num(spearmanr(p[:,i], t[:,i])[0])

    score /= 30

    return score



@torch.no_grad()

def validation_fn(model, loader, loss_fn):

    y_pred, y_true, tloss = [], [], []

    for q, a, target in loader:

        outputs = model(q, a)

        loss = loss_fn(outputs, target)

        tloss.append(loss.item())

        y_true.append(target.detach().cpu().numpy())

        y_pred.append(outputs.detach().cpu().numpy())

        

    tloss = np.array(tloss).mean()

    y_pred = np.concatenate(y_pred)

    y_true = np.concatenate(y_true)

    metric = metric_fn(y_pred, y_true)

    return tloss, metric
### Table for results

header = r'''

           Train       Validation

Epoch | Loss |Spearm| Loss |Spearm| Time, m

'''

#          Epoch         metrics            time

raw_line = '{:6d}' + '\u2502{:6.3f}'*4 + '\u2502{:6.2f}'
def oof_preds(train_ds, test_loader, embs_vocab, epochs = 4):



    for loader, vloader in splits_cv(train_ds, cv):

        

        model = RNN_QA(embs_vocab, hidden_size=128, dropout=0.1, bidirectional=True).to(DEVICE)

        

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 1e-3,

                                                     betas=(0.75, 0.999), weight_decay=1e-3)

        loss_fn = torch.nn.BCELoss()

        print(header)

        for epoch in range(1,epochs+1):      

            y_pred, y_true = [], []

            start_time = time.time()

            tloss = []          

            model.train()

            

            for q, a, target in loader:

                optimizer.zero_grad()

                outputs = model(q, a)

                loss = loss_fn(outputs, target)

                tloss.append(loss.item())

                loss.backward()

                optimizer.step()

                y_true.append(target.detach().cpu().numpy())

                y_pred.append(outputs.detach().cpu().numpy())



            tloss = np.array(tloss).mean()

            y_pred = np.concatenate(y_pred)

            y_true = np.concatenate(y_true)

            tmetric = metric_fn(y_pred, y_true)



            vloss, vmetric = validation_fn(model, vloader, loss_fn)

            if epoch % 2 == 0:

                print(raw_line.format(epoch,tloss,tmetric,vloss,vmetric,(time.time()-start_time)/60**1))



       

        # Get prediction for test set

        qa_id, preds = [], [] 

        with torch.no_grad():

            for qaids, q, a in test_loader:

                outputs = model(q, a)

                qa_id.append(qaids.cpu().numpy())

                preds.append(outputs.detach().cpu().numpy())

            

        # Save prediction of test set

        qa_id = np.concatenate(qa_id)

        preds = np.concatenate(preds)

        subm.loc[qa_id, labels]  =  subm.loc[qa_id, labels].values + preds / N_SPLITS

        
oof_preds(train_ds, test_loader, embs_vocab, epochs = 20)
subm.to_csv('submission.csv')