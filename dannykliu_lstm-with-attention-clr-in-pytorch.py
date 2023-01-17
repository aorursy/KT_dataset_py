# import standard numerical packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# import pytorch modules
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
# seed so results are reproducible
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# read training data 
data = pd.read_csv("../input/train.csv")
# split into validation and train sets
data_train, data_val = train_test_split(data, test_size=0.12, random_state=6)
# read testing data
data_test = pd.read_csv('../input/test.csv')
# check proportion of positive examples in train and val set
print(np.sum(data_train.target) / len(data_train))
print(np.sum(data_val.target) / len(data_val))
# text preprocessing inspired from following kernel: https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/notebook
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
char_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }    

def clean_text(x):
    for dic in [contraction_mapping, mispell_dict, char_mapping]:
        for word in dic.keys():
            x = x.replace(word, dic[word])
    return x

# apply text cleaning to training, validation and test set
pd.set_option('mode.chained_assignment', None) # ignore copy on slice of DataFrame warning
data_train['question_text'] = data_train['question_text'].fillna("").apply(lambda x: clean_text(x))
data_val['question_text'] = data_val['question_text'].fillna("").apply(lambda x: clean_text(x))
data_test['question_text'] = data_test['question_text'].fillna("").apply(lambda x: clean_text(x))
# save files to disk and remove unncessary memory afterwards
!rm -rf split_data
!mkdir split_data
data_train.to_csv('split_data/train.csv', index=False)
data_val.to_csv('split_data/val.csv', index=False)
data_test.to_csv('split_data/test.csv', index=False)
del data, data_train, data_val, data_test 
# initialize torchtext Field objects 
text = torchtext.data.Field(lower=True, batch_first=True, tokenize='spacy', include_lengths=True)
target = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
qid = torchtext.data.Field()
# use field objects to read training, validation and test sets
train = torchtext.data.TabularDataset(path='split_data/train.csv', format='csv',
                                      fields={'question_text': ('text',text),
                                              'target': ('target',target)})
val = torchtext.data.TabularDataset(path='split_data/val.csv', format='csv',
                                    fields={'question_text': ('text',text),
                                              'target': ('target',target)})
test = torchtext.data.TabularDataset(path='split_data/test.csv', format='csv',
                                     fields={'qid': ('qid', qid),
                                             'question_text': ('text',text)})
# build vocabulary object from datasets
text.build_vocab(train, val, test, min_freq=3)
qid.build_vocab(test)
# load glove embedding into vocab object
text.vocab.load_vectors(torchtext.vocab.Vectors("../input/embeddings/glove.840B.300d/glove.840B.300d.txt"))
print(text.vocab.vectors.shape)
print(f"Unique tokens in text vocabulary: {len(text.vocab)}")
# helper functions to be used later
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_metrics(outs, y):
    outs = sigmoid(outs.cpu().data.numpy())
    y = y.cpu().data.numpy()
    y_pred = (outs >= 0.5).astype(int)
    acc = np.sum(y_pred == y) / len(y)
    tp = np.sum((y_pred == y) & (y_pred == 1))
    fp = np.sum((y_pred != y) & (y_pred == 1))
    fn = np.sum((y_pred != y) & (y_pred == 0))
    return acc, tp, fp, fn
# initialize iterators over datasets. we will use these to train our model
batch_size = 512
train_iter = torchtext.data.BucketIterator(dataset=train,
                                           batch_size=batch_size,
                                           sort_key=lambda x: x.text.__len__(),
                                           shuffle=True,
                                           sort_within_batch=True) 
val_iter = torchtext.data.BucketIterator(dataset=val,
                                         batch_size=batch_size,
                                         sort_key=lambda x: x.text.__len__(),
                                         train=False,
                                         sort_within_batch=True)
test_iter = torchtext.data.BucketIterator(dataset=test,
                                          batch_size=batch_size,
                                          sort_key=lambda x: x.text.__len__(),
                                          sort_within_batch=True)
# attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            
        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )
    
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions
# define our own model which is an lstm followed by two dense layers
class MyLSTM(nn.Module):
    def __init__(self, pretrained_lm, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(MyLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.weight.requires_grad = False
        self.lstm1 = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1, 
                            bidirectional=True)
        self.atten1 = Attention(hidden_dim*2, batch_first=True) # 2 is bidrectional
        self.lstm2 = nn.LSTM(input_size=hidden_dim*2,
                            hidden_size=hidden_dim,
                            num_layers=1, 
                            bidirectional=True)
        self.atten2 = Attention(hidden_dim*2, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim*lstm_layer*2, hidden_dim*lstm_layer*2),
                                 nn.BatchNorm1d(hidden_dim*lstm_layer*2),
                                 nn.ReLU()) 
        self.fc2 = nn.Linear(hidden_dim*lstm_layer*2, 1)

    
    def forward(self, x, x_len):
        x = self.embedding(x)
        x = self.dropout(x)
        
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out1, (h_n, c_n) = self.lstm1(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        x, _ = self.atten1(x, lengths) # skip connect

        out2, (h_n, c_n) = self.lstm2(out1)
        y, lengths = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True)
        y, _ = self.atten2(y, lengths)
        
        z = torch.cat([x, y], dim=1)
        z = self.fc1(self.dropout(z))
        z = self.fc2(self.dropout(z))
        return z
# code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
# weight for positive observation is 15x negative observation
pos_weight = 15

# computes validation score
def get_val_score(model, val_iter, loss_func):
    epoch_loss, epoch_acc = 0, 0
    epoch_tp, epoch_fp, epoch_fn = 0, 0, 0
    model.eval()
    
    with torch.no_grad():
        for batch in val_iter:
            question, x_len = batch.text
            x = question.cuda()
            y = batch.target.type(torch.Tensor).cuda()
            outs = model.forward(x, x_len).view(-1)
            weight = torch.FloatTensor(np.ones(len(y))).cuda()
            weight[(weight==1).nonzero()] = pos_weight
            loss_func.weight = weight
            loss = loss_func(outs, y)
            acc, tp, fp, fn = get_metrics(outs, y)
            
            epoch_loss += loss.item() / len(val_iter)
            epoch_acc += acc / len(val_iter)
            epoch_tp += tp
            epoch_fp += fp
            epoch_fn += fn
    
    epoch_precision = epoch_tp / (epoch_tp + epoch_fp)
    epoch_recall = epoch_tp / (epoch_tp + epoch_fn)
    epoch_f1 = 2 * epoch_precision * epoch_recall / (epoch_precision + epoch_recall)
    
    return epoch_loss, epoch_acc, epoch_f1

# does one epoch of training
def train_one_epoch(model, train_iter, val_iter, optimizer, loss_func, min_loss, scheduler, eval_every=1000):
    epoch_loss, epoch_acc = 0, 0
    epoch_tp, epoch_fp, epoch_fn = 0, 0, 0
    
    step = 0
    # iterate over batches in training set
    for batch in train_iter:
        model.train()
        # update learning rate
        if scheduler:
            scheduler.batch_step()
        step += 1
        model.zero_grad()
        # get question and label from batch
        question, x_len = batch.text
        x = question.cuda()
        y = batch.target.type(torch.Tensor).cuda()
        # compute forward pass
        outs = model.forward(x, x_len).view(-1)
        # put weights on positive examples
        weight = torch.FloatTensor(np.ones(len(y))).cuda()
        weight[(weight==1).nonzero()] = pos_weight
        loss_func.weight = weight
        # compute loss function and metrics
        loss = loss_func(outs, y)
        acc, tp, fp, fn = get_metrics(outs, y)
        # compute gradients wrt loss and do a step update
        loss.backward()
        optimizer.step()
        # consolidate metrics for batch
        epoch_loss += loss.item() / len(train_iter)
        epoch_acc += acc / len(train_iter)
        epoch_tp += tp
        epoch_fp += fp
        epoch_fn += fn
        # save model if val_f1 > max_f1
        if step % eval_every == 0:
            val_loss, val_acc, val_f1 = get_val_score(model, val_iter, loss_func)
            print('epoch', epoch, 'step', step, 'val_loss', val_loss, 'val_f1', val_f1, 'lr', scheduler.get_lr())
            if val_loss < min_loss:
                save(m=model, info={'step': step, 'epoch': epoch, 'val_loss': val_loss, 'val_f1': val_f1})
                min_loss = val_loss
                
    # compute f1 score over training set
    epoch_precision = epoch_tp / (epoch_tp + epoch_fp)
    epoch_recall = epoch_tp / (epoch_tp + epoch_fn)
    epoch_f1 = 2 * epoch_precision * epoch_recall / (epoch_precision + epoch_recall)
    
    return epoch_loss, epoch_acc, epoch_f1, min_loss

# save and load model functions
def save(m, info):
    torch.save(info, 'best_model.info')
    torch.save(m, 'best_model.m')
    
def load():
    m = torch.load('best_model.m')
    info = torch.load('best_model.info')
    return m, info
# initialize model, loss function, optimizer and scheduler
model = MyLSTM(text.vocab.vectors, hidden_dim=64, lstm_layer=2, dropout=0.3).cuda()
loss_func = nn.BCEWithLogitsLoss()
base_lr, max_lr = 0.001, 0.003
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=max_lr)
step_size = 300
scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
               step_size=step_size, mode='exp_range',
               gamma=0.99994)
# note that we always save our model at the minimum learning rate
eval_every = 2 * step_size
# training!
min_loss = float('inf')
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_f1, min_loss = train_one_epoch(model, train_iter, val_iter, optimizer, loss_func, min_loss, scheduler, eval_every)
    print('train_loss', train_loss, 'train_acc', train_acc, 'train_f1', train_f1)
    val_loss, val_acc, val_f1 = get_val_score(model, val_iter, loss_func)
    print('val_loss', val_loss, 'val_acc', val_acc, 'val_f1', val_f1)
    if val_loss < min_loss:
        save(m=model, info={'step': 'none', 'epoch': epoch, 'val_loss': val_loss, 'val_f1': val_f1})
        val_loss = min_loss
# load best model
model, m_info = load()
m_info
# flatten parameters for evaluation
model.lstm1.flatten_parameters()
model.lstm2.flatten_parameters()
model.eval()
model.zero_grad()
# save validation predictions 
val_preds = []
val_labels = []
for batch in val_iter:
    question, x_len = batch.text
    x = question.cuda()
    y = batch.target.type(torch.Tensor).cuda()
    outs = model.forward(x, x_len).view(-1)
    outs = sigmoid(outs.cpu().data.numpy()).tolist()
    y = y.cpu().data.numpy().tolist()
    val_preds += outs
    val_labels += y

val_preds = np.array(val_preds)
val_labels = np.array(val_labels)
# tune threshold of sigmoid to optimizer for f1 score
val_scores = []
thresholds = np.arange(0.1, 1.0, 0.001)
for threshold in thresholds:
    threshold = np.round(threshold, 3)
    f1 = f1_score(y_true=val_labels, y_pred=(val_preds > threshold).astype(int))
    val_scores.append(f1)

best_threshold = np.argmax(val_scores)
best_val_f1 = np.max(val_scores)
best_threshold = np.round(thresholds[np.argmax(val_scores)], 3)

plt.plot(thresholds, val_scores)
print('best_threshold', best_threshold, 'best_val_f1', best_val_f1)
# get test predictions
test_pred, test_id = [], []
for batch in test_iter:
    question, x_len = batch.text
    x = question.cuda()
    outs = model.forward(x, x_len).view(-1) 
    outs = sigmoid(outs.cpu().data.numpy()).tolist()
    test_pred += outs
    test_id += batch.qid.view(-1).data.numpy().tolist()
    
sub_df = pd.DataFrame()
sub_df['qid'] = [qid.vocab.itos[i] for i in test_id]
sub_df['prediction'] = (np.array(test_pred) >= best_threshold).astype(int)
sub_df.to_csv("submission.csv", index=False)