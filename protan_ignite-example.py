from torchtext import data

from torchtext import datasets

from torchtext.vocab import Vectors

import torchtext
import torch

import torch.nn as nn

import torch.nn.functional as F



SEED = 1234

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)
import pathlib

import numpy as np

import sklearn.metrics as skm

import pprint



import csv

import sys

csv.field_size_limit(sys.maxsize)  # needed for torchtext
!pip install pytorch-ignite
from ignite.engine import Engine, Events

from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall

from ignite.handlers import ModelCheckpoint, EarlyStopping

from ignite.contrib.handlers import ProgressBar
input_path = '../input/dlinnlp-spring-2019-clf'

vectors_path = '../input/glove840b300dtxt/glove.840B.300d.txt'

cache_path = '../input/glove840b300dtxt'
%matplotlib inline

import pandas as pd
df = pd.read_csv(f'{input_path}/train.csv')
df.head()
df.label.value_counts()
MAX_TITLE_LEN = 40

MAX_BODY_LEN = 2000
df.title.apply(lambda x: len(str(x).split())).clip_upper(MAX_TITLE_LEN).plot(kind='hist', bins=MAX_TITLE_LEN)
df.text.apply(lambda x: len(str(x).split())).clip_upper(MAX_BODY_LEN).plot(kind='hist', bins=50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index2label = ['news', 'clickbait', 'other']

label2index = {l: i for i, l in enumerate(index2label)}



title_field = torchtext.data.Field(lower=True, include_lengths=False, fix_length=MAX_TITLE_LEN, batch_first=True)

body_field = torchtext.data.Field(lower=True, include_lengths=False, fix_length=MAX_BODY_LEN, batch_first=True)

label_field = torchtext.data.Field(sequential=False, is_target=True, use_vocab=False,

                                   preprocessing=lambda x: label2index[x])



train_dataset = torchtext.data.TabularDataset(f'{input_path}/train.csv',

                                              format='csv',

                                              fields={'title': ('title', title_field),

                                                      'text': ('body', body_field),

                                                      'label': ('label', label_field)})



val_dataset = torchtext.data.TabularDataset(f'{input_path}/valid.csv',

                                            format='csv',

                                            fields={'title': ('title', title_field),

                                                    'text': ('body', body_field),

                                                    'label': ('label', label_field)})



test_dataset = torchtext.data.TabularDataset(f'{input_path}/test.csv',

                                            format='csv',

                                            fields={'title': ('title', title_field),

                                                    'text': ('body', body_field)})



body_field.build_vocab(train_dataset, min_freq=2)

label_field.build_vocab(train_dataset)

vocab = body_field.vocab

title_field.vocab = vocab



print('Vocab size: ', len(vocab))

print(train_dataset[0].title)

print(train_dataset[0].body[:15])

print(train_dataset[0].label)

vocab.load_vectors(Vectors(vectors_path))



print ('Attributes of title_field.vocab : ', [attr for attr in dir(vocab) if '_' not in attr])

print ('First 5 values title_field.vocab.itos : ', vocab.itos[0:5]) 

print ('First 5 key, value pairs of title_field.vocab.stoi : ', {key:value for key,value in list(vocab.stoi.items())[0:5]}) 

print ('Shape of title_field.vocab.vectors.shape : ', vocab.vectors.shape)
train_loader = data.Iterator(train_dataset, batch_size=64, device='cuda', shuffle=True, sort=False)

val_loader = data.Iterator(val_dataset, batch_size=64, device='cuda', shuffle=False, sort=False)

test_loader = data.Iterator(test_dataset, batch_size=64, device='cuda', shuffle=False, sort=False)
batch = next(iter(train_loader))

print(batch)
next(iter(test_loader))
batch = next(iter(train_loader))

print('batch.label[0] : ', batch.label[0])

print('batch.title[0] : ', batch.title[0][batch.title[0] != 1])

print('batch.body[0] : ', batch.body[0][batch.body[0] != 1])



lengths = []

for i, batch in enumerate(train_loader):

    x = batch.title

    lengths.append(x.shape[1])

    if i == 10:

        break



print ('Lengths of first 10 batches : ', lengths)
class TextModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters,

                 num_classes, d_prob, mode, hidden_dim, lstm_units,

                 emb_vectors=None):

        super(TextModel, self).__init__()

        self.vocab_size = vocab_size

        self.embedding_dim = embedding_dim

        self.kernel_sizes = kernel_sizes

        self.num_filters = num_filters

        self.num_classes = num_classes

        self.d_prob = d_prob

        self.mode = mode

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)

        self.load_embeddings(emb_vectors)

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,

                                             out_channels=num_filters,

                                             kernel_size=k, stride=1) for k in kernel_sizes])

        self.conv2 = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,

                                             out_channels=num_filters,

                                             kernel_size=k, stride=1) for k in kernel_sizes])

        self.conv_body = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,

                                             out_channels=num_filters,

                                             kernel_size=k, stride=1) for k in kernel_sizes])

        self.lstm1 = nn.LSTM(embedding_dim, lstm_units, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.lstm_body = nn.LSTM(embedding_dim, lstm_units, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(d_prob)

        self.fc = nn.Linear(len(kernel_sizes) * num_filters, hidden_dim)

        self.fc_body = nn.Linear(len(kernel_sizes) * num_filters, hidden_dim)

        self.fc_total = nn.Linear(hidden_dim * 2 + lstm_units * 6, hidden_dim)

        self.fc_final = nn.Linear(hidden_dim, num_classes)





    def forward(self, x):

        title, body = x

        batch_size, sequence_length = title.shape



        title_emb = self.embedding(title)

        title = [F.relu(conv(title_emb.transpose(1, 2))) for conv in self.conv]

        title = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in title]

        title = [F.relu(conv(title_emb.transpose(1, 2))) for conv in self.conv2]

        title = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in title]

        title = torch.cat(title, dim=1)

        title = self.fc(self.dropout(title))

        

        h_lstm1, _ = self.lstm1(title_emb)

        h_lstm2, _ = self.lstm2(h_lstm1)

        

        # average pooling

        avg_pool2 = torch.mean(h_lstm2, 1)

        # global max pooling

        max_pool2, _ = torch.max(h_lstm2, 1)



        batch_size, sequence_length = body.shape

        body_emb = self.embedding(body)

        body = [F.relu(conv(body_emb.transpose(1, 2))) for conv in self.conv_body]

        body = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in body]

        body = torch.cat(body, dim=1)

        body = self.fc_body(self.dropout(body))

        

        h_lstm_body, _ = self.lstm_body(body_emb)

        max_pool_body, _ = torch.max(h_lstm_body, 1)



        x = torch.cat([title, body, avg_pool2, max_pool2, max_pool_body], dim=1)

        x = F.relu(self.fc_total(self.dropout(x)))

        x = self.fc_final(x)

        

        return x

    

    def load_embeddings(self, emb_vectors):

        if 'static' in self.mode:

            self.embedding.weight.data.copy_(emb_vectors)

            if 'non' not in self.mode:

                self.embedding.weight.data.requires_grad = False

                print('Loaded pretrained embeddings, weights are not trainable.')

            else:

                self.embedding.weight.data.requires_grad = True

                print('Loaded pretrained embeddings, weights are trainable.')

        elif self.mode == 'rand':

            print('Randomly initialized embeddings are used.')

        else:

            raise ValueError('Unexpected value of mode. Please choose from static, nonstatic, rand.')

            
vocab_size, embedding_dim = vocab.vectors.shape



model = TextModel(vocab_size=vocab_size,

                  embedding_dim=embedding_dim,

                  kernel_sizes=[3, 4, 5],

                  num_filters=64,

                  num_classes=3, 

                  d_prob=0.6,

                  mode='nonstatic',

                  hidden_dim=256,

                  emb_vectors=vocab.vectors,

                  lstm_units=64)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
def process_function(engine, batch):

    model.train()

    optimizer.zero_grad()

    x, y = (batch.title, batch.body), batch.label

    y_pred = model(x)

    loss = criterion(y_pred, y)

    loss.backward()

    optimizer.step()

    return loss.item()
def eval_function(engine, batch):

    model.eval()

    with torch.no_grad():

        x, y = (batch.title, batch.body), batch.label

        y_pred = model(x)

        return y_pred, y
trainer = Engine(process_function)

train_evaluator = Engine(eval_function)

validation_evaluator = Engine(eval_function)
RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
def thresholded_output_transform(output):

    y_pred, y = output

    y_pred = torch.round(y_pred)

    return y_pred, y
Accuracy(output_transform=thresholded_output_transform).attach(train_evaluator, 'accuracy')

Loss(criterion).attach(train_evaluator, 'ce')



precision = Precision(average=False)

recall = Recall(average=False)

F1 = (precision * recall * 2 / (precision + recall)).mean()



precision.attach(train_evaluator, 'precision')

recall.attach(train_evaluator, 'recall')

F1.attach(train_evaluator, 'F1')
Accuracy(output_transform=thresholded_output_transform).attach(validation_evaluator, 'accuracy')

Loss(criterion).attach(validation_evaluator, 'ce')



precision = Precision(average=False)

recall = Recall(average=False)

F1 = (precision * recall * 2 / (precision + recall)).mean()



precision.attach(validation_evaluator, 'precision')

recall.attach(validation_evaluator, 'recall')

F1.attach(validation_evaluator, 'F1')
pbar = ProgressBar(persist=True, bar_format="")

pbar.attach(trainer, ['loss'])
def score_function(engine):

    val_loss = engine.state.metrics['F1']

    return val_loss



handler = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)

validation_evaluator.add_event_handler(Events.COMPLETED, handler)
@trainer.on(Events.EPOCH_COMPLETED)

def log_training_results(engine):

    train_evaluator.run(train_loader)

    metrics = train_evaluator.state.metrics

    pbar.log_message(

        "Training Results - Epoch: {} \nMetrics\n{}"

        .format(engine.state.epoch, pprint.pformat(metrics)))

    

def log_validation_results(engine):

    validation_evaluator.run(val_loader)

    metrics = validation_evaluator.state.metrics

    metrics = validation_evaluator.state.metrics

    pbar.log_message(

        "Validation Results - Epoch: {} \nMetrics\n{}"

        .format(engine.state.epoch, pprint.pformat(metrics)))

    pbar.n = pbar.last_print_n = 0



trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
checkpointer = ModelCheckpoint('checkpoint', 'textcnn', save_interval=1, n_saved=2, create_dir=True, save_as_state_dict=True)

best_model_save = ModelCheckpoint(

    'best_model', 'textcnn', n_saved=1,

    create_dir=True, save_as_state_dict=True,

    score_function=score_function)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'textcnn': model})

validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_model_save, {'textcnn': model})
trainer.run(train_loader, max_epochs=20)
model_path = next(pathlib.Path('best_model').rglob('*'))

model_path
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)
predictions = []

labels = []



# change model mode to 'evaluation'

# disable dropout and use learned batch norm statistics

model.eval()
predictions = []

labels = []



# change model mode to 'evaluation'

# disable dropout and use learned batch norm statistics

model.eval()



with torch.no_grad():

    for batch in val_loader:

        x, label = batch

#         logits = model(title)

        logits = model(x)



        y_pred = torch.max(logits, dim=1)[1]

        # move from GPU to CPU and convert to numpy array

        y_pred_numpy = y_pred.cpu().numpy()



        predictions = np.concatenate([predictions, y_pred_numpy])

        labels = np.concatenate([labels, label.cpu().numpy()])

skm.f1_score(labels, predictions, average='micro')
skm.f1_score(labels, predictions, average='macro')
# Do not shuffle test set! You need id to label mapping

test_loader = torchtext.data.Iterator(test_dataset, batch_size=128, device='cuda', shuffle=False)



predictions = []



model.eval()



with torch.no_grad():

    for batch in test_loader:

        x, label = batch

#         logits = model(title)

        logits = model(x)



        y_pred = torch.max(logits, dim=1)[1]

        # move from GPU to CPU and convert to numpy array

        y_pred_numpy = y_pred.cpu().numpy()



        predictions = np.concatenate([predictions, y_pred_numpy])
predictions_str = [index2label[int(p)] for p in predictions]



# test.csv index in a contiguous integers from 0 to len(test_set)

# to this should work fine

submission = pd.DataFrame({'id': list(range(len(predictions_str))), 'label': predictions_str})

submission.head()
submission.to_csv('submission_ignite.csv', index=False)