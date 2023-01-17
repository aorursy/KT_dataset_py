%%capture

import os
import time
import string

import numpy as np
import pandas as pd

# PyTorch 相关
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vectors
from torchtext import data

# 安装特定版本库
!pip install -v pytorch-ignite==0.4rc.0.post1
!pip install --upgrade scikit-learn

# Ignite 相关
from ignite.engine import Events, Engine
from ignite.metrics import Precision, Recall, Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.metrics import RocCurve, ROC_AUC
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from sklearn.metrics import RocCurveDisplay
# Input data files are available in the "/kaggle/input" directory.

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
EMBEDDING_FILE = '/kaggle/input/imdb-word2vec/word2vec.txt'
# EMBEDDING_FILE = '/kaggle/input/glove6b100dtxt/glove.6B.100d.txt'

def load_file(file_path, device, embedding_file):

    TEXT = data.Field(sequential=True, lower=True, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    
    datafields = [('clean_text', TEXT), ('label', LABEL)]
    # Step two construction our dataset.
    train, valid, test = data.TabularDataset.splits(path=file_path,
                                                    train="Train_clean.csv", validation="Valid_clean.csv",
                                                    test="Test_clean.csv", format="csv",
                                                    skip_header=True, fields=datafields)
    # because of input dir is read-only we must change the cache path.
    cache = ('/kaggle/working/.vector_cache')
    if not os.path.exists(cache):
        os.mkdir(cache)
    # using the pretrained word embedding.
    vector = Vectors(name=embedding_file, cache=cache)
    TEXT.build_vocab(train, vectors=vector, max_size=25000, unk_init=torch.Tensor.normal_)
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test), device=device, batch_size=64, 
                                                             sort_key=lambda x:len(x.clean_text), sort_within_batch=True)
    
    return TEXT, LABEL, train_iter, valid_iter, test_iter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

TEXT, LABEL, train_iter, valid_iter, test_iter = load_file('/kaggle/input/cleaned-imdb-data', 
                                                          device, EMBEDDING_FILE)
TEXT.vocab
class SentimentModelRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, text_lengths):

        embedded = self.embedding(text)
        
        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))
class SentimentModelLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        hidden = self.dropout(torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)).squeeze()    

        return self.fc(hidden)
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model_rnn = SentimentModelRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model_lstm = SentimentModelLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model_rnn', count_parameters(model_rnn))
print('model_lstm', count_parameters(model_lstm))
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model_rnn.embedding.weight.data.copy_(pretrained_embeddings)
model_lstm.embedding.weight.data.copy_(pretrained_embeddings)
optimizer_rnn = optim.Adam(model_rnn.parameters())
optimizer_lstm = optim.Adam(model_lstm.parameters())

loss_rnn = nn.BCEWithLogitsLoss()
loss_lstm = nn.BCEWithLogitsLoss()

model_rnn = model_rnn.to(device)
model_lstm = model_lstm.to(device)

loss_rnn = loss_rnn.to(device)
loss_lstm = loss_lstm.to(device)
def get_trainer_callback(model, optimizer, loss_fn):
    def train(engine, batch):
        model.train()
        
        text, text_lengths = batch.clean_text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = loss_fn(predictions, batch.label.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return train

def get_evaluator_callback(model):
    def evaluate(engine, batch):
        model.eval()
        with torch.no_grad():
            text, text_lengths = batch.clean_text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            y_pred =torch.sigmoid(predictions)
            y = batch.label.float()

            return y_pred, y

    return evaluate
def output_transform(output):
    y_pred = torch.round(output[0])
    y = output[1]
    return y_pred, y

def create_trainer_evaluator(model, optimizer, loss_fn, model_name):
    trainer = Engine(get_trainer_callback(model, optimizer, loss_fn))
    
    evaluator = Engine(get_evaluator_callback(model))

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer)
    
    saver = ModelCheckpoint('./', 'checkpoint', n_saved=2, require_empty=False)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, saver, {model_name: model})
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_iter)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f}"
              .format(engine.state.epoch, metrics['accuracy']))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_iter)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f}"
              .format(engine.state.epoch, metrics['accuracy']))
        
    accuracy = Accuracy(output_transform=output_transform)
    precision = Precision(output_transform=output_transform, average=False)
    recall = Recall(output_transform=output_transform, average=False)
    F1 = (precision * recall * 2 / (precision + recall)).mean()
    roc_curve = RocCurve()
    roc_auc = ROC_AUC()

    accuracy.attach(evaluator, "accuracy")
    precision.attach(evaluator, "precision")
    recall.attach(evaluator, "recall")
    F1.attach(evaluator, "F1")
    roc_curve.attach(evaluator, "roc_curve")
    roc_auc.attach(evaluator, "roc_auc")

    return trainer, evaluator
trainer_rnn, evaluator_rnn = create_trainer_evaluator(model_rnn, optimizer_rnn, loss_rnn, 'model_rnn')
trainer_rnn.run(train_iter, max_epochs=5)
torch.save(model_rnn.state_dict(), 'model_rnn')
trainer_lstm, evaluator_lstm = create_trainer_evaluator(model_lstm, optimizer_lstm, loss_lstm, 'model_lstm')
trainer_lstm.run(train_iter, max_epochs=5)
torch.save(model_lstm.state_dict(), 'model_lstm')
def print_plot_scores(model, evaluator, name):
    evaluator.run(test_iter)
    
    metrics = evaluator.state.metrics

    for key, value in metrics.items():
        if key != 'roc_curve':
            print('{}: {:.3f}'.format(key, value))

    fpr, tpr, _ = metrics['roc_curve']

    roc_auc = metrics['roc_auc']

    viz = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name
    )

    viz.plot(name=name)
model_rnn.load_state_dict(torch.load('model_rnn'))
print_plot_scores(model_rnn, evaluator_rnn, 'rnn')
model_lstm.load_state_dict(torch.load('model_lstm'))
print_plot_scores(model_lstm, evaluator_lstm, 'lstm')
def predict_sentiment(model, sentence):
    model.eval()
    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()
    tokenized = [tok for tok in tokenizer(sentence)]
    print(tokenized)
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length).to(device)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()
predict_sentiment(model_lstm, "i love it")
predict_sentiment(model_lstm, "This movie sucks")