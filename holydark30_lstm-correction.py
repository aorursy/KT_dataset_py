import pandas as pd

from math import ceil

import torch

import torch.nn as nn

!pip install tokenizers

import torch.optim as optim

from torch.nn.utils import clip_grad_norm_ 

import random

from torch.nn.utils.rnn import pad_sequence


from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(

    "../input/correct/correction/pretrain_token/thai_1-vocab.json",

    "../input/correct/correction/pretrain_token/thai_1-merges.txt"

)

train_df = pd.read_csv('../input/correct/correction/sentence_correction_data/train.csv')

train_df.head()

val_df = pd.read_csv('../input/correct/correction/sentence_correction_data/val.csv')

val_df.head()

src_txt = list(train_df['src'].values)

trg_txt = list(train_df['tgt'].values)



val_src_txt = list(val_df['src'].values)

val_trg_txt = list(val_df['tgt'].values)
sos_tok = 0

eos_toke = 2

pad_tok = 1



src_tokenize = [torch.tensor([sos_tok] + tokenizer.encode(txt).ids + [eos_toke]) for txt in src_txt]

trg_tokenize = [torch.tensor([sos_tok] + tokenizer.encode(txt).ids + [eos_toke]) for txt in trg_txt]



val_src_tokenize = [torch.tensor([sos_tok] + tokenizer.encode(txt).ids + [eos_toke]) for txt in val_src_txt]

val_trg_tokenize = [torch.tensor([sos_tok] + tokenizer.encode(txt).ids + [eos_toke]) for txt in val_trg_txt]

device = 'cuda' if torch.cuda.is_available() else 'cpu'



src_padded = pad_sequence(src_tokenize, batch_first=True, padding_value=pad_tok).to(device) 

trg_padded = pad_sequence(trg_tokenize, batch_first=True, padding_value=pad_tok).to(device) 



val_src_padded = pad_sequence(val_src_tokenize, batch_first=True, padding_value=pad_tok).to(device) 

val_trg_padded = pad_sequence(val_trg_tokenize, batch_first=True, padding_value=pad_tok).to(device) 



max_src_seq = src_padded.shape[-1]

max_trg_seq = trg_padded.shape[-1]



batch_size = 32

n_batch = ceil(src_padded.shape[0] // batch_size)

val_batch = ceil(val_src_padded.shape[0] // batch_size)
class Encoder(nn.Module):

    def __init__(self, input_size, emb, hidden, num_layers):

        super(Encoder, self).__init__()

        self.hidden_size = hidden

        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, emb)

        self.lstm = nn.LSTM(emb, hidden, num_layers, batch_first=True, dropout=0.3)



    def forward(self, x, hidden=None):

        embedded = self.embedding(x)

        output, hidden = self.lstm(embedded, hidden)

        return hidden



# Received one token at a time - non autoregresive

class Decoder(nn.Module):

    def __init__(self, output_size, emb, hidden, num_layers):

      super(Decoder, self).__init__()

      self.hidden_size = hidden



      self.embedding = nn.Embedding(output_size, emb)

      self.lstm = nn.LSTM(emb, hidden, num_layers, batch_first=True, dropout=0.3)

      self.out = nn.Linear(hidden, output_size)





    def forward(self, x, hidden):

      # x = (batch_size, 1)

      embedded = self.embedding(x)

      # (batch_size, emb)

      output, hidden = self.lstm(embedded, hidden)

      output = self.out(output)

      # output -> (batch_size, 1, hid)

      return output, hidden







class Seq2Seq(nn.Module):

  def __init__(self, encoder, decoder):



    super(Seq2Seq, self).__init__()

    self.encoder = encoder

    self.decoder = decoder





  def forward(self, src, trg, teacher_ratio = 0.5):



    hidden = self.encoder(src)

    bs = trg.shape[0]

    max_seq = trg.shape[-1]

    outputs = torch.zeros(bs, max_seq, input_size).to(device)



    nxt_trg = trg[:,0].unsqueeze(1)





    for i in range(1, max_seq):

      teacher_force = random.random() < teacher_ratio

      

      nxt_trg, hidden = self.decoder(nxt_trg, hidden)



      outputs[:,i] = nxt_trg.squeeze(1)



      max_pred = nxt_trg.argmax(-1)



      nxt_trg =  trg[:,i].unsqueeze(1) if teacher_force else max_pred

    

    return outputs

input_size = 10000

enc_emb = 256

enc_hidden = 256



dec_emb = 256

dec_hidden = 256

num_layers = 1



encoder = Encoder(input_size, enc_emb, enc_hidden, num_layers)

decoder = Decoder(input_size, dec_emb, dec_hidden, num_layers)

model = Seq2Seq(encoder, decoder).to(device)
import time

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
criterion = nn.CrossEntropyLoss(ignore_index = pad_tok)

optimizer = optim.Adam(model.parameters())



epochs = 6

clip = 1

hist = []



for epoch in range(epochs):



  epoch_loss = 0

  start_time = time.time()

  model.train()



  for n in range(n_batch):

    train_src = src_padded[n*batch_size:(n+1)*batch_size]

    train_trg = trg_padded[n*batch_size:(n+1)*batch_size]



    optimizer.zero_grad()



    preds = model(train_src, train_trg)



    output_dim = preds.shape[-1]



    preds = preds[1:].view(-1, output_dim)

    train_trg = train_trg[1:].view(-1)



    loss = criterion(preds, train_trg)

    loss.backward()

    

    clip_grad_norm_(model.parameters(), clip)

 

    optimizer.step()



    epoch_loss += loss.item()

  end_time = time.time()

    

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)



  model.eval()

  val_loss = 0

  with torch.no_grad():

    for n in range(val_batch):

      val_src = val_src_padded[n*batch_size:(n+1)*batch_size]

      val_trg = val_trg_padded[n*batch_size:(n+1)*batch_size]

      preds = model(val_src, val_trg, teacher_ratio = 0.0)



      output_dim = preds.shape[-1]



      preds = preds[1:].view(-1, output_dim)

      val_trg = val_trg[1:].view(-1)



      loss = criterion(preds, val_trg)

      val_loss += loss.item()



      



  print(f'Epoch: {epoch+1} | Time: {epoch_mins}m {epoch_secs}s')

  print(f'Train Loss:{epoch_loss / n_batch}')

  print(f'Val Loss:{val_loss / val_batch}')

  hist.append([epoch_loss / n_batch, val_loss / val_batch])



checkpoint = {'model': Seq2Seq(encoder, decoder),

              'state_dict': model.state_dict(),

              'optimizer' : optimizer.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')
