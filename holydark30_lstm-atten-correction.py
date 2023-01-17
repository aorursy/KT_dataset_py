!pip install tokenizers
import pandas as pd

from math import ceil

import torch

import torch.nn as nn

import torch.optim as optim

from torch.nn.utils import clip_grad_norm_ 

import random

from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F


from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(

    "../input/correct/correction/pretrain_token/thai_2-vocab.json",

    "../input/correct/correction/pretrain_token/thai_2-merges.txt"

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
src_len = torch.tensor([tokens.shape[0] for tokens in src_tokenize]).to(device)

val_src_len = torch.tensor([tokens.shape[0] for tokens in val_src_tokenize]).to(device)
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

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):

        super().__init__()

        

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, batch_first=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, src, src_len):

        #src = [batch size, src_len]

        #src_len = [batch size]

        

        src_max_len = src.shape[1]

        embedded = self.dropout(self.embedding(src))

        

        #embedded = [batch size, src len, emb dim]



        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)



        packed_outputs, hidden = self.rnn(packed_embedded)

                                 

        #packed_outputs is a packed sequence containing all hidden states

        #hidden is now from the final non-padded element in the batch



        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=src_max_len) 



        #outputs is now a non-packed sequence, all hidden states obtained

        #  when the input is a pad token are all zeros

            

        #outputs = [batch size, src len, hid dim * num directions]

        #hidden = [n layers * num directions, batch size, hid dim]

        

        

        #initial decoder hidden is final hidden state of the forwards and backwards 

        #  encoder RNNs fed through a linear layer



        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        

        #outputs = [batch size, src len, enc hid dim * 2]

        #hidden = [batch size, dec hid dim]

        

        return outputs, hidden
class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):

        super().__init__()

        

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)

        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

        

    def forward(self, hidden, encoder_outputs, mask):

        

        #hidden = [batch size, dec hid dim]

        #encoder_outputs = [batch size, src_len, enc hid dim * 2]

        

        batch_size = encoder_outputs.shape[0]

        src_len = encoder_outputs.shape[1]

        

        #repeat decoder hidden state src_len times

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

  

        

        #hidden = [batch size, src len, dec hid dim]

        #encoder_outputs = [batch size, src len, enc hid dim * 2]



        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 

        

        #energy = [batch size, src len, dec hid dim]



        attention = self.v(energy).squeeze(2)

        

        #attention = [batch size, src len]

        

        attention = attention.masked_fill(mask == 0, -1e10)

        

        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):

        super().__init__()



        self.output_dim = output_dim

        self.attention = attention

        

        self.embedding = nn.Embedding(output_dim, emb_dim)

        

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)

        

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, input, hidden, encoder_outputs, mask):

            

        input = input.unsqueeze(1)

        

        #input = [batch size, 1]



        embedded = self.dropout(self.embedding(input))



        #embedded = [batch size, 1,emb dim]

        

        a = self.attention(hidden, encoder_outputs, mask)

                

        #a = [batch size, src len]

        

        a = a.unsqueeze(1)

        

        #a = [batch size, 1, src len]



        weighted = torch.bmm(a, encoder_outputs)

        

        #weighted = [batch size, 1, enc hid dim * 2]



        rnn_input = torch.cat((embedded, weighted), dim = 2)

        

        #rnn_input = [batch size, 1, (enc hid dim * 2) + emb dim]

            

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        

        #output = [batch size, seq len, dec hid dim * n directions]

        #hidden = [n layers * n directions, batch size, dec hid dim]

        

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:

        #output = [batch size, 1, dec hid dim]

        #hidden = [1, batch size, dec hid dim]

        

        embedded = embedded.squeeze(1)

        output = output.squeeze(1)

        weighted = weighted.squeeze(1)

        

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        

        #prediction = [batch size, output dim]

        

        return prediction, hidden.squeeze(0), a.squeeze(1)
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, src_pad_idx, device):

        super().__init__()

        

        self.encoder = encoder

        self.decoder = decoder

        self.src_pad_idx = src_pad_idx

        self.device = device

        

    def create_mask(self, src):

        mask = (src != self.src_pad_idx)

        return mask

        

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):

        

        #src = [batch size, src len]

        #src_len = [batch size]

        #trg = [batch size, trg len]

        #teacher_forcing_ratio is probability to use teacher forcing

        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

                    

        batch_size = src.shape[0]

        trg_len = trg.shape[1]

        trg_vocab_size = self.decoder.output_dim

        

        #tensor to store decoder outputs

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        

        #encoder_outputs is all hidden states of the input sequence, back and forwards

        #hidden is the final forward and backward hidden states, passed through a linear layer

        encoder_outputs, hidden = self.encoder(src, src_len)

                        

        input = trg[:,0]

        

        mask = self.create_mask(src)



        #mask = [batch size, src len]

                

        for t in range(1, trg_len):

            

            #insert input token embedding, previous hidden state, all encoder hidden states 

            #  and mask

            #receive output tensor (predictions) and new hidden state

            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            

            outputs[:,t] = output

            

            teacher_force = random.random() < teacher_forcing_ratio

            

            top1 = output.argmax(1) 



            input = trg[:,t] if teacher_force else top1

            

        return outputs
input_size = 20000

enc_emb = 256

enc_hidden = 256



dec_emb = 256

dec_hidden = 256

num_layers = 1



attention = Attention(enc_hidden, dec_hidden)

encoder = Encoder(input_size, enc_emb, enc_hidden, dec_hidden, 0.3) 

decoder = Decoder(input_size, dec_emb, enc_hidden, dec_hidden, 0.3, attention)

model = Seq2Seq(encoder, decoder, pad_tok, device).to(device)
import time

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
criterion = nn.CrossEntropyLoss(ignore_index = pad_tok)

optimizer = optim.Adam(model.parameters())



epochs = 7

clip = 1

hist = []



for epoch in range(epochs):



    epoch_loss = 0

    start_time = time.time()

    model.train()



    for n in range(n_batch):

        train_src = src_padded[n*batch_size:(n+1)*batch_size]

        train_trg = trg_padded[n*batch_size:(n+1)*batch_size]

        n_src_len = src_len[n*batch_size:(n+1)*batch_size]



        optimizer.zero_grad()



        preds = model(train_src, n_src_len, train_trg)



        #trg = [batch size,trg len]

        #output = [batch size,trg len, output dim]        



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

            n_src_len = val_src_len[n*batch_size:(n+1)*batch_size]



            preds = model(val_src, n_src_len, val_trg, teacher_forcing_ratio = 0.0)



            output_dim = preds.shape[-1]



            preds = preds[1:].view(-1, output_dim)

            val_trg = val_trg[1:].view(-1)



            loss = criterion(preds, val_trg)

            val_loss += loss.item()



    print(f'Epoch: {epoch+1} | Time: {epoch_mins}m {epoch_secs}s')

    print(f'Train Loss:{epoch_loss / n_batch}')

    print(f'Val Loss:{val_loss / val_batch}')

    hist.append([epoch_loss / n_batch, val_loss / val_batch])



checkpoint = {'model': Seq2Seq(encoder, decoder, pad_tok, device),

              'state_dict': model.state_dict(),

              'optimizer' : optimizer.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')
