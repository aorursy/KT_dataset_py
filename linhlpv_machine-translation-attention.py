# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# !git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

    

!pip install pyvi
import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F





from torchtext.datasets import TranslationDataset, Multi30k

from torchtext.data import Field, BucketIterator, TabularDataset, Iterator



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



import spacy



import random

import math

import time

import pandas as pd



import  codecs

import numpy as np

from pyvi import ViTokenizer, ViPosTagger, ViUtils



# from apex import amp

from tqdm import tqdm_notebook as tqdm
SEED = 1234



random.seed(SEED)

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
def tokenize_in(text):

    """

    Tokenizes input text

    """

    text = ViTokenizer.tokenize(text)

    text = text.replace('_', ' ')

    return [word for word in text.split()]



def tokenize_out(text):

    text = ViTokenizer.tokenize(text)

    text = text.replace('_', ' ')



    return [word for word in text.split()]

SRC = Field(tokenize=tokenize_in, 

            init_token='<sos>', 

            eos_token='<eos>', 

            include_lengths = True,

            lower=True)



TRG = Field(tokenize = tokenize_out, 

            init_token='<sos>', 

            eos_token='<eos>',

            lower=True)
# def read_txt():

#     f = codecs.open('./vietnamese_tone_prediction/train.txt', encoding='utf-8')

#     data = []

#     null_data = []

#     for i, line in enumerate(f):

#         data.append(line[:len(line)-1])

#     return data, null_data
# data_src = [ViUtils.remove_accents(sent).decode('utf-8') for sent in data_trg]
data_fields = [('vi_no_accents', SRC), ('vi', TRG)]

train_data, test_data = TabularDataset.splits(path='../input/tone-prediction/', train='train_200k_r.csv', validation='test_200k_r.csv', format='csv',fields=data_fields)
SRC.build_vocab(train_data, min_freq=2)

TRG.build_vocab(train_data, min_freq=2)
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")

print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64



train_iterator, test_iterator = BucketIterator.splits(

    (train_data, test_data), 

     batch_size = BATCH_SIZE,

     sort_within_batch = True,

     sort_key = lambda x : len(x.vi_no_accents),

     device = device)
class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):

        super().__init__()

        

        self.embedding = nn.Embedding(input_dim, emb_dim)

        

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, src, src_len):

        

        #src = [src sent len, batch size]

        #src_len = [src sent len]

        

        embedded = self.dropout(self.embedding(src))

        

        #embedded = [src sent len, batch size, emb dim]

                

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

                

        packed_outputs, hidden = self.rnn(packed_embedded)

                                 

        #packed_outputs is a packed sequence containing all hidden states

        #hidden is now from the final non-padded element in the batch

            

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 

            

        #outputs is now a non-packed sequence, all hidden states obtained

        #  when the input is a pad token are all zeros

            

        #outputs = [sent len, batch size, hid dim * num directions]

        #hidden = [n layers * num directions, batch size, hid dim]

        

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]

        #outputs are always from the last layer

        

        #hidden [-2, :, : ] is the last of the forwards RNN 

        #hidden [-1, :, : ] is the last of the backwards RNN

        

        #initial decoder hidden is final hidden state of the forwards and backwards 

        #  encoder RNNs fed through a linear layer

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        

        #outputs = [sent len, batch size, enc hid dim * 2]

        #hidden = [batch size, dec hid dim]

        

        return outputs, hidden
class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):

        super().__init__()

        

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)

        self.v = nn.Parameter(torch.rand(dec_hid_dim))

        

    def forward(self, hidden, encoder_outputs, mask):

        

        #hidden = [batch size, dec hid dim]

        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        #mask = [batch size, src sent len]

        

        batch_size = encoder_outputs.shape[1]

        src_len = encoder_outputs.shape[0]

        

        #repeat encoder hidden state src_len times

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        

        #hidden = [batch size, src sent len, dec hid dim]

        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 

        

        #energy = [batch size, src sent len, dec hid dim]

                

        energy = energy.permute(0, 2, 1)

        

        #energy = [batch size, dec hid dim, src sent len]

        

        #v = [dec hid dim]

        

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        

        #v = [batch size, 1, dec hid dim]

            

        attention = torch.bmm(v, energy).squeeze(1)

        

        #attention = [batch size, src sent len]

        

        attention = attention.masked_fill(mask == 0, -1e10)

        

        return F.softmax(attention, dim = 1)
class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):

        super().__init__()



        self.output_dim = output_dim

        self.attention = attention

        

        self.embedding = nn.Embedding(output_dim, emb_dim)

        

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        

        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, input, hidden, encoder_outputs, mask):

             

        #input = [batch size]

        #hidden = [batch size, dec hid dim]

        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        #mask = [batch size, src sent len]

        

        input = input.unsqueeze(0)

        

        #input = [1, batch size]

        

        embedded = self.dropout(self.embedding(input))

        

        #embedded = [1, batch size, emb dim]

        

        a = self.attention(hidden, encoder_outputs, mask)

                

        #a = [batch size, src sent len]

        

        a = a.unsqueeze(1)

        

        #a = [batch size, 1, src sent len]

        

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        

        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        

        weighted = torch.bmm(a, encoder_outputs)

        

        #weighted = [batch size, 1, enc hid dim * 2]

        

        weighted = weighted.permute(1, 0, 2)

        

        #weighted = [1, batch size, enc hid dim * 2]

        

        rnn_input = torch.cat((embedded, weighted), dim = 2)

        

        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

            

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        

        #output = [sent len, batch size, dec hid dim * n directions]

        #hidden = [n layers * n directions, batch size, dec hid dim]

        

        #sent len, n layers and n directions will always be 1 in this decoder, therefore:

        #output = [1, batch size, dec hid dim]

        #hidden = [1, batch size, dec hid dim]

        #this also means that output == hidden

        assert (output == hidden).all()

        

        embedded = embedded.squeeze(0)

        output = output.squeeze(0)

        weighted = weighted.squeeze(0)

        

        output = self.out(torch.cat((output, weighted, embedded), dim = 1))

        

        #output = [bsz, output dim]

        

        return output, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):

        super().__init__()

        

        self.encoder = encoder

        self.decoder = decoder

        self.pad_idx = pad_idx

        self.sos_idx = sos_idx

        self.eos_idx = eos_idx

        self.device = device

        

    def create_mask(self, src):

        mask = (src != self.pad_idx).permute(1, 0)

        return mask

        

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):

        

        #src = [src sent len, batch size]

        #src_len = [batch size]

        #trg = [trg sent len, batch size]

        #teacher_forcing_ratio is probability to use teacher forcing

        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        

        if trg is None:

            assert teacher_forcing_ratio == 0, "Must be zero during inference"

            inference = True

            trg = torch.zeros((100, src.shape[1])).long().fill_(self.sos_idx).to(src.device)

        else:

            inference = False

            

        batch_size = src.shape[1]

        max_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim

        

        #tensor to store decoder outputs

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        

        #tensor to store attention

        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)

        

        #encoder_outputs is all hidden states of the input sequence, back and forwards

        #hidden is the final forward and backward hidden states, passed through a linear layer

        encoder_outputs, hidden = self.encoder(src, src_len)

                

        #first input to the decoder is the <sos> tokens

        input = trg[0,:]

        

        mask = self.create_mask(src)

                

        #mask = [batch size, src sent len]

                

        for t in range(1, max_len):

            

            #insert input token embedding, previous hidden state, all encoder hidden states 

            # and mask

            #receive output tensor (predictions), new hidden state and attention tensor

            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)

            

            #place predictions in a tensor holding predictions for each token

            outputs[t] = output

            

            #place attentions in a tensor holding attention value for each input token

            attentions[t] = attention

            

            #decide if we are going to use teacher forcing or not

            teacher_force = random.random() < teacher_forcing_ratio

            

            #get the highest predicted token from our predictions

            top1 = output.argmax(1) 

            

            #if teacher forcing, use actual next token as next input

            #if not, use predicted token

            input = trg[t] if teacher_force else top1

            

            #if doing inference and next token/prediction is an eos token then stop

            if inference and input.item() == self.eos_idx:

                return outputs[:t], attentions[:t]

            

        return outputs, attentions
INPUT_DIM = len(SRC.vocab)

OUTPUT_DIM = len(TRG.vocab)

ENC_EMB_DIM = 100

DEC_EMB_DIM = 100

ENC_HID_DIM = 200

DEC_HID_DIM = 200

ENC_DROPOUT = 0.5

DEC_DROPOUT = 0.5

PAD_IDX = SRC.vocab.stoi['<pad>']

SOS_IDX = TRG.vocab.stoi['<sos>']

EOS_IDX = TRG.vocab.stoi['<eos>']



attn = Attention(ENC_HID_DIM, DEC_HID_DIM)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)



model = Seq2Seq(enc, dec, PAD_IDX, SOS_IDX, EOS_IDX, device).to(device)
def init_weights(m):

    for name, param in m.named_parameters():

        if 'weight' in name:

            nn.init.normal_(param.data, mean=0, std=0.01)

        else:

            nn.init.constant_(param.data, 0)

            

model.apply(init_weights)


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters())
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
def train(model, iterator, optimizer, criterion, clip):

    

    model.train()

    

    epoch_loss = 0

    

    

    for i, batch in enumerate(iterator):

#         if i % 100 == 0:

#             print('iterator ', i)

        

        src, src_len = batch.vi_no_accents

        trg = batch.vi

        

        optimizer.zero_grad()

        

        output, attetion = model(src, src_len, trg)

        

        #trg = [trg sent len, batch size]

        #output = [trg sent len, batch size, output dim]

        

        output = output[1:].view(-1, output.shape[-1])

        trg = trg[1:].view(-1)

        

        #trg = [(trg sent len - 1) * batch size]

        #output = [(trg sent len - 1) * batch size, output dim]

        

        loss = criterion(output, trg)

#         with amp.scale_loss(loss, optimizer) as scaled_loss:

#             scaled_loss.backward()

        loss.backward()

        

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        

        optimizer.step()

        

        epoch_loss += loss.item()

        

    return epoch_loss / len(iterator)
def evaluate(model, iterator, criterion):

    

    model.eval()

    

    epoch_loss = 0

    

    with torch.no_grad():

    

        for i, batch in enumerate(iterator):



            src, src_len = batch.vi_no_accents

            trg = batch.vi



            output, attention = model(src, src_len, trg, 0) #turn off teacher forcing



            #trg = [trg sent len, batch size]

            #output = [trg sent len, batch size, output dim]



            output = output[1:].view(-1, output.shape[-1])

            trg = trg[1:].view(-1)



            #trg = [(trg sent len - 1) * batch size]

            #output = [(trg sent len - 1) * batch size, output dim]



            loss = criterion(output, trg)



            epoch_loss += loss.item()

        

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
N_EPOCHS = 20

CLIP = 1



best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):

    print(epoch)

    

    start_time = time.time()

    

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)

    valid_loss = evaluate(model, test_iterator, criterion)

    

    end_time = time.time()

    

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'tut4-model.pt')

    

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
print(train_losses, valid_losses)
model.load_state_dict(torch.load('tut4-model.pt'))



test_loss = evaluate(model, test_iterator, criterion)



print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
def translate_sentence(model, tokenized_sentence):

    model.eval()

    tokenized_sentence = ['<sos>'] + [t.lower() for t in tokenized_sentence] + ['<eos>']

    numericalized = [SRC.vocab.stoi[t] for t in tokenized_sentence] 

    sentence_length = torch.LongTensor([len(numericalized)]).to(device) 

    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device) 

    translation_tensor_logits, attention = model(tensor, sentence_length, None, 0) 

    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)

    translation = [TRG.vocab.itos[t] for t in translation_tensor]

    translation, attention = translation[1:], attention[1:]

    return translation, attention
# def predict(model, data_iterator):

#     with torch.no_grad():

    

#         for i, batch in enumerate(data_iterator):

#             src, src_len = batch.vi_no_accents

#             trg = batch.vi

#             output, attention = model(src, src_len, trg, 0)

#             print(output.shape)

#             break
# predict(model,train_iterator)

def display_attention(sentence, translation, attention):

    

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)

    

    attention = attention.squeeze(1).cpu().detach().numpy()

    

    cax = ax.matshow(attention, cmap='bone')

   

    ax.tick_params(labelsize=15)

    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 

                       rotation=45)

    ax.set_yticklabels(['']+translation)



    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))



    plt.show()

    plt.close()
example_idx = 2000



src = vars(train_data.examples[example_idx])['vi_no_accents']

trg = vars(train_data.examples[example_idx])['vi']



print(f'src = {src}')

print(f'trg = {trg}')
translation, attention = translate_sentence(model, src)



print(f'predicted trg = {translation}')
display_attention(src, translation, attention)
example_idx = 35



src = vars(test_data.examples[example_idx])['vi_no_accents']

trg = vars(test_data.examples[example_idx])['vi']



print(f'src = {src}')

print(f'trg = {trg}')
example_idx = 1809



src = vars(test_data.examples[example_idx])['vi_no_accents']

trg = vars(test_data.examples[example_idx])['vi']



print(f'src = {src}')

print(f'trg = {trg}')
translation, attention = translate_sentence(model, src)



print(f'predicted trg = {translation}')



display_attention(src, translation, attention)
torch.save(model.state_dict(), 'model.pth')