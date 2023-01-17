!pip install comet_ml
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from comet_ml import Experiment

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch import Tensor

 

from torchtext.datasets import TranslationDataset, Multi30k, IWSLT

from torchtext.data import Field, BucketIterator

import spacy

import os

from typing import Tuple

import random

import math

import time



from typing import Tuple



SEED = 1001

 

def seed_torch(seed=1):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

 

 

seed_torch(SEED)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!python -m spacy download en

!python -m spacy download de
api = 'GGF21Vtrnid3Cgat9n1nL9Vcc'



SRC = Field(tokenize = "spacy",

            tokenizer_language="en",

            init_token = '<sos>',

            eos_token = '<eos>',

            lower = True)



TRG = Field(tokenize = "spacy",

            tokenizer_language="de",

            init_token = '<sos>',

            eos_token = '<eos>',

            lower = True)



train_data, valid_data, test_data = IWSLT.splits(exts = ('.en', '.de'),

                                                    fields = (SRC, TRG))
SRC.build_vocab(train_data, min_freq = 2)

TRG.build_vocab(train_data, min_freq = 2)
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



BATCH_SIZE = 16



train_iterator, valid_iterator, test_iterator = BucketIterator.splits(

    (train_data, valid_data, test_data),

    batch_size = BATCH_SIZE,

    device = device)
class Encoder(nn.Module):

    def __init__(self,

                 input_dim: int,

                 emb_dim: int,

                 enc_hid_dim: int,

                 dec_hid_dim: int,

                 dropout: float):

        super().__init__()



        self.input_dim = input_dim

        self.emb_dim = emb_dim

        self.enc_hid_dim = enc_hid_dim

        self.dec_hid_dim = dec_hid_dim

        self.dropout = dropout



        self.embedding = nn.Embedding(input_dim, emb_dim)



        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)



        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)



        self.dropout = nn.Dropout(dropout)



    def forward(self,

                src: Tensor) -> Tuple[Tensor]:



        embedded = self.dropout(self.embedding(src))



        outputs, hidden = self.rnn(embedded)



        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))



        return outputs, hidden





class Attention(nn.Module):

    def __init__(self,

                 enc_hid_dim: int,

                 dec_hid_dim: int,

                 attn_dim: int):

        super().__init__()

       

        self.enc_hid_dim = enc_hid_dim

        self.dec_hid_dim = dec_hid_dim

       

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

       

        self.attn = nn.Linear(self.attn_in, attn_dim)

        self.v = nn.Parameter(torch.rand(attn_dim))

       

    def forward(self,

                decoder_hidden: Tensor,

                encoder_outputs: Tensor) -> Tensor:

       

        #hidden = [batch size, dec hid dim]

        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]

       

        batch_size = encoder_outputs.shape[1]

        src_len = encoder_outputs.shape[0]

       

        #repeat decoder hidden state src_len times

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

       

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

       

        #decoder_hidden = [batch size, src sent len, dec hid dim]

        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]

       

        # Step 1: to enable feeding through "self.attn" pink box above, concatenate

        # `repeated_decoder_hidden` and `encoder_outputs`:

        # torch.cat((hidden, encoder_outputs), dim = 2) has shape

        # [batch_size, seq_len, enc_hid_dim * 2 + dec_hid_dim]

       

        # Step 2: feed through self.attn to end up with:

        # [batch_size, seq_len, attn_dim]

       

        # Step 3: feed through tanh      

       

        energy = torch.tanh(self.attn(torch.cat((

            repeated_decoder_hidden,

            encoder_outputs),

            dim = 2)))

       

        #energy = [batch size, src sent len, attn_dim]

       

        energy = energy.permute(0, 2, 1)

       

        #energy = [batch size, attn_dim, src sent len]

       

        #v = [attn_dim]

       

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

       

        #v = [batch size, 1, attn_dim]

       

        # High level: energy a function of both encoder element outputs and most recent decoder hidden state,

        # of shape attn_dim x enc_seq_len for each observation

        # v, being 1 x attn_dim, transforms this into a vector of shape 1 x enc_seq_len for each observation

        # Then, we take the softmax over these to get the output of the attention function

 

        attention = torch.bmm(v, energy).squeeze(1)

       

        #attention= [batch size, src len]

       

        return F.softmax(attention, dim=1)





class Decoder(nn.Module):

    def __init__(self,

                 output_dim: int,

                 emb_dim: int,

                 enc_hid_dim: int,

                 dec_hid_dim: int,

                 dropout: int,

                 attention: nn.Module):

        super().__init__()



        self.emb_dim = emb_dim

        self.enc_hid_dim = enc_hid_dim

        self.dec_hid_dim = dec_hid_dim

        self.output_dim = output_dim

        self.dropout = dropout

        self.attention = attention



        self.embedding = nn.Embedding(output_dim, emb_dim)



        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)



        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)



        self.dropout = nn.Dropout(dropout)





    def _weighted_encoder_rep(self,

                              decoder_hidden: Tensor,

                              encoder_outputs: Tensor) -> Tensor:



        a = self.attention(decoder_hidden, encoder_outputs)



        a = a.unsqueeze(1)



        encoder_outputs = encoder_outputs.permute(1, 0, 2)



        weighted_encoder_rep = torch.bmm(a, encoder_outputs)



        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)



        return weighted_encoder_rep





    def forward(self,

                input: Tensor,

                decoder_hidden: Tensor,

                encoder_outputs: Tensor) -> Tuple[Tensor]:



        input = input.unsqueeze(0)



        embedded = self.dropout(self.embedding(input))



        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,

                                                          encoder_outputs)



        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)



        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))



        embedded = embedded.squeeze(0)

        output = output.squeeze(0)

        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)



        output = self.out(torch.cat((output,

                                     weighted_encoder_rep,

                                     embedded), dim = 1))



        return output, decoder_hidden.squeeze(0)





class Seq2Seq(nn.Module):

    def __init__(self,

                 encoder: nn.Module,

                 decoder: nn.Module,

                 device: torch.device):

        super().__init__()



        self.encoder = encoder

        self.decoder = decoder

        self.device = device



    def forward(self,

                src: Tensor,

                trg: Tensor,

                teacher_forcing_ratio: float = 0.5) -> Tensor:



        batch_size = src.shape[1]

        max_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim



        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)



        encoder_outputs, hidden = self.encoder(src)



        # first input to the decoder is the <sos> token

        output = trg[0,:]



        for t in range(1, max_len):

            output, hidden = self.decoder(output, hidden, encoder_outputs)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.max(1)[1]

            output = (trg[t] if teacher_force else top1)



        return outputs





INPUT_DIM = len(SRC.vocab)

OUTPUT_DIM = len(TRG.vocab)

ENC_EMB_DIM = 32

DEC_EMB_DIM = 32

ENC_HID_DIM = 64

DEC_HID_DIM = 64

ATTN_DIM = 8

ENC_DROPOUT = 0.5

DEC_DROPOUT = 0.5



enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)



attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)



dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)



model = Seq2Seq(enc, dec, device).to(device)





def init_weights(m: nn.Module):

    for name, param in m.named_parameters():

        if 'weight' in name:

            nn.init.normal_(param.data, mean=0, std=0.01)

        else:

            nn.init.constant_(param.data, 0)





model.apply(init_weights)



optimizer = optim.Adam(model.parameters())





def count_parameters(model: nn.Module):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)





print(f'The model has {count_parameters(model):,} trainable parameters')
PAD_IDX = TRG.vocab.stoi['<pad>']



MODEL_PATH = './EN_to_DE_IWSLT.pt'

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
import spacy



trg = 'Möchtest du eine Tasse Kaffee?'

src = 'Do you want a cup of coffee?'

 

spacy_de = spacy.load('de')

spacy_en = spacy.load('en')

 

SOS_DE = TRG.vocab.stoi['<sos>']

EOS_DE = TRG.vocab.stoi['<eos>']

 

MAX_LEN = 50

 

def tokenize_de(text):

    """

    Tokenizes German text from a string into a list of strings

    """

    return [tok.text for tok in spacy_de.tokenizer(text)]

 



def tokenize_en(text):

    """

    Tokenizes English text from a string into a list of strings

    """

    return [tok.text for tok in spacy_en.tokenizer(text)]

 

def translate(model, src):

   

    src = ['<sos>'] + tokenize_de(src.lower()) + ['<eos>']

    src = [SRC.vocab.stoi[el] for el in src]

    #print(src)

   

    src = torch.tensor(src, device=device).unsqueeze(1)

   

    encoder_outputs, hidden = model.encoder(src)

   

    # first input to the decoder is the <sos> token

    output = torch.tensor([SOS_DE], device=device)

   

    trg_vocab_size = model.decoder.output_dim

    outputs = torch.zeros(MAX_LEN, 1, trg_vocab_size).to(device)

   

    translation = ''

    for t in range(1, MAX_LEN):

        translation += TRG.vocab.itos[output]

        translation += ' '

        #input = [batch size]

        #hidden = [batch size, dec hid dim]

        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        output, hidden = model.decoder(output, hidden, encoder_outputs)

        outputs[t] = output

        output = output.max(1)[1]

    return translation

 

import math

import time



experiment = Experiment(api_key=api, project_name="EN to DE RNN with attention", workspace="comet-ml testing")



def train(model: nn.Module,

          iterator: BucketIterator,

          optimizer: optim.Optimizer,

          criterion: nn.Module,

          clip: float):



    model.train()



    epoch_loss = 0

    for _, batch in enumerate(iterator):



        src = batch.src

        trg = batch.trg



        optimizer.zero_grad()



        output = model(src, trg)



        output = output[1:].view(-1, output.shape[-1])

        trg = trg[1:].view(-1)



        loss = criterion(output, trg)

        

        experiment.log_metric("train loss", loss.item())

        

        loss.backward()



        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)



        optimizer.step()



        epoch_loss += loss.item()



    return epoch_loss / len(iterator)





def evaluate(model: nn.Module,

             iterator: BucketIterator,

             criterion: nn.Module):



    model.eval()



    epoch_loss = 0



    with torch.no_grad():



        for _, batch in enumerate(iterator):



            src = batch.src

            trg = batch.trg



            output = model(src, trg, 0) #turn off teacher forcing



            output = output[1:].view(-1, output.shape[-1])

            trg = trg[1:].view(-1)



            loss = criterion(output, trg)



            epoch_loss += loss.item()



    return epoch_loss / len(iterator)





def epoch_time(start_time: int,

               end_time: int):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs





N_EPOCHS = 5

CLIP = 1

 

best_valid_loss = float('inf')

 

for epoch in range(N_EPOCHS):

    start_time = time.time()

 

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)

    valid_loss = evaluate(model, valid_iterator, criterion)

    experiment.log_metric("validation loss", train_loss)

    experiment.log_metric("epoch", epoch)

   

    end_time = time.time()

 

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

 

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), MODEL_PATH)

       

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    translation = translate(model, src)

    experiment.log_text("EN src: " + src + "\n" + "DE trg: " + trg + "\n" + "DE res: " + translation)

    experiment.log_epoch_end(N_EPOCHS)



experiment.end()



test_loss = evaluate(model, test_iterator, criterion)



print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
experiment.end()