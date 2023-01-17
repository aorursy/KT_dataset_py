import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



from torchtext.data import TabularDataset

from torchtext.vocab import Vectors

from torchtext.data import Field, BucketIterator, Iterator



import spacy

import numpy as np

from nltk.corpus import stopwords

import string



import random

import math

import time

import pandas as pd

import os
spacy_en = spacy.load('en')
stopwords = set(stopwords.words('english'))
SEED = 1234



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
def tokenize_en(text):

    """

    Tokenizes English text from a string into a list of strings (tokens)

    """

    return [tok.text for tok in spacy_en.tokenizer(text)]
def remove_punctuation(text):

    words = [wrd for wrd in text if wrd not in string.punctuation]

    return [word for word in words if not word.startswith('http')]
SRC = Field(tokenize = tokenize_en,

            preprocessing=remove_punctuation,

            init_token = '<sos>', 

            eos_token = '<eos>', 

            lower = True)



TRG = Field(tokenize = tokenize_en,

            preprocessing=remove_punctuation,

            init_token = '<sos>', 

            eos_token = '<eos>', 

            lower = True)
fields = {'text': ('text', SRC),'selected_text': ('label', TRG)}
dataset = TabularDataset(path='/kaggle/input/tweet-sentiment-extraction/train.csv',

                        format='csv',

                        fields=fields)
train, valid = dataset.split(split_ratio=0.75)
# SRC.build_vocab(train, min_freq = 2)

# TRG.build_vocab(train, min_freq = 2)



cache = '/kaggle/working/vector_cache'

if not os.path.exists(cache):

    os.mkdir(cache)

vector = Vectors(name='/kaggle/input/glove6b/glove.6B.100d.txt', cache=cache)

SRC.build_vocab(train,

                 vectors = vector, 

                 unk_init = torch.Tensor.normal_)



TRG.build_vocab(train,

                 vectors = vector, 

                 unk_init = torch.Tensor.normal_)
print(vars(train.examples[0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ID = Field()
fields = {'textID': ('ID', ID), 'text':('text', SRC)}
test = TabularDataset(path='/kaggle/input/tweet-sentiment-extraction/test.csv',

                        format='csv',

                        fields=fields)
ID.build_vocab(test)
ID.vocab.itos[2877]
test.examples[1].text
train_iterator, valid_iterator, _ = BucketIterator.splits((train, valid, None),batch_sizes=(128, 128, 0),

                                                            sort_key=lambda x: len(x.text), device=device)
test_iterator = Iterator(dataset=test,device='cpu', batch_size=1, sort=False)
class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):

        super().__init__()

        

        self.embedding = nn.Embedding(input_dim, emb_dim)

        

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, src):

        

        #src = [src len, batch size]

        

        embedded = self.dropout(self.embedding(src))

        

        #embedded = [src len, batch size, emb dim]

        

        outputs, hidden = self.rnn(embedded)

                

        #outputs = [src len, batch size, hid dim * num directions]

        #hidden = [n layers * num directions, batch size, hid dim]

        

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]

        #outputs are always from the last layer

        

        #hidden [-2, :, : ] is the last of the forwards RNN 

        #hidden [-1, :, : ] is the last of the backwards RNN

        

        #initial decoder hidden is final hidden state of the forwards and backwards 

        #  encoder RNNs fed through a linear layer

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        

        #outputs = [src len, batch size, enc hid dim * 2]

        #hidden = [batch size, dec hid dim]

        

        return outputs, hidden
class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):

        super().__init__()

        

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)

        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

        

    def forward(self, hidden, encoder_outputs):

        

        #hidden = [batch size, dec hid dim]

        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        

        batch_size = encoder_outputs.shape[1]

        src_len = encoder_outputs.shape[0]

        

        #repeat decoder hidden state src_len times

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        

        #hidden = [batch size, src len, dec hid dim]

        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 

        

        #energy = [batch size, src len, dec hid dim]



        attention = self.v(energy).squeeze(2)

        

        #attention= [batch size, src len]

        

        return F.softmax(attention, dim=1)
class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):

        super().__init__()



        self.output_dim = output_dim

        self.attention = attention

        

        self.embedding = nn.Embedding(output_dim, emb_dim)

        

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, input, hidden, encoder_outputs):

             

        #input = [batch size]

        #hidden = [batch size, dec hid dim]

        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        

        input = input.unsqueeze(0)

        

        #input = [1, batch size]

        

        embedded = self.dropout(self.embedding(input))

        

        #embedded = [1, batch size, emb dim]

        

        a = self.attention(hidden, encoder_outputs)

                

        #a = [batch size, src len]

        

        a = a.unsqueeze(1)

        

        #a = [batch size, 1, src len]

        

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        

        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        

        weighted = torch.bmm(a, encoder_outputs)

        

        #weighted = [batch size, 1, enc hid dim * 2]

        

        weighted = weighted.permute(1, 0, 2)

        

        #weighted = [1, batch size, enc hid dim * 2]

        

        rnn_input = torch.cat((embedded, weighted), dim = 2)

        

        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

            

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        

        #output = [seq len, batch size, dec hid dim * n directions]

        #hidden = [n layers * n directions, batch size, dec hid dim]

        

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:

        #output = [1, batch size, dec hid dim]

        #hidden = [1, batch size, dec hid dim]

        #this also means that output == hidden

        assert (output == hidden).all()

        

        embedded = embedded.squeeze(0)

        output = output.squeeze(0)

        weighted = weighted.squeeze(0)

        

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        

        #prediction = [batch size, output dim]

        

        return prediction, hidden.squeeze(0)
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):

        super().__init__()

        

        self.encoder = encoder

        self.decoder = decoder

        self.device = device

        

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        

        if trg is None:

            trg = torch.zeros((25, src.shape[1])).fill_(2).long().to(src.device)

            assert teacher_forcing_ratio == 0, "Must be zero during inference"

        

        #src = [src len, batch size]

        #trg = [trg len, batch size]

        #teacher_forcing_ratio is probability to use teacher forcing

        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        

        batch_size = src.shape[1]

        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim

        

        #tensor to store decoder outputs

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        

        #encoder_outputs is all hidden states of the input sequence, back and forwards

        #hidden is the final forward and backward hidden states, passed through a linear layer

        encoder_outputs, hidden = self.encoder(src)

                

        #first input to the decoder is the <sos> tokens

        input = trg[0,:]

        

        for t in range(1, trg_len):

            

            #insert input token embedding, previous hidden state and all encoder hidden states

            #receive output tensor (predictions) and new hidden state

            output, hidden = self.decoder(input, hidden, encoder_outputs)

            

            #place predictions in a tensor holding predictions for each token

            outputs[t] = output

            

            #decide if we are going to use teacher forcing or not

            teacher_force = random.random() < teacher_forcing_ratio

            

            #get the highest predicted token from our predictions

            top1 = output.argmax(1) 

            

            #if teacher forcing, use actual next token as next input

            #if not, use predicted token

            input = trg[t] if teacher_force else top1



        return outputs
INPUT_DIM = len(SRC.vocab)

OUTPUT_DIM = len(TRG.vocab)

ENC_EMB_DIM = 100

DEC_EMB_DIM = 100

ENC_HID_DIM = 512

DEC_HID_DIM = 512

ENC_DROPOUT = 0.5

DEC_DROPOUT = 0.5



attn = Attention(ENC_HID_DIM, DEC_HID_DIM)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)



model = Seq2Seq(enc, dec, device).to(device)
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

pretrained_embeddings = SRC.vocab.vectors



print(pretrained_embeddings)
model.encoder.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = SRC.vocab.stoi[SRC.unk_token]

PAD_IDX = SRC.vocab.stoi[SRC.pad_token]





model.encoder.embedding.weight.data[UNK_IDX] = torch.zeros(ENC_EMB_DIM)

model.encoder.embedding.weight.data[PAD_IDX] = torch.zeros(ENC_EMB_DIM)



print(model.encoder.embedding.weight.data)
optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]



criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
def train(model, iterator, optimizer, criterion, clip):

    

    model.train()

    

    epoch_loss = 0

    

    for i, batch in enumerate(iterator):

        

        src = batch.text

        trg = batch.label

        

        optimizer.zero_grad()

        

        output = model(src, trg)

        

        #trg = [trg len, batch size]

        #output = [trg len, batch size, output dim]

        

        output_dim = output.shape[-1]

        

        output = output[1:].view(-1, output_dim)

        trg = trg[1:].view(-1)

        

        #trg = [(trg len - 1) * batch size]

        #output = [(trg len - 1) * batch size, output dim]

        

        loss = criterion(output, trg)

        

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



            src = batch.text

            trg = batch.label



            output = model(src, trg, 0) #turn off teacher forcing



            #trg = [trg len, batch size]

            #output = [trg len, batch size, output dim]



            output_dim = output.shape[-1]

            

            output = output[1:].view(-1, output_dim)

            trg = trg[1:].view(-1)



            #trg = [(trg len - 1) * batch size]

            #output = [(trg len - 1) * batch size, output dim]



            loss = criterion(output, trg)



            epoch_loss += loss.item()

        

    return epoch_loss / len(iterator)
def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
N_EPOCHS = 10 #25

CLIP = 1



best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):

    

    start_time = time.time()

    

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)

    valid_loss = evaluate(model, valid_iterator, criterion)

    

    end_time = time.time()

    

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'tut3-model.pt')

    

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
def translate_sentence(idx, numericalized):

    

    tensor = torch.LongTensor(numericalized) #convert to tensor and add batch dimension

    translation_tensor_probs = model(tensor, None, 0).squeeze(1) #pass through model to get translation probabilities

    translation_tensor = torch.argmax(translation_tensor_probs, 1) #get translation from highest probabilities

    translation = [TRG.vocab.itos[t] for t in translation_tensor if TRG.vocab.itos[t] not in ['<eos>','<unk>']][1:] #we ignore the first token, just like we do in the training loop

    output_seq = " "

    for trans in translation:

        if trans not in output_seq:

            output_seq += trans+" "

    return (ID.vocab.itos[idx.item()], output_seq.strip())
model.load_state_dict(torch.load('/kaggle/working/tut3-model.pt'))

model.to('cpu')
textID = []

selected_text = []

for i, batch in enumerate(test_iterator):

    src = batch.text

    idx = batch.ID

    output = translate_sentence(idx,src)

    textID.append(output[0])

    selected_text.append(output[1])
submission_file = pd.DataFrame({"textID": textID, "selected_text": selected_text})
submission_file.head()
submission_file.to_csv("submission.csv",index=False)