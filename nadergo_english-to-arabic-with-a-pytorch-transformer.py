"""
torch modules
with ofc  numpy  and pandas

"""

import pandas as pd
import numpy as np 

from torch import nn
import torch
from torchtext import data
from torch.nn  import functional as F
import torch.optim as  optim 
if torch.cuda.is_available():  
  dev = "cuda:0" 

  print("gpu up")
else:  
  dev = "cpu"  
device = torch.device(dev)
import random
SEED= 32
"""
regex and the tokenizers
"""

import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.ar import Arabic
from nltk.translate.bleu_score import sentence_bleu

enNLP = English()
arNLP = Arabic()

enTokenizer = Tokenizer(enNLP.vocab)
arTokenizer =  Tokenizer(arNLP.vocab)



df = pd.read_csv("/kaggle/input/arabic-to-english-translation-sentences/ara_eng.txt",delimiter="\t",names=["eng","ar"])
df
"""
defining the tokenizers for arabic and english  

creating the fields for the dataset from torchtext 
that class is the simple way I could find for turning a df into a torch dataset

نهها and ببدأ are just arbitrary words for init and end of sentence tokens  
for some reason when I choose an arabic word for the unknown token  the vocab doesn't replace words that are not in the vocab  
"""

def myTokenizerEN(x):
 return  [word.text for word in 
          enTokenizer(re.sub(r"\s+\s+"," ",re.sub(r"[\.\'\`\"\r+\n+]"," ",x.lower())).strip())]
def myTokenizerAR(x):
 return  [word.text for word in 
          arTokenizer(re.sub(r"\s+\s+"," ",re.sub(r"[\.\'\`\"\r+\n+]"," ",x.lower())).strip())]

SRC = data.Field(tokenize=myTokenizerEN,batch_first=False,init_token="<sos>",eos_token="<eos>")
TARGET = data.Field(tokenize=myTokenizerAR,batch_first=False,tokenizer_language="ar",init_token="ببدأ",eos_token="نهها")

class DataFrameDataset(data.Dataset):

    def __init__(self, df, src_field, target_field, is_test=False, **kwargs):
        fields = [('eng', src_field), ('ar',target_field)]
        examples = []
        for i, row in df.iterrows():
            eng = row.eng 
            ar = row.ar
            examples.append(data.Example.fromlist([eng, ar], fields))

        super().__init__(examples, fields, **kwargs)

        
torchdataset = DataFrameDataset(df,SRC,TARGET)

train_data, valid_data = torchdataset.split(split_ratio=0.8, random_state = random.seed(SEED))
SRC.build_vocab(train_data,min_freq=2)
TARGET.build_vocab(train_data,min_freq=2)  


#Commonly used words
print(TARGET.vocab.freqs.most_common(10))  

"""
we are using batches for validation and test set because of memory usage we can't pass the whole set at once

try lowering the batch size if you are out of memory 
"""
BATCH_SIZE = 64



train_iterator,valid_iterator = data.BucketIterator.splits(
    (train_data,valid_data), 
    batch_size = BATCH_SIZE,
    device = device,
    sort=False,
    sort_within_batch=False,
shuffle=True)
"""
to point out one thing about the transformer what it could do is to enable 
training on the whole sequence at once but on really using it for translation it predicts the next word 
then it feeds the prediction into the sequence again until the model predict <eos> token (with a max length ofc)

"""
class TranslateTransformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        max_len,
    ):
        super(TranslateTransformer, self).__init__()
        self.srcEmbeddings = nn.Embedding(src_vocab_size,embedding_size)
        self.trgEmbeddings= nn.Embedding(trg_vocab_size,embedding_size)
        self.srcPositionalEmbeddings= nn.Embedding(max_len,embedding_size)
        self.trgPositionalEmbeddings= nn.Embedding(max_len,embedding_size)
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.src_pad_idx = src_pad_idx
        self.max_len = max_len
    
    def make_src_mask(self, src):
        src_mask = src.transpose(0,1) == self.src_pad_idx

        return src_mask.to(device)

    def forward(self,x,trg):
        src_seq_length = x.shape[0]
        N = x.shape[1]
        trg_seq_length = trg.shape[0]
        #adding zeros is an easy way
        src_positions = (
            torch.arange(0, src_seq_length)
            .reshape(src_seq_length,1)  + torch.zeros(src_seq_length,N) 
        ).to(device)
        
        trg_positions = (
            torch.arange(0, trg_seq_length)
            .reshape(trg_seq_length,1)  + torch.zeros(trg_seq_length,N) 
        ).to(device)


        srcWords = self.dropout(self.srcEmbeddings(x.long()) +self.srcPositionalEmbeddings(src_positions.long()))
        trgWords = self.dropout(self.trgEmbeddings(trg.long())+self.trgPositionalEmbeddings(trg_positions.long()))
        
        src_padding_mask = self.make_src_mask(x)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(device)
        
        
        out = self.transformer(srcWords,trgWords, src_key_padding_mask=src_padding_mask,tgt_mask=trg_mask )
        out= self.fc_out(out)
        return out
        


#No. of unique tokens in text
src_vocab_size  = len(SRC.vocab)
print("Size of english vocabulary:",src_vocab_size)

#No. of unique tokens in label
trg_vocab_size =len(TARGET.vocab)
print("Size of arabic vocabulary:",trg_vocab_size)

num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3

max_len= 227
embedding_size= 256
src_pad_idx =SRC.vocab.stoi["<pad>"]


model = TranslateTransformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    max_len
).to(device)
loss_track = []
loss_validation_track= []
"""
I'm using adagrad because it assigns bigger updates to less frequently updated weights so 
so thought it could be useful for words not used a lot 
"""

optimizer = optim.Adagrad(model.parameters(),lr = 0.003)
EPOCHS = 60


pad_idx = SRC.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) 

for i in range(50,EPOCHS):
    stepLoss=[]
    model.train() # the training mode for the model (applies dropout and batchnorms)
    for batch  in train_iterator:
        input_sentence = batch.eng.to(device)
        trg = batch.ar.to(device)

        optimizer.zero_grad()
        out = model(input_sentence,trg[:-1])
        out = out.reshape(-1,trg_vocab_size)
        trg = trg[1:].reshape(-1)
        loss = criterion(out,trg)
        
        
        loss.backward()
        optimizer.step()
        stepLoss.append(loss.item())
        

    loss_track.append(np.mean(stepLoss))
    print("train crossentropy at epoch {} loss: ".format(i),np.mean(stepLoss))
    
    stepValidLoss=[]
    model.eval() # the evaluation mode for the model (doesn't apply dropout and batchNorm)
    for batch  in valid_iterator:
        input_sentence = batch.eng.to(device)
        trg = batch.ar.to(device)

        optimizer.zero_grad()
        out = model(input_sentence,trg[:-1])
        out = out.reshape(-1,trg_vocab_size)
        trg = trg[1:].reshape(-1)
        loss = criterion(out,trg)
        
        stepValidLoss.append(loss.item())
  
    loss_validation_track.append(np.mean(stepValidLoss))
    print("validation crossentropy at epoch {} loss: ".format(i),np.mean(stepValidLoss))
    
    
        
import matplotlib.pyplot as plt 

#the train loss after 50 epoch
plt.figure(figsize=(10,5))
plt.plot(range(60),loss_track,label="train loss")
plt.plot(range(60),loss_validation_track,label="valiadtion loss")
plt.legend()
plt.show()
"""
this function takes some arguments and returns the translated arabic sentence 

"""

def translate(model,sentence,srcField,targetField,srcTokenizer):
    model.eval()
    processed_sentence = srcField.process([srcTokenizer(sentence)]).to(device)
    trg = ["ببدأ"]
    for _ in range(60):
        
        trg_indecies = [targetField.vocab.stoi[word] for word in trg]
        outputs = torch.Tensor(trg_indecies).unsqueeze(1).to(device)
        outputs = model(processed_sentence,outputs)
        
        if targetField.vocab.itos[outputs.argmax(2)[-1:].item()] == "<unk>":
            continue 
        trg.append(targetField.vocab.itos[outputs.argmax(2)[-1:].item()])
        if targetField.vocab.itos[outputs.argmax(2)[-1:].item()] == "نهها":
            break
    return " ".join([word for word in trg if word != "<unk>"][1:-1])
    
    
translate(model,"I'm happy" ,SRC,TARGET,myTokenizerEN)
translate(model,"what do you want" ,SRC,TARGET,myTokenizerEN) 
translate(model,"what do you like to have " ,SRC,TARGET,myTokenizerEN) 
translate(model,"I am going outside" ,SRC,TARGET,myTokenizerEN) 
translate(model,"he is here" ,SRC,TARGET,myTokenizerEN) 
translate(model,"he is not here" ,SRC,TARGET,myTokenizerEN) 
translate(model,"I need help" ,SRC,TARGET,myTokenizerEN)
translate(model,"I'm not at home" ,SRC,TARGET,myTokenizerEN)  # it totally get negation right 

translate(model,"I'm ready" ,SRC,TARGET,myTokenizerEN)

print(translate(model,"it's not important" ,SRC,TARGET,myTokenizerEN) )
"""
and the model also can't handle any slightly complicated sentences 
"""
translate(model,"he apologized for the mistake he did" ,SRC,TARGET,myTokenizerEN)