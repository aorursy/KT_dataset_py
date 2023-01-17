# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile
for file in ['train_tcn.csv.zip','train_en.csv.zip']:
  with zipfile.ZipFile(file, 'r') as zip_ref:
      zip_ref.extractall()
import pandas as pd
df_cn=pd.read_csv('train_tcn.csv')
df_cn.head(10)
from google.colab import drive
drive.mount('/content/drive/')
!pip install transformers
from transformers import BertTokenizer
from transformers import BertModel
en_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased') 
# en_bert_layer=BertModel.from_pretrained('/content/drive/My Drive/BERT_CONFIG')
cn_tokenizer=BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# cn_bert_layer=BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",is_decoder=True)
import torch
import torch.nn as nn
import torch.nn.parameter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
en_tokenizer.vocab.update({'<s>':len(en_tokenizer),'</s>':len(en_tokenizer)+1})
cn_tokenizer.vocab.update({'<s>':len(cn_tokenizer),'</s>':len(cn_tokenizer)+1})
class Monolingual_CN_Dataset(Dataset):
  def __init__(self,df):
    self.df=df
    self.cn_pad_id=cn_tokenizer.convert_tokens_to_ids('[PAD]')     

  def __len__(self): 
    return len(self.df)

  def __getitem__(self,index):
    tokens=cn_tokenizer.tokenize(str(self.df.iloc[index]['product_title']))
    # tokens=analyzer.parse(self.df.iloc[index]['product_title']).tokens()
    cn_encoding=[cn_tokenizer.convert_tokens_to_ids(token) for token in tokens]
    return {'src':tokens,'x_len':len(tokens),'x':cn_encoding}
def pad(tokens,en=True):
  en_pad_id=en_tokenizer.convert_tokens_to_ids('[PAD]')
  cn_pad_id=cn_tokenizer.convert_tokens_to_ids('[PAD]')
  if(len(tokens)<50):
    if(en):
      return tokens+[en_pad_id for _ in range(50-len(tokens))]
    else:
      return tokens+[cn_pad_id for _ in range(50-len(tokens))]      
  else:
    return tokens[:50]

def my_collate(batch,en=True):
    max_x=np.max([item['x_len'] for item in batch])
    if(max_x>50):
      max_x=50 
    x = [pad(item['x'],en=False) for item in batch]
    if(en):
      attn_mask=[list(map(lambda tok: [1 if tok!=\
                    en_tokenizer.convert_tokens_to_ids('[PAD]') else 0],item)) for item in x ]    
    else:
      attn_mask=[list(map(lambda tok: [1 if tok!=\
                    cn_tokenizer.convert_tokens_to_ids('[PAD]') else 0],item)) for item in x ]       
    src=[item['src'] for item in batch]
    return {'x':x,'src':src,'attn_mask':attn_mask}
data=df_cn.sample(10000,axis=0)
msk = np.random.rand(len(data)) < 0.8
train=data[msk]
val=data[~msk]
train_set=Monolingual_CN_Dataset(train)
val_set=Monolingual_CN_Dataset(val)
train_loader=DataLoader(train_set, batch_size = 20,collate_fn=my_collate)
val_loader = DataLoader(val_set, batch_size = 20,collate_fn=my_collate)  
import string
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
class Translator_EN_CN(nn.Module):
  def __init__(self):
        super(Translator_EN_CN, self).__init__()
        # self.enc_layer = BertModel.from_pretrained('bert-base-uncased',\
        #                                            output_hidden_states=True).half()
        self.enc_layer=BERT_en_encoder
        for p in self.enc_layer.parameters():
            p.requires_grad = False        
        # self.dec_layer = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext"\
        #                             ,is_decoder=True,num_hidden_layers=1,num_attention_heads=1)
        self.dec_layer = Decoder(768,len(cn_tokenizer))
        self.dec_layer.load_my_state_dict(BERT_cn_encoder.state_dict())        
        self.linear = nn.Linear(768, len(cn_tokenizer))        
        self.copy = nn.Linear(768,768)
        self.enc_layer.resize_token_embeddings(len(en_tokenizer))
        # self.dec_layer.resize_token_embeddings(len(cn_tokenizer))
  def load_my_state_dict(self, state_dict):
      own_state = self.state_dict()
      for name, param in state_dict.items():
          if name not in own_state:
                continue
          if isinstance(param, torch.nn.Parameter):
              # backwards compatibility for serialized parameters
              param = param.data
          own_state[name].copy_(param)        
  def forward(self, en_encoding,en_attn_mask,target,length=50):      
      enc_hidden,_,all_hidden_layer = self.enc_layer(en_encoding, attention_mask = en_attn_mask)
      batch_size=enc_hidden.size(0)
      # pred=torch.zeros((batch_size,512),device=device)
      y=torch.tensor([cn_tokenizer.convert_tokens_to_ids('<s>')] * batch_size,dtype=torch.long,device=device).view(-1,1)
      for _e in range(length):     
        output=self.dec_layer(y,all_hidden_layer)
        # print(output.shape)
        # _,output=self.dec_layer(input_ids=y,encoder_hidden_states=enc_hidden,encoder_attention_mask=en_attn_mask)        
        transformed=self.copy(output[:,-1:,:])
        copy_scores=torch.bmm(enc_hidden,transformed.transpose(1,2))  
        # copy_scores=torch.bmm(enc_hidden,transformed.unsqueeze(-1))  
        gen_scores=self.linear(output[:,-1:,:])
        # y=torch.tensor(pred[:,_e],dtype=torch.long,device=device).view(-1,1) 
        # own decoder       
        y_pred=torch.tensor(torch.argmax(gen_scores,dim=-1),dtype=torch.long,device=device).view(-1,1) 
        # teacher forcing
        y_teacher=target[:,_e].view(-1,1)    
        y_pred=y_teacher*(y_teacher<len(cn_tokenizer))+\
            (y_teacher>len(cn_tokenizer))*y_pred
        y=torch.cat((y,y_pred),dim=-1)
        yield gen_scores,copy_scores 
class Translator_CN_EN(nn.Module):
  def __init__(self):
        super(Translator_CN_EN, self).__init__()
        # self.enc_layer = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",\
        #                                            output_hidden_states=True).half()
        self.enc_layer=BERT_cn_encoder
        for p in self.enc_layer.parameters():
            p.requires_grad = False                                
        self.dec_layer = Decoder(768,len(en_tokenizer))
        self.dec_layer.load_my_state_dict(BERT_en_encoder.state_dict())
        self.linear = nn.Linear(768, len(en_tokenizer))        
        self.copy = nn.Linear(768,768)
        self.enc_layer.resize_token_embeddings(len(cn_tokenizer))
        # self.dec_layer.resize_token_embeddings(len(en_tokenizer))
  def load_my_state_dict(self, state_dict):
      own_state = self.state_dict()
      for name, param in state_dict.items():
          if name not in own_state:
                continue
          if isinstance(param, torch.nn.Parameter):
              # backwards compatibility for serialized parameters
              param = param.data
          own_state[name].copy_(param)
  def forward(self, cn_encoding,cn_attn_mask,length=50):
      enc_hidden,_,all_layer_hidden = self.enc_layer(cn_encoding, attention_mask = cn_attn_mask)      
      batch_size=enc_hidden.size(0)
      # pred=torch.zeros((batch_size,512),device=device)
      y=torch.tensor([en_tokenizer.convert_tokens_to_ids('<s>')] * batch_size,dtype=torch.long,device=device).view(-1,1)
      for _e in range(length):     
        # print(self.dec_layer.embeddings) 
        output=self.dec_layer(y,all_layer_hidden)
        # _,output=self.dec_layer(y,encoder_hidden_states=enc_hidden,encoder_attention_mask=cn_attn_mask)
        # print('cn_en:{}{}'.format(output.shape,_.shape))
        transformed=self.copy(output[:,-1:,:])
        copy_scores=torch.bmm(enc_hidden,transformed.transpose(1,2)) 
        # copy_scores=torch.bmm(enc_hidden,transformed.unsqueeze(-1)) 
        gen_scores=self.linear(output[:,-1:,:])
        # pred[:,_e]=self.linear(output)
        # pred=self.linear(output)
        # y=torch.tensor(pred[:,_e],dtype=torch.long,device=device).view(-1,1)  
        y_pred=torch.tensor(torch.argmax(gen_scores,dim=-1),dtype=torch.long,device=device).view(-1,1)      
        y=torch.cat((y,y_pred),dim=-1)
        yield gen_scores,copy_scores
def token_to_string(id):
  for d in dicts:
    if(id in d):
      return d[id]
  return None 
import math
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 50):
        super(PositionalEncoder,self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model,requires_grad=False,dtype=torch.float16).to(device)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        self.register_buffer('pe',pe.unsqueeze(0))
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(float(self.d_model))
        #add positional embedding
        seq_len = x.size(1)
        # pe = torch.tensor(self.pe[:,:seq_len],requires_grad=False).to(device)
        pe = self.pe[:,:seq_len,:].clone().detach()
        return x + pe.expand_as(x)
def attention(q, k, v, d_k,mask=False,cover=None):
  scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
  scores = F.softmax(scores, dim=-1) 
  if mask:
    tensor_mask=torch.ones(scores.shape,dtype=torch.int8).to(device)
    for row in range(tensor_mask.size(-2)):
      for col in range(tensor_mask.size(-1)):
        if(row>col):
          tensor_mask[:,:,row,col]=0 
    scores=scores * tensor_mask       
    scores=scores.transpose(-2,-1)   
  if(not cover is None):
    tensor_mask=torch.ones([scores.size(0),scores.size(1),scores.size(-1),scores.size(-1)],\
                           dtype=torch.int8).to(device)
    for row in range(tensor_mask.size(-2)):
      for col in range(tensor_mask.size(-1)):
        if(row>=col):
          tensor_mask[:,:,row,col]=0 
    tensor_mask=tensor_mask.transpose(-2,-1) 
    cv=torch.matmul(scores,tensor_mask)
    cover+=torch.sum(torch.min(cv,scores),[0,1,2,3])     
    # print(cover)     
  output = torch.matmul(scores, v)
  return output
class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_size,heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads=heads 
        self.d_k = hidden_size // heads
        self.hidden_size=hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size,bias=False)
        self.v_linear = nn.Linear(hidden_size, hidden_size,bias=False)
        self.k_linear = nn.Linear(hidden_size, hidden_size,bias=False)        
    
    def forward(self, q, k, v,mask=False,cover=None):        
        bs = q.size(0)
        k = self.k_linear(k).view(bs,-1,self.heads,self.d_k)
        q = self.q_linear(q).view(bs,-1,self.heads,self.d_k)
        v = self.v_linear(v).view(bs,-1,self.heads,self.d_k)

        k=k.transpose(1,2)
        q=q.transpose(1,2)
        v=v.transpose(1,2)

        scores = attention(q, k, v, self.d_k,mask,cover)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.hidden_size)
        return concat
class Norm(nn.Module):
  def __init__(self,hidden_size):
    super(Norm, self).__init__()
    self.norm=nn.LayerNorm(hidden_size,elementwise_affine=False)
  def forward(self,x,res):
    x=x+self.norm(res)
    return x
class FeedForward(nn.Module):
  def __init__(self,hidden_size,d_ff=2048):
    super(FeedForward, self).__init__()
    self.linear_1=nn.Linear(hidden_size,d_ff)
    self.linear_2=nn.Linear(d_ff,hidden_size)
  def forward(self,x):
    output=self.linear_1(x)
    output=self.linear_2(output)
    return output
class Decoder(nn.Module):
  def __init__(self,hidden_size,vocab_size):
    super(Decoder, self).__init__()
    self.hidden_size=hidden_size
    self.pos=PositionalEncoder(hidden_size)
    self.embedding=nn.Embedding(vocab_size,hidden_size,padding_idx=0)
    self.embedding.weight.requires_grad = False
    self.attn1_layer_list=nn.ModuleList([])
    self.attn2_layer_list=nn.ModuleList([])
    self.norm_layer_list=nn.ModuleList([])
    self.ff_layer_list=nn.ModuleList([])
    for indx in range(6):
      self.attn1_layer_list.append(MultiHeadAttention(hidden_size))    
    for indx in range(6):
      self.attn2_layer_list.append(MultiHeadAttention(hidden_size))
    for indx in range(6):
      self.norm_layer_list.append(Norm(hidden_size))
    for indx in range(6):
      self.ff_layer_list.append(FeedForward(hidden_size)) 
  def load_my_state_dict(self, state_dict):
      own_state = self.state_dict()
      for name, param in state_dict.items():          
          if name !='embeddings.word_embeddings.weight':
                continue                              
          # own_state['embedding.weight'][:param.size(0),:].copy_(param)
          own_state['embedding.weight'][:param.size(0),:]=param                 
  def forward(self,seq,all_hidden_layer,coverage=None):
    embedded=self.embedding(seq)
    embedded=self.pos(embedded)

    # output=self.attn_1(embedded,embedded,embedded,True)
    for indx,layer in enumerate(all_hidden_layer):      
      dec_hidden=self.attn1_layer_list[indx](embedded,embedded,embedded,True)
      dec_hidden=self.norm_layer_list[indx](embedded,dec_hidden)
      res=self.attn2_layer_list[indx](dec_hidden,layer,layer,cover=coverage)
      output=self.norm_layer_list[indx](embedded,res)
      res=self.ff_layer_list[indx](output)
      dec_hidden=self.norm_layer_list[indx](output,res)
      embedded=torch.clone(dec_hidden)
    return dec_hidden
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BERT_en_encoder=BertModel.from_pretrained('bert-base-uncased',\
                                        output_hidden_states=True,num_hidden_layers=5)
BERT_cn_encoder=BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",\
                                        output_hidden_states=True,num_hidden_layers=5)
model_en_cn=Translator_EN_CN().to(device)
model_en_cn.load_my_state_dict(torch.load('/content/drive/My Drive/translator_en_cn.pt'))
model_cn_en=Translator_CN_EN().to(device)
model_cn_en.load_my_state_dict(torch.load('/content/drive/My Drive/translator_cn_en.pt'))

optimizer = optim.Adam(list(model_en_cn.parameters())+list(model_cn_en.parameters()) , lr = 1e-5)
val_losses=[]
en_id_str_dict={value:key for key,value in en_tokenizer.vocab.items()}
cn_id_str_dict={value:key for key,value in cn_tokenizer.vocab.items()}
for iter in range(100):
  print('=============Iteration {}============='.format(iter))
  train_loss=0
  for i,batch in enumerate(train_loader):
    loss=0
    optimizer.zero_grad()
    cn_encoding=torch.tensor(batch['x']).to(device)
    cn_attn_mask=torch.tensor(batch['attn_mask']).to(device)
    # print('original cn :',batch['src'])     
    
    # CN-EN translation          

    vocab_en={}
    indx=0
    for tokens in batch['src']:
      for token in tokens:
        if(token in vocab_en or token in en_tokenizer.vocab):
          continue
        vocab_en.update({token:indx})
        indx+=1 
    vocab_en={key:value+len(en_tokenizer) for key,value in vocab_en.items()}
    vocab_en_id_str={value:key for key,value in vocab_en.items()}  
    
    expanded_x_en=[]
    copy_mask_en=[]
    for tokens in batch['src']:
      expanded_x_en.append([vocab_en[token]  if token in vocab_en\
                      else en_tokenizer.convert_tokens_to_ids(token) for token in tokens])      
      copy_mask_en.append([1 if isEnglish(token) else 0 for token in tokens])   
    expanded_x_en = [pad(item,en=True) for item in expanded_x_en]
    copy_mask_en = [pad(item,en=False) for item in copy_mask_en]
    expanded_x_en=torch.tensor(expanded_x_en,device=device)
    copy_mask_en=torch.tensor(copy_mask_en,device=device)

    pred=[]
    for i,(gen_scores,copy_scores) in enumerate(model_cn_en(cn_encoding,cn_attn_mask)):
      # expand gen scores to cover extra vocab
      # print(gen_scores.shape,len(vocab_cn))
      gen_scores=F.pad(gen_scores,(0,len(vocab_en)),'constant')             
      gen_scores.squeeze(1).scatter_add_(1,expanded_x_en,copy_scores.squeeze(-1)\
                                          [:,:expanded_x_en.size(1)]*copy_mask_en)
      # gen_scores.scatter_add_(1,expanded_x_en,copy_scores.view(1,-1)[:,:expanded_x_en.size(1)]*copy_mask_en)
      pred.extend([torch.argmax(gen_scores,dim=-1).tolist()]) 
      # pred.extend([torch.argmax(gen_scores,dim=-1).tolist()])   
    pred=[list(itm) for itm in zip(*pred)]    
    dicts=[vocab_en_id_str,en_id_str_dict]   
    # pred=list(map(list, zip(*pred)))    
    pred=[list(map(token_to_string,[itm[0] for itm in batch])) for batch in pred]
    
    # print('english translation:',pred)

    # create extra vocab
    vocab={}
    indx=len(cn_tokenizer) 
    for tokens in pred:
      for token in tokens:
        if(token in vocab):
          continue
        vocab.update({token:indx})
        indx+=1          
    vocab_id_str={value:key for key,value in vocab.items()} 
    expanded_x_cn=[]
    for tokens in pred:
      expanded_x_cn.append([vocab[token] for token in tokens])      

    # expanded_x=[pad(item) for item in expanded_x]
    expanded_x_cn=torch.tensor(expanded_x_cn,device=device)

    # encode target with expanded vocab
    target=[]
    for tokens in batch['src']:
      target.append([vocab[token] if token in vocab else \
                     cn_tokenizer.convert_tokens_to_ids(token) for token in tokens]\
                    +[cn_tokenizer.convert_tokens_to_ids('</s>')])
    target=[pad(item,en=False) for item in target]
    target=torch.tensor(target,device=device)    
    # if(target.size(1)<100):
    #   target=F.pad(target,(0,100-target.size(1)),'constant')

    # EN-CN translation
    # expanded_x=[]
    en_encoding=[]
    en_attn_mask=[]
    # cn_mask=[]
    for tokens in pred:
      # cn_mask.append([1 if token in vocab else 0 for token in tokens])
      # expanded_x.append([vocab[token] if token in vocab else cn_tokenizer.convert_tokens_to_ids(token)\
      #                    for token in tokens])     
      en_encoding.append([en_tokenizer.convert_tokens_to_ids(token) for token in tokens])
      en_attn_mask.append([1 if e!=en_tokenizer.convert_tokens_to_ids('[PAD]') else 0 for e in en_encoding[-1]])

    # expanded_x=torch.tensor(expanded_x,device=device)
    en_encoding=torch.tensor(en_encoding,device=device)
    en_attn_mask=torch.tensor(en_attn_mask,device=device)
    # en_mask=torch.tensor(en_mask,device=device)
    pred=[]
    # print(cn_encoding.shape,cn_attn_mask.shape)
    for i,(gen_scores,copy_scores) in enumerate(model_en_cn(en_encoding,en_attn_mask,target)):
      # expand gen scores to cover extra vocab
      gen_scores=F.pad(gen_scores,(0,len(vocab)),'constant')      
      gen_scores.squeeze(1).scatter_add_(1,expanded_x_cn,copy_scores.squeeze(-1)\
                                          [:,:expanded_x_cn.size(1)]) 
      # print(gen_scores[:,:,0])
      # gen_scores.scatter_add_(1,expanded_x_cn,copy_scores.view(1,-1)[:,:expanded_x_cn.size(1)])     
      loss+=F.cross_entropy(gen_scores.permute(0,2,1),target[:,i].view(-1,1),\
                            ignore_index=cn_tokenizer.convert_tokens_to_ids('[PAD]')) 
      # loss+=F.cross_entropy(gen_scores.unsqueeze(-1),target[:,i].view(-1,1),\
                            # ignore_index=cn_tokenizer.convert_tokens_to_ids('[PAD]'))                                      
      pred.extend([torch.argmax(gen_scores,dim=-1).tolist()])      
      # pred.extend([torch.argmax(gen_scores,dim=-1).tolist()])
    pred=[list(itm) for itm in zip(*pred)]
    dicts=[vocab_id_str,cn_id_str_dict]   
    # pred=list(map(list, zip(*pred)))    
    pred=[list(map(token_to_string,[itm[0] for itm in batch])) for batch in pred] 
    # print('chinese back:',pred)
    loss.backward() 
    optimizer.step()
    train_loss+=loss.data.item()
  print('train loss:{}'.format(train_loss))

  val_loss=0
  with torch.no_grad():
    for i,batch in enumerate(val_loader):
      loss=0
      optimizer.zero_grad()
      cn_encoding=torch.tensor(batch['x']).to(device)
      cn_attn_mask=torch.tensor(batch['attn_mask']).to(device)
      # print('original cn :',batch['src'])     
      
      # CN-EN translation          

      vocab_en={}
      indx=0
      for tokens in batch['src']:
        for token in tokens:
          if(token in vocab_en or token in en_tokenizer.vocab):
            continue
          vocab_en.update({token:indx})
          indx+=1 
      vocab_en={key:value+len(en_tokenizer) for key,value in vocab_en.items()}
      vocab_en_id_str={value:key for key,value in vocab_en.items()}  
      
      expanded_x_en=[]
      copy_mask_en=[]
      for tokens in batch['src']:
        expanded_x_en.append([vocab_en[token]  if token in vocab_en\
                        else en_tokenizer.convert_tokens_to_ids(token) for token in tokens])      
        copy_mask_en.append([1 if isEnglish(token) else 0 for token in tokens])   
      expanded_x_en = [pad(item,en=True) for item in expanded_x_en]
      copy_mask_en = [pad(item,en=False) for item in copy_mask_en]      
      expanded_x_en=torch.tensor(expanded_x_en,device=device)
      copy_mask_en=torch.tensor(copy_mask_en,device=device)

      pred=[]
      for i,(gen_scores,copy_scores) in enumerate(model_cn_en(cn_encoding,cn_attn_mask)):
        # expand gen scores to cover extra vocab
        # print(gen_scores.shape,len(vocab_cn))
        gen_scores=F.pad(gen_scores,(0,len(vocab_en)),'constant')           
        gen_scores.squeeze(1).scatter_add_(1,expanded_x_en,copy_scores.squeeze(-1)\
                                            [:,:expanded_x_en.size(1)]*copy_mask_en)
        # gen_scores.scatter_add_(1,expanded_x_en,copy_scores.view(1,-1)[:,:expanded_x_en.size(1)]*copy_mask_en)
        pred.extend([torch.argmax(gen_scores,dim=-1).tolist()]) 
        # pred.extend([torch.argmax(gen_scores,dim=-1).tolist()])  
      pred=[list(itm) for itm in zip(*pred)]  
      dicts=[vocab_en_id_str,en_id_str_dict]   
      # pred=list(map(list, zip(*pred)))    
      pred=[list(map(token_to_string,[itm[0] for itm in batch])) for batch in pred]
      
      # print('english translation:',pred)

      # create extra vocab
      vocab={}
      indx=len(cn_tokenizer) 
      for tokens in pred:
        for token in tokens:
          if(token in vocab):
            continue
          vocab.update({token:indx})
          indx+=1          
      vocab_id_str={value:key for key,value in vocab.items()} 
      expanded_x_cn=[]
      for tokens in pred:
        expanded_x_cn.append([vocab[token] for token in tokens])      

      # expanded_x=[pad(item) for item in expanded_x]
      expanded_x_cn=torch.tensor(expanded_x_cn,device=device)

      # encode target with expanded vocab
      target=[]
      for tokens in batch['src']:
        target.append([vocab[token] if token in vocab else \
                      cn_tokenizer.convert_tokens_to_ids(token) for token in tokens]\
                      +[cn_tokenizer.convert_tokens_to_ids('</s>')])
      target=[pad(item,en=False) for item in target]
      target=torch.tensor(target,dtype=torch.long,device=device)
      # if(target.size(1)<100):
      #   target=F.pad(target,(0,100-target.size(1)),'constant')

      # EN-CN translation
      # expanded_x=[]
      en_encoding=[]
      en_attn_mask=[]
      # cn_mask=[]
      for tokens in pred:
        # cn_mask.append([1 if token in vocab else 0 for token in tokens])
        # expanded_x.append([vocab[token] if token in vocab else cn_tokenizer.convert_tokens_to_ids(token)\
        #                    for token in tokens])     
        en_encoding.append([en_tokenizer.convert_tokens_to_ids(token) for token in tokens])
        en_attn_mask.append([1 if e!=en_tokenizer.convert_tokens_to_ids('[PAD]') else 0 for e in en_encoding[-1]])

      # expanded_x=torch.tensor(expanded_x,device=device)
      en_encoding=torch.tensor(en_encoding,device=device)
      en_attn_mask=torch.tensor(en_attn_mask,device=device)
      # en_mask=torch.tensor(en_mask,device=device)
      pred=[]
      # print(cn_encoding.shape,cn_attn_mask.shape)
      for i,(gen_scores,copy_scores) in enumerate(model_en_cn(en_encoding,en_attn_mask,target)):
        # expand gen scores to cover extra vocab
        gen_scores=F.pad(gen_scores,(0,len(vocab)),'constant')      
        gen_scores.squeeze(1).scatter_add_(1,expanded_x_cn,copy_scores.squeeze(-1)\
                                            [:,:expanded_x_cn.size(1)]) 
        # print(gen_scores[:,:,0])
        # gen_scores.scatter_add_(1,expanded_x_cn,copy_scores.view(1,-1)[:,:expanded_x_cn.size(1)])     
        loss+=F.cross_entropy(gen_scores.permute(0,2,1),target[:,i].view(-1,1),\
                              ignore_index=cn_tokenizer.convert_tokens_to_ids('[PAD]')) 
        # loss+=F.cross_entropy(gen_scores.unsqueeze(-1),target[:,i].view(-1,1),\
                              # ignore_index=cn_tokenizer.convert_tokens_to_ids('[PAD]'))                                      
        pred.extend([torch.argmax(gen_scores,dim=-1).tolist()])      
        # pred.extend([torch.argmax(gen_scores,dim=-1).tolist()])
      pred=[list(itm) for itm in zip(*pred)] 
      dicts=[vocab_id_str,cn_id_str_dict]   
      # pred=list(map(list, zip(*pred)))    
      pred=[list(map(token_to_string,[itm[0] for itm in batch])) for batch in pred] 
      # print('chinese back:',pred)
      val_loss+=loss.data.item()
    print('validation loss:{}'.format(val_loss))
    if(len(val_losses)>0 and val_loss<min(val_losses)):
      torch.save(model_en_cn.state_dict(), '/content/drive/My Drive/translator_en_cn.pt')
      torch.save(model_cn_en.state_dict(), '/content/drive/My Drive/translator_cn_en.pt')
    val_losses.append(val_loss)