import pandas as pd
import re
import string
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from tqdm import tqdm

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import random

from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
stop=set(stopwords.words('english'))
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
tweet.head(3)
df=pd.concat([tweet,test])
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

remove_URL(example)
df['text']=df['text'].apply(lambda x : remove_URL(x))
example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
print(remove_html(example))
df['text']=df['text'].apply(lambda x : remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

remove_emoji("Omg another Earthquake 😔😔")
df['text']=df['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

example="I am a #king"
print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus
corpus=create_corpus(df)
for i in range(0, len(corpus)):
    
    corpus[i] = " ".join(corpus[i])
df["text"] = corpus
train_df = df[0:7613]
test_df = df[7613:]
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence padding
def padding_zero(vec, max_length):
    if len(vec) < max_length:
        vec = np.pad(vec, ( (0, max_length-len(vec)), (0,0) ), 'constant')
    elif len(vec) > max_length:
        vec = vec[:max_length]

    return vec
text_data = list(train_df["text"].values)
train_lst = list()

for i in tqdm(range(0, len(text_data))):
    input_ids = torch.tensor(tokenizer.encode(text_data[i], add_special_tokens=True, max_length=100)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0][0]
    bert_arr = padding_zero(last_hidden_states.detach().numpy(), 10)
    train_lst += [bert_arr]
train_arr = np.array(train_lst)
target_lst= list(train_df["target"].values)
len(train_arr), len(target_lst)
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data, target):
        
        self.data = data
        self.target = target
        self.length = len(self.data)
        
    def __len__(self):
        
        return self.length
    
    def __getitem__(self, index): 
        
        ret = self.data[index]
        target = self.target[index]
    
        return ret, target
trainset = MyDataSet(train_arr, target_lst)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
for data in trainloader:
    print(data[0].shape, data[1].shape)
    break
# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        encoder = nn.TransformerEncoderLayer(768, 32, dim_feedforward=512) # 次元数, Attentionの数, # feed_forward
        self.encoder = nn.TransformerEncoder(encoder, 4)
        
        self.pool = nn.MaxPool2d((128, 1))
        self.linear = nn.Linear(7680, 2)
    
    def forward(self, s):
        b = s.shape[0]
        
        s = s.permute(1, 0, 2)
        s = self.encoder(s)
        s = s.permute(1, 0, 2)
#         s = self.pool(s)
        s = s.reshape(b, -1)
        s = self.linear(s)
        s = torch.log_softmax(s, dim=1)
        
        return s
transModel = Model()
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(transModel.parameters(), lr=0.0001)
for data in trainloader:
    res = transModel(data[0])
    print(res.shape)
    break
for epoch in tqdm(range(10)):
    print(epoch)
    
    train_loss = 0    
    model.train()
    
    for data in trainloader:
        
        x, t = data[0], data[1].long()
        
        out = transModel(x)
        
        loss = criterion(out, t)
        train_loss += loss.item()
    
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"{train_loss/len(trainloader)}")
text_data = list(test_df["text"].values)
test_lst = list()

for i in tqdm(range(0, len(text_data))):
    input_ids = torch.tensor(tokenizer.encode(text_data[i], add_special_tokens=True, max_length=100)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0][0]
    bert_arr = padding_zero(last_hidden_states.detach().numpy(), 10)
    test_lst += [bert_arr]
test_arr = np.array(test_lst)
transModel.eval()
res_lst = list()
for i in tqdm(range(0, len(test_arr))):
    res_lst.append(np.exp(transModel(torch.FloatTensor(test_arr[i]).unsqueeze(-1).permute(2, 0, 1)).detach().numpy())[0])
fin_res_lst = list()
for i in range(0, len(res_lst)):
    
    if res_lst[i][0] > res_lst[i][1]:
        fin_res_lst.append(0)
    else:
        fin_res_lst.append(1)
sample_df = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sample_df["target"] = fin_res_lst
sample_df.to_csv('submission.csv',index=False)
