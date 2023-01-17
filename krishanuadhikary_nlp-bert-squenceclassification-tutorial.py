# Importing Libraries
import os
import pandas as pd
import string,re
from urllib.parse import urlparse
import spacy
import itertools
from torch import nn
from torchtext import data  
from nltk.corpus import stopwords 
import torch
from torchtext.data import BucketIterator
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt
from torch.backends import cudnn
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cudnn.benchmark = True
device
#DataFrame Options for display
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_colwidth', -1)  
# loading spacy model
nlp = spacy.load('en_core_web_sm') 
# Dataset
path  = r'../input/nlp-getting-started/'
trainDataset = pd.read_csv(os.path.join(path,'train.csv'),index_col=0)
trainDataset.head(5)
# The keyword/location column is inconsistent
trainDataset = trainDataset[['text','target']]
trainDataset.head()
from unidecode import unidecode
def remove_non_ascii(text):
    return unidecode(text)
def clean(text):
    # removing the spaces
    text = ' '.join(text.split()) 
    # removing url 
    text = ' '.join([token for token in text.split() if not urlparse(token).scheme]) 
    # removing token starts with @
    text = ' '.join([token for token in text.split() if not (re.match(r'^@',token))])
    # remove all punctuation ( except . and ,)
    text = text.translate(str.maketrans('', '','!"$%&\'()*+-/:;<=>?@[\\]^_`{|}~')) 
    # keep only alphabets
    text = re.sub(r"\d", "", text) 
    text = ' '.join(list(itertools.chain.from_iterable([[s for s in re.split("([A-Z][^A-Z]*)", token.replace('#','')) if s] if (re.match(r'^#',token)) else [token] for token in text.split()])))
    # removing token less than 3 char length and removing non-ascii character
    text = ' '.join([remove_non_ascii(token) for token in text.split()]) 
    text = text.lower().strip()
    return text
text = '''#raining #flooding #Florida #TampaBay #Tampa 18 or 19 days. I've lost count'''   
flatten = [[s for s in re.split("([A-Z][^A-Z]*)", token.replace('#','')) if s] if (re.match(r'^#',token)) else [token] for token in text.split()] # removing token starts with @
flattened = list(itertools.chain(*flatten))
' '.join(flattened)
trainDataset['text'] = trainDataset['text'].apply(lambda x:clean(x))
trainDataset.head()
len(trainDataset)
trainDataset = trainDataset[trainDataset['text']!='']
trainDataset.shape
trainDataset = trainDataset.drop_duplicates(subset ="text", 
                     keep = False) 
trainDataset.shape
# The classes are two 0 and 1
trainDataset.to_csv(os.path.join(os.getcwd(),'input.csv'),index=False)
trainDataset = pd.read_csv(os.path.join(os.getcwd(),'input.csv'))
trainDataset.head()
# Stopwords = spacy stopword + nltk stopword
nlp.Defaults.stop_words|= set(stopwords.words('english'))
nlp.Defaults.stop_words|= {'the',}

#list(nlp.Defaults.stop_words)
len(nlp.Defaults.stop_words)
# Bert Tokenizer
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
# Pipelines used for Text Field
pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
maxSequenceLength = 64 # not used
TEXT = data.Field(use_vocab=False,tokenize=tokenizer.encode,batch_first=True,include_lengths=True,pad_token=pad_index,unk_token=unk_index)
# Pipelines used for LabelField Field
LABEL = data.LabelField(dtype = torch.long,batch_first=True)
fields = [('text',TEXT),('target', LABEL)]
# Reading and transforming the data
training_data=data.TabularDataset(path=os.path.join(os.getcwd(),'input.csv'),format = 'csv',fields = fields,skip_header = True)
vars(training_data.examples[5])
# This will be used to match the results
# Sequential is False to treat it as a number and no one hot encode
ID = data.Field(sequential=False,dtype = torch.int,batch_first=True,use_vocab=False)
# Reading and transforming the test data
test_data = pd.read_csv(os.path.join(path,'test.csv'),index_col=0)
test_data.head()
test_data = test_data[['text']]
test_data.head()
test_data['text'] = test_data['text'].apply(lambda x:clean(x))
test_data.head()
test_data.to_csv(os.path.join(os.getcwd(),'test.csv'))
test_data = data.TabularDataset(path=os.path.join(os.getcwd(),'test.csv'),format = 'csv',fields =[('id', ID),('text',TEXT)],skip_header = True)
vars(test_data.examples[2])
# This is replaced by bert tokenizer 
#TEXT.build_vocab(training_data,min_freq=3,vectors = "glove.6B.100d")  
LABEL.build_vocab(training_data,)
#No. of unique tokens in text
#print("Size of TEXT vocabulary:",len(TEXT.vocab))
#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))
# Bert tokenizer numerical to string
tokenizer.decode([101,
  2045,
  3224,
  2543,
  3962,
  8644,
  28519,
  2024,
  14070,
  2408,
  1996,
  2395,
  3685,
  3828,
  2068,
  2035,
  102])
# Bert tokenizer string to label
tokenizer.encode(['there','forest','fire'])

#set batch size
BATCH_SIZE = 128
# All same size text grouped together
#Load an iterator
train_iterator = data.BucketIterator(
    training_data, 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True,
    device = device)
# Each batch has 128 texts 
#vars(next(iter(train_iterator)))
# Iteration loop
for (trainX,Length),trainY in train_iterator:
    print(trainX.shape,trainY)
    bertTrainX = trainX
    bertTrainY = trainY
    break
# All same size text grouped together
#Load an iterator
test_iterator = data.BucketIterator(
    test_data, 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True,
    device = device)
#vars(next(iter(test_iterator)))
for (trainID,(trainX,Length)),trainY in test_iterator:
    print(trainID,trainX.shape,Length,trainY)
    break
# BERT BASE MODEL
bertLayer = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased')  # Update configuration during loading
# BERT BASE MODEL
class BertModel(nn.Module):
    def __init__(self,):
        super(BertModel,self).__init__()
        self.bertLayer = bertLayer
        
        # This step is important as it stops training the bert layers
        for name, param in self.bertLayer.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False   
            
    def forward(self,x,y):
        return self.bertLayer(x,labels=y)
# BERT BASE MODEL
epochs=10
losses=[]
# BERT BASE MODEL
modelObj = BertModel()
modelObj.to(device)
# BERT BASE MODEL
optimizer = torch.optim.Adam(modelObj.parameters(), lr=0.000001)
# BERT BASE MODEL
for (trainX,text_lengths),trainY in train_iterator:
    modelObj.eval()
    loss,ypred= modelObj(trainX,trainY)
    break
print(loss,ypred)
# Bert Model
for epoch in range(epochs):
    modelObj.train()
    for (trainX,text_lengths),trainY in train_iterator:
        loss,ypred= modelObj(trainX,trainY)
        loss.backward()
        with torch.no_grad():
            optimizer.step()
    losses.append(loss)
    print(f'Epoch {epoch} Loss is {loss}')
# Converting float for displaying PLOT
lossesCPU = [float(loss.to('cpu')) for loss in losses]
# Pyplot
plt.plot(lossesCPU)
submit_frame = pd.DataFrame(columns=['id','target'])
for (testID,(testX,text_lengths)),trainY in test_iterator:
    modelObj.eval()
    output = modelObj(testX,trainY)[0]
    submit_frame_temp = pd.DataFrame(columns=['id','target'])
    submit_frame_temp['id'] = testID.to('cpu').numpy()
    submit_frame_temp['target'] = torch.argmax(output,dim=1).to('cpu').numpy()
    submit_frame = submit_frame.append(submit_frame_temp)
submit_frame = submit_frame.sort_values(by=['id'],ascending=True).reset_index(drop=True)
submit_frame.to_csv(os.path.join(os.getcwd(),'submission.csv'),index=False)
os.getcwd()
submit_frame.head()