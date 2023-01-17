# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from wordcloud import WordCloud,STOPWORDS

import re

import operator

import warnings

warnings.filterwarnings('ignore')





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

EMBED_PATH='../input/glove-reddit-comments/GloVeReddit120B/GloVe.Reddit.120B.300D.txt'

MAX_LEN=300

MAX_FEATURES=50000

BATCH_SIZE=32

NUM_EPOCHS=10
import spacy

from sklearn import model_selection

from tqdm import tqdm

from html import unescape

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from nltk.tokenize import word_tokenize,wordpunct_tokenize,regexp_tokenize

from sklearn import metrics

import random

tqdm.pandas()



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader,Dataset

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.autograd import Variable

import torch.nn.functional as F

def seed_torch(seed=45):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
#Read the data,

df=pd.read_csv("../input/60k-stack-overflow-questions-with-quality-rate/data.csv")
df.head()
print(f'Number of rows:{df.shape[0]} and Number of columns:{df.shape[1]}')
df.isnull().sum() # no null values in any column
df['Y'].value_counts()
hq_data=df.loc[df['Y']=='HQ',]['Body'].values
hq_data[0:4]
lq_data=df.loc[df['Y']=='LQ_EDIT',]['Body'].values
lq_data[0:4]
lq_close=df.loc[df['Y']=='LQ_CLOSE',]['Body'].values
lq_close[0:4]
#REFERENCE https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    #create a set of common html tags,

    html_tags={'<p>','</p>','<code>','</code>','<pre>','</pre>','<div>','</div>','<br/>','<title>','</title>','<body>','</body>','title'}

    stopwords.union(html_tags)

    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(df.loc[df['Y']=='HQ',]['Body'], title="Questions in HQ category")
plot_wordcloud(df.loc[df['Y']=='LQ_EDIT',]['Body'], title="Questions in LQ_EDIT category")
plot_wordcloud(df.loc[df['Y']=='LQ_CLOSE',]['Body'], title="Questions in LQ_CLOSE category")
print(f'''Average length of the question body for category HQ {np.mean(df.loc[df['Y']=='HQ',]['Body'].apply(lambda x:len(x.split())))}''')

print(f'''Average length of the question body for category LQ_EDIT {np.mean(df.loc[df['Y']=='LQ_EDIT',]['Body'].apply(lambda x:len(x.split())))}''')

print(f'''Average length of the question body for category LQ_CLOSE {np.mean(df.loc[df['Y']=='LQ_CLOSE',]['Body'].apply(lambda x:len(x.split())))}''')
##mapping the targets as integer values for modelling,

target_dict={'HQ':0,'LQ_EDIT':1,'LQ_CLOSE':2}



df.loc[:,'Y']=df['Y'].map(target_dict)
## Split the data into train and validation set:

### reference : Abhiskek Thakur - https://www.youtube.com/user/abhisheksvnit

df['kfold']=-1

kf=model_selection.KFold(n_splits=5,random_state=40,shuffle=True)

for fold_,(trn_,val_) in enumerate(kf.split(df)):

    print(f'Fold {fold_} Training {len(trn_)} Validation {len(val_)}')

    print("")

    df.loc[val_,'kfold']=fold_

    print(f'Split of categories for fold {fold_}')

    print(df.loc[df['kfold']==fold_,'Y'].value_counts())

    print("")
spacy.load('en')

lemma=spacy.lang.en.English()
def clean_text(text):

    text=str(text)

    #cleaning URLs

    text=re.sub(r'https?://\S+|www\.\S+', '', text)

    #cleaning html elements,

    text=re.sub(r'<.*?>', '', text)

    #replace carriage return with space

    text=text.replace("\n"," ").replace("\r"," ")

    #replace punctuations with space,

    punct='?!.,"#$%\'()*+-/:;=@[\\]^_`{|}~<>&'

    for p in punct:

        text=text.replace(p," ")

    #replace single quote with empty character,

    text=text.replace("'`",'')

    return text


def my_tokenizer(doc):

    tokens = lemma(doc)

    return([token.lemma_ for token in tokens])



def reg_tokenize(doc):

    doc=clean_text(doc)

    doc=regexp_tokenize(doc,pattern='\s+',gaps=True) # gaps=True,since we want to find separators between tokens.

    return doc
sample=list(df['Body'])[0:3]

count_vect=CountVectorizer(tokenizer=reg_tokenize,

                           token_pattern=None,

                           ngram_range=(1,1),

                           stop_words='english')

count_vect.fit(sample)

print(sample)

print("\n\n")

print(count_vect.vocabulary_)

print("")

print(len(count_vect.vocabulary_))
sample=list(df['Body'])[0:3]

count_vect=CountVectorizer(tokenizer=my_tokenizer,

                           token_pattern=None,

                           ngram_range=(1,1),

                           stop_words='english')

count_vect.fit(sample)

print(sample)

print("\n\n")

print(count_vect.vocabulary_)

print("")

print(len(count_vect.vocabulary_))
##Reference :sklearn documentation

#https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

losses=[]

for fold_ in range(5):

    #initialize train data,

    x_train=df[df.kfold!=fold_].reset_index(drop=True)

    #initialize valid data,

    x_valid=df[df.kfold==fold_].reset_index(drop=True)

    #print(f'Shape of x_train {x_train.shape} Shape of x_valid {x_valid.shape}')

    #initialize count vectorizer,

    count_vect=CountVectorizer(tokenizer=reg_tokenize,

                           token_pattern=None,

                           ngram_range=(1,1),

                           stop_words='english')

    #print("Fitting the count vectorizer")

    count_vect.fit(x_train['Body'])

    

    x_train_counts=count_vect.transform(x_train['Body'])

    

    x_valid_counts=count_vect.transform(x_valid['Body'])

    #print(type(x_train_counts),type(x_valid_counts))

    #transform train and test question body,

    

    

    clf=MultinomialNB()

    

    clf.fit(x_train_counts,x_train['Y'].values)

    

    predicted_class=clf.predict_proba(x_valid_counts)

    

    loss=metrics.log_loss(x_valid['Y'].values,predicted_class)

    losses.append(loss)

    

    print(f'Fold {fold_} Log loss {loss}')



print("Average Log Loss",np.mean(losses))
#Reference https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

def build_vocab(sentences,verbose=True):

    #initialize empty dictionary

    vocab={}

    for sentence in tqdm(sentences): ## loop over each sentence

        for word in sentence: #for each word in the sentence create a vocab id

            try:

                vocab[word]+=1

            except KeyError:

                vocab[word]=1

    return vocab
sentences=df['Body'].progress_apply(lambda x:x.split()).values

vocab=build_vocab(sentences)

print({k:vocab[k] for k in list(vocab)[:5]})
#import embeddings :

#Reference:https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# embeddings_index={}

# f=open(EMBED_PATH)

# for line in tqdm(f):

#     values=line.split()

#     word=values[0]

#     coefs=np.asarray(values[1:],dtype='float32')

#     embeddings_index[word]=coefs

# f.close()

    
def get_coefs(word,*arr): return word,np.array(arr,dtype='float32')



def load_embeddings(EMBED_PATH):

    embedding_index=dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(EMBED_PATH)))

    return embedding_index
embedding_index=load_embeddings(EMBED_PATH)
def check_coverage(vocab,embedding_index):

    a={}

    oov={}

    k=0

    i=0

    for word in tqdm(vocab):

        #print("\nWord",word)

        #print("\nVocab Word",vocab[word])

        try:

            a[word]=embedding_index[word] ## check if the word is present in the embedding matrix

            k+=vocab[word]

            #print("\n K value",k)

        except:

            oov[word]=vocab[word]

            i+=vocab[word]

            #print("\n i value",i)

            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sortedx=sorted(oov.items(),key=operator.itemgetter(1))[::-1]

    

    return sortedx
oov=check_coverage(vocab,embedding_index)
oov[:10]
df['Body']=df['Body'].apply(lambda x:clean_text(x))

sentences=df['Body'].progress_apply(lambda x:x.split()).values

vocab=build_vocab(sentences)

oov=check_coverage(vocab,embedding_index)
oov[:10]
def clean_char_num(text):

    text=re.sub('[0-9]{5,}','#####',text)

    text=re.sub('[0-9]{4}','####',text)

    text=re.sub('[0-9]{3}','###',text)

    text=re.sub('[0-9]{2}','##',text)

    #replace same repeating chacters with one eg.yyyyy to y

    text=re.sub(r'[^\w\s]|(.)(?=\1)', '', text)

    return text
df['Body']=df['Body'].apply(lambda x:clean_char_num(x))

sentences=df['Body'].progress_apply(lambda x:x.split()).values

vocab=build_vocab(sentences)

oov=check_coverage(vocab,embedding_index)
oov[:100]
word_mapping={'Eror':'Error',

        'ArayList':'Arraylist',

        'Scaner':'Scanner',

        'botstrap':'bootstrap',

        'sucesfuly':'successfully',

        'arays':'arrays',

        'bufer':'buffer',

        'calback':'callback',

        'ApCompatActivity':'app compact activity',

        'Pasword':'password',

        'inerHTML':'inner html',

        'clasName':'classname',

        'TypeEror':'type error',

        'maloc':'malloc',

        'ApData':'appdata',

        'foter':'footer',

        'Bolean':'boolean',

        'ThreadPolExecutor':'thread pool executor',

        'styleshet':'stylesheet',

        'tolbar':'toolbar',

        'Colections':'collections',

        '1px':'one pixel',

        'SESION':'session',

        'Arays':'arrays',

        'BuferedReader':'buffered reader',

        'getAplicationContext':'get application context',

        '0px':'pixel',

        'NulPointerException':'null pointer exception',

        'SqlComand':'sql command',

        'dispatchMesage':'dispatch message',

        'MesageBox':'messagebox',

        'DefaultBuildOperationExecutor':'default build operation executor',

        'MethodAndArgsCaler':'method and args caller',

        'AdWithValue':'ad with value',

        'notebok':'notebook',

        'debuger':'debugger',

        'hadop':'hadoop',

        'Fluter':'flutter',

        '5px':'pixel',

        'claspath':'classpath',

        'NativeMethodAcesorImpl':'native method accessor impl',

        'MyClas':'myclass',

        'iHealAp':'app',

        'AbstractAutowireCapableBeanFactory':'abstract autowire capable bean factory',

        'PASWORD':'password',

        'SqlConection':'sql connection',

        'MyAp':'my app',

        'SyntaxEror':'syntax error',

        'ClasLoader':'class loader',

        'SpringAplication':'spring application',

        'Tols':'Tools',

        'INER':'Inner',

        'Botstrap':'bootstrap',

        'adEventListener':'ad event lister',

        'ValueEror':'value error',

        'NString':'n string',

        'adreses':'addresses',

        'handleMesage':'handle message',

        'getMesage':'get message',

        'ViewControler':'view controller',

        'apcompat':'app compact',

        'Tolbar':'toolbar',

        'ArayAdapter':'array adapter',

        'JButon':'j button',

        'UIViewControler':'ui view controller',

        'Atempt':'attempt',

        'custable0':'table',

        'programaticaly':'programatically',

        'midleware':'middleware',

        'Extent1':'extent',

        'opensl':'open ssl',

        'myap':'myapp',

        'ApComponent':'app component',

        'AbstractBeanFactory':'abstract bean factory',

        'stder':'std error',

        'Acept':'accept',

        'buton1':'button',

        'wordpres':'word press',

        'Nulable':'nullable',

        'iphonesimulator':'iphone simulator',

        'NonNul':'non null',

        'DelegatingMethodAcesorImpl':'delegating method',

        'JSONAray':'json array',

        'acesing':'accessing',

        'lokup':'lookup',

        'nulptr':'null pointer',

        'HtpClient':'http client',

        'Loger':'logger',

        'ToInt':'to int',

        'Aplications':'applications',

        'acesible':'accessible',

        'ViewRotImpl':'view',

        'alocator':'allocator',

        'ContentValues':'content values',

        'Iluminate':'illuminate',

        'adClas':'add class',

        'asoc':'associate',

        'Runable':'runnable',

        '0xF':'F',

        'contentValues':'content values',

        'findviewbyid':'find view by id',

         'activitythread':'activity thread',

         'araylist':'array list',

       'oncreate':'on create',

     'getelementbyid':'get element by id',

     'savedinstancestate':'saved instance state',

     'setext':'text',

     'editext':'edit text',

     'mainactivity':'main activity',

     'getext':'get text',

    'getstring':'get string'}
known=[]

unknown=[]

for word,key in word_mapping.items():

    #print(word)

    sent=[k for k in key.split(" ")]

    for s in sent:

        if s in embeddings_index :

            #print("True")

            known.append(key)

        else:

            unknown.append(key)

print("Total words present in the embedding",len(known))
def replace_words(text,word_mapping):

    return ' '.join(word_mapping[t] if t in word_mapping else t for t in text.split(' '))
df['Body']=df['Body'].apply(lambda x:replace_words(x,word_mapping))

sentences=df['Body'].progress_apply(lambda x:x.split()).values

vocab=build_vocab(sentences)

oov=check_coverage(vocab,embedding_index)
oov[:10]
def create_embed_matrix(word_index,embedding_index):

    num_words=min(MAX_FEATURES,len(word_index))

    

    embed_matrix=np.random.normal(0,1,(num_words,MAX_LEN))

    

    for word,i in word_index.items():

        if i>=MAX_FEATURES:continue

        embedding_vector=embedding_index.get(word)

        if embedding_vector is not None:embed_matrix[i]=embedding_vector

    

    return embed_matrix

    
class SOF_DATASET:

    def __init__(self,Body,Y):

        self.Body=Body

        self.Y=Y

        

    def __len__(self):

        return len(self.Body)

    

    def __getitem__(self,item):

        body=self.Body[item,:]

        target=self.Y[item]

        

        return {

            "body":torch.tensor(body,dtype=torch.long),

            "target":torch.tensor(target,dtype=torch.float)

        }
class NNet(nn.Module):

    def __init__(self):

        super(NNet,self).__init__()

        

        hidden_size=128

        

        self.embedding=nn.Embedding(num_embeddings=MAX_FEATURES,embedding_dim=300) 

        

        self.embedding.weight=nn.Parameter(torch.tensor(embed_matrix,dtype=torch.float32))

        

        self.embedding.weight.requires_grad=False

        

        self.embedding_dropout=nn.Dropout2d(0.3)

        

        self.lstm=nn.LSTM(input_size=300,hidden_size=hidden_size,bidirectional=True,batch_first=True)

        

        self.linear=nn.Linear(512,16)

        

        self.relu=nn.ReLU()

        

        self.dropout=nn.Dropout(0.3)

        

        self.softmax=nn.Softmax()

        

        self.out=nn.Linear(16,3)

        

    def forward(self,x):

        x=self.embedding(x)

        

        #print(x.shape)

        

        #x=torch.squeeze(self.embedding_dropout(torch.unsqeeze(x,0))) ##https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch

        

        x,_=self.lstm(x)  #(output (#shape (seq_len,batch_size,num_dir*hidden_size)),h_n,c_n)

        

        avg_pool=torch.mean(x,1)

        

        max_pool,_=torch.max(x,1) #returns max_values,indices

        

        concat=torch.cat((avg_pool,max_pool),1)

        

        concat=self.relu(self.linear(concat))

        

        concat=self.dropout(concat)

        #https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net

        out=self.softmax(self.out(concat))

        

        #print(out.shape)

        #print(out)

        

        #out=torch.argmax(out,dim=1)

        

        #print(out)

        

        return out
tokenizer=Tokenizer(num_words=MAX_FEATURES)

    

tokenizer.fit_on_texts(df.Body.values.tolist())



embed_matrix=create_embed_matrix(tokenizer.word_index,embedding_index)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using {device}")

model=NNet().to(device)

optimizer=optim.Adam(model.parameters(),lr=3e-3)

scheduler=ReduceLROnPlateau(optimizer,verbose=False)
def loss_fn(outputs, targets):

    """

    Defining categorical cross-entropy loss function.

    #https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-pytorch/85165/3

    """

    

    return nn.NLLLoss()(torch.log(outputs), targets)





def accuracy(preds, y):

    """

    Returns accuracy per batch

    """



    ind= torch.argmax(preds,dim= 1)

    correct = (ind == y).float()

    acc = correct.sum()/float(len(correct))

    return acc
for fold_ in range(5):

    

    #initialize train data,

    df_train=df[df.kfold!=fold_].reset_index(drop=True)

    

    #initialize valid data,

    df_valid=df[df.kfold==fold_].reset_index(drop=True)

    

    #tokenize the sentences,

    xtrain=tokenizer.texts_to_sequences(df_train.Body.values)

    xvalid=tokenizer.texts_to_sequences(df_valid.Body.values)

    

    #pad the sequences,

    xtrain=pad_sequences(xtrain,maxlen=MAX_LEN)

    xvalid=pad_sequences(xvalid,maxlen=MAX_LEN)

    

    

    x_train=SOF_DATASET(Body=xtrain,Y=df_train['Y'])

    

    x_train_loader=torch.utils.data.DataLoader(x_train,batch_size=BATCH_SIZE,shuffle=False)

    

    x_valid=SOF_DATASET(Body=xvalid,Y=df_valid['Y'])

    

    x_valid_loader=torch.utils.data.DataLoader(x_valid,batch_size=BATCH_SIZE,shuffle=False)

    

    model.train()

    avg_train_loss=0.0

    avg_train_accuracy=0.0

    print(f"----Starting training fold {fold_+1}--------")

    total_step = len(x_train_loader)

    for epoch in range(NUM_EPOCHS):

    

        for i,data in enumerate(x_train_loader):



            body=data['body']



            targets=data['target']



            targets=Variable(targets).long()



            body=torch.tensor(body,dtype=torch.long).cuda()



            targets=torch.tensor(targets,dtype=torch.long).cuda()



            outputs=model(body)



            loss=loss_fn(outputs,targets)



            acc=accuracy(outputs,targets)



            optimizer.zero_grad()



            loss.backward()



            optimizer.step()



            scheduler.step(loss)

            

            if (i+1) % 500 == 0:

                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 

                       .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))





    model.eval()

    avg_val_loss=0.0

    avg_val_accuracy=0.0

    print(f"--------Starting Validation {fold_+1}------")

    for i,data in enumerate(x_valid_loader):

        epoch_loss=0

        epoch_acc=0

        body=data['body']



        targets=data['target']



        body=torch.tensor(body,dtype=torch.long).cuda()



        targets=torch.tensor(targets,dtype=torch.long).cuda()



        outputs=model(body)



        loss=loss_fn(outputs,targets)



        epoch_loss+=loss.item()



        acc=accuracy(outputs,targets)



        avg_val_loss+=loss.item()



        avg_val_accuracy+=acc.item()











    print(f"Fold {fold_} Validation NLLoss {avg_val_loss/len(x_valid_loader):.3f} Validation accuracy {(avg_val_accuracy/len(x_valid_loader))*100:.2f} %")





        

        

        

        

    

    

    