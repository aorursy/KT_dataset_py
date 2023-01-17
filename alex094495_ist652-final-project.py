#Load packages
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import defaultdict
from collections import  Counter
from matplotlib import pyplot as plt
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
df_train = pd.read_csv('../input/nlp-getting-started/train.csv', sep=',')
df_test = pd.read_csv('../input/nlp-getting-started/test.csv',sep=',')
df_train.shape

ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

train_orig = df_train 
train_orig[train_orig['id'].isin(ids_with_target_error)]

train_orig.at[train_orig['id'].isin(ids_with_target_error),'target'] = 0

train_orig[train_orig['id'].isin(ids_with_target_error)]

df_train = train_orig
#### Check missing values
df_train.isnull().sum()
### Check distribution of class labels. 
x = df_train.target.value_counts()
countplt = sns.barplot(x.index,x)
countplt.set_xticklabels(['0: Not Disaster (4342)', '1: Disaster (3271)'])
#plt.gca().set_ylabel('samples')
#Ref: https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
### Plot tweets distribution over locations, that are aggregated in to a country's level
train = df_train
train['location'].replace({'United States':'USA',
                           'New York':'USA',
                            "London":'UK',
                            "Los Angeles, CA":'USA',
                            "Washington, D.C.":'USA',
                            "California":'USA',
                             "Chicago, IL":'USA',
                             "Chicago":'USA',
                            "New York, NY":'USA',
                            "California, USA":'USA',
                            "FLorida":'USA',
                            "Nigeria":'Africa',
                            "Kenya":'Africa',
                            "Everywhere":'Worldwide',
                            "San Francisco":'USA',
                            "Florida":'USA',
                            "United Kingdom":'UK',
                            "Los Angeles":'USA',
                            "Toronto":'Canada',
                            "San Francisco, CA":'USA',
                            "NYC":'USA',
                            "Seattle":'USA',
                            "Earth":'Worldwide',
                            "Ireland":'UK',
                            "London, England":'UK',
                            "New York City":'USA',
                            "Texas":'USA',
                            "London, UK":'UK',
                            "Atlanta, GA":'USA',
                            "Mumbai":"India"},inplace=True)

sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],
            orient='h')

# Ref: https://www.kaggle.com/alex094495/getting-started-with-nlp-a-general-intro/edit
## examine the "keyword" distribution
df_train['keyword'].value_counts()
print('keywords for disaster tweets:','\n', df_train[df_train.target==1].keyword.value_counts().head(10), '\n')
print('keywords for non-disaster tweets:','\n',df_train[df_train.target==0].keyword.value_counts().head(10))
tweet = df_train
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

word=tweet[tweet['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
#ax1.set_title('disaster')
word=tweet[tweet['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
#ax2.set_title('Not disaster')
#fig.suptitle('Average word length in each tweet')

### As we can see, tweet with disasters have greater average lenth of text. 
### Our assumptions is that description of real disasters are more formal.
### Tweet lenghths distributions comparison between real disaster tweet and non-disaster ones
def length(text):    
    '''a function which returns the length of text'''
    return len(text)
tweet = df_train
tweet['length'] = tweet['text'].apply(length)

plt.rcParams['figure.figsize'] = (18.0, 6.0)
bins = 150
plt.hist(tweet[tweet['target'] == 0]['length'], alpha = 0.6, bins=bins, label='Not')
plt.hist(tweet[tweet['target'] == 1]['length'], alpha = 0.8, bins=bins, label='Real')
plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper right')
plt.xlim(0,150)
plt.grid()
plt.show()

##Ref: https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert
df=pd.concat([df_train,df_test])
df
!pip install pyspellchecker


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
df['text']=df['text'].apply(lambda x : remove_URL(x))


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
df['text']=df['text'].apply(lambda x : remove_html(x))
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


df['text']=df['text'].apply(lambda x: remove_emoji(x))

def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text
df['text']=df['text'].apply(lambda x: clean_text(x))


from spellchecker import SpellChecker

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
        
text = "corect me plese"
correct_spellings(text)

def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text
df['text']=df['text'].apply(lambda x: clean_text(x))
# Ref: https://www.kaggle.com/aaroha33/disaster-tweets-evaluation-with-nlp 

### Turn cleaned tweets into corpus (lower case, non stop, alphabetical words)
from tqdm import tqdm ### This is for a progress bar
def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus
corpus=create_corpus(df)

### Generate embedding dict from the GloVe txt file. So in the dictionary, every word is 
### associated with the GloVe representation of them
import numpy as np
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()
MAX_LEN=50
tokenizer_obj=Tokenizer() ## Initialize tokenizer
tokenizer_obj.fit_on_texts(corpus) 
sequences=tokenizer_obj.texts_to_sequences(corpus)### Convert each tweet in the corpus into sequence of numbers

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')### Pad the sequences so that they have the same length

word_index=tokenizer_obj.word_index
word_index
#Now each word in the corpus is associated with a number representing the location it is in the sequence
### Generate embedding matrix using the dictionary "embedding_dict={}" from GloVe
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,50))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
### Train test split
train=tweet_pad[:tweet.shape[0]]
test=tweet_pad[tweet.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)
### Neural Network
model=Sequential() ### Initiate neural network

embedding=Embedding(num_words,50,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False) ### First layer is the embedding layer

model.add(embedding) ## Add the GloVe embedding layer

model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2)) ### RNN
model.add(Dense(1, activation='sigmoid')) ### Sigmoid for binary classification


optimzer=Adam(learning_rate=1e-5)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])



model.summary() 



## Train the model
history=model.fit(X_train,y_train,batch_size=4,epochs=5,validation_data=(X_test,y_test),verbose=True)




predictions=model.predict(test)
predictions=np.round(predictions).astype(int).reshape(3263)

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = predictions
sample_submission.tail(50)
sample_submission.to_csv("submission.csv", index=False)
from tensorflow.keras import metrics
METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'), 
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
      metrics.AUC(name='auc')]
import matplotlib.gridspec as gridspec
def plot_model_eval(history):

    string = ['loss', 'accuracy']  
    cnt = 0
    ncols, nrows = 2, 1  
    fig = plt.figure(constrained_layout=True, figsize = (10,10))
    gs = gridspec.GridSpec(ncols = 3, nrows = 2, figure = fig)
    for i in range(nrows):
        for j in range(ncols):
            ax = plt.subplot(gs[i,j]) 
            ax.plot(history.history[string[cnt]])
            ax.plot(history.history['val_'+string[cnt]]) 
            ax.set_xlabel("Epochs")
            ax.set_ylabel(string[cnt])
            ax.legend([string[cnt], 'val_'+string[cnt]])
            cnt +=1
        
plot_model_eval(history)


## Getting Started : https://www.kaggle.com/alex094495/getting-started-with-nlp-a-general-intro/edit
## (很简单很基础的东西）
