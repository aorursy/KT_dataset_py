import pandas as pd
import re
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, GRU, Embedding
from keras.layers import Activation, Bidirectional, GlobalMaxPool1D, GlobalMaxPool2D, Dropout
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dropout
#from tensorflow.keras.layers import concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, adam
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
import seaborn as sns
import transformers
from transformers import AutoTokenizer
from tokenizers import BertWordPieceTokenizer
from keras.initializers import Constant
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from collections import Counter

stop=set(stopwords.words('english'))


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df.head()
df.dropna(inplace=True)
#https://www.kaggle.com/shahules/complete-eda-baseline-model-0-708-lb

def basic_cleaning(text):
    text=re.sub(r'https?://www\.\S+\.com','',text)
    text=re.sub(r'[^A-Za-z|\s]','',text)
    text=re.sub(r'\*+','swear',text) #capture swear words that are **** out
    return text

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

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

def remove_multiplechars(text):
    text = re.sub(r'(.)\1{3,}',r'\1', text)
    return text


def clean(df):
    for col in ['text']:#,'selected_text']:
        df[col]=df[col].astype(str).apply(lambda x:basic_cleaning(x))
        df[col]=df[col].astype(str).apply(lambda x:remove_emoji(x))
        df[col]=df[col].astype(str).apply(lambda x:remove_html(x))
        df[col]=df[col].astype(str).apply(lambda x:remove_multiplechars(x))

    return df
df_clean = clean(df)
df_clean.head()
shape = df_clean.shape
print(f"There are {shape[0]} tweets in the dataset")
colors=['orange','red','green']
plt.bar(df.sentiment.unique(),df.sentiment.value_counts(), color=colors);
plt.xlabel('Tweet Sentiment');
plt.ylabel('Tweet Count');
sent=df.sentiment.unique()
fig,ax= plt.subplots(1,3,figsize=(12,6),sharey=True)
for i in range(0,3):
    lengths = df_clean[df_clean['sentiment']==sent[i]]['text'].str.split().str.len()
    ax[i].boxplot(lengths)
    ax[i].set_title(sent[i])
ax[0].set_ylabel('Number of words in Tweet')
fig.suptitle("Distribution of number Words in Tweets", fontsize=14);
def preprocess_news(df,stop=stop,n=1,col='text'):
    '''Function to preprocess and create corpus'''
    new_corpus=[]
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    for text in df[col]:
        words=[w for w in word_tokenize(text) if (w not in stop)]
       
        words=[lem.lemmatize(w) for w in words if(len(w)>n)]
     
        new_corpus.append(words)
        
    new_corpus=[word for l in new_corpus for word in l]
    return new_corpus
fig,ax=plt.subplots(1,3,figsize=(12,6))
for i in range(3):
    new=df_clean[df_clean['sentiment']==sent[i]]
    corpus_train=preprocess_news(new,n=3)
    counter=Counter(corpus_train)
    most=counter.most_common()
    x=[]
    y=[]
    for word,count in most[:10]:
        if (word not in stop) :
            x.append(word)
            y.append(count)
    sns.barplot(x=y,y=x,ax=ax[i],color=colors[i])
    ax[i].set_title(sent[i],color=colors[i])
fig.suptitle("Common words in tweet text")
df_clean_selection = df_clean.sample(frac=0.2)
X = df_clean_selection.text.values
y = pd.get_dummies(df_clean_selection.sentiment)
tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(X))
list_tokenized_train = tokenizer.texts_to_sequences(X)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=128)
def create_model(): 
    embedding_size = 128
    input_ = Input(shape=(128,))
    x = Embedding(20000, embedding_size)(input_)
    x = Bidirectional(LSTM(50,return_sequences=True))(x)
    x = Bidirectional(LSTM(25,return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu', kernel_regularizer='L1L2')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs=input_, outputs=x)
    return model

#model = KerasClassifier(create_model)
optimizers_names = ['SGD', 'RMSprop', 'Adam']
model = create_model()

for optim in optimizers_names: 
    
    model.compile(loss='categorical_crossentropy',optimizer=optim,metrics=['accuracy'])
    history = model.fit(X_t,y,batch_size=32,epochs=2,validation_split=0.1)

learning_rate = [0.001,0.01,0.1]

for rate in learning_rate:
    Adam_name = adam(lr=rate)
    model.compile(loss='categorical_crossentropy',optimizer=Adam_name,metrics=['accuracy'])
    history = model.fit(X_t,y,batch_size=32,epochs=2,validation_split=0.1)
tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")  ## change it to commit

# Save the loaded tokenizer locally
save_path = '/kaggle/working/distilbert_base_uncased/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.save_pretrained(save_path)

# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowercase=True)
fast_tokenizer
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=128):    
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
X = fast_encode(df_clean_selection.text.astype(str), fast_tokenizer, maxlen=128)
X.shape
transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
embedding_size = 128
input_ = Input(shape=(100,))

inp = Input(shape=(128, ))
#inp2= Input(shape=(1,))

embedding_matrix=transformer_layer.weights[0].numpy()

x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],embeddings_initializer=Constant(embedding_matrix),trainable=False)(inp)
x = Bidirectional(LSTM(50, return_sequences=True))(x)
x = Bidirectional(LSTM(25, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(50, activation='relu', kernel_regularizer='L1L2')(x)
x = Dropout(0.5)(x)
x = Dense(3, activation='softmax')(x)

model_DistilBert = Model(inputs=[inp], outputs=x)
model_DistilBert.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_DistilBert.summary()
model_DistilBert.fit(X,y,batch_size=32,epochs=2,validation_split=0.1)
df_clean_final = df_clean.sample(frac=1)
X_train = fast_encode(df_clean_selection.text.astype(str), fast_tokenizer, maxlen=128)
y_train = y
Adam_name = adam(lr=0.001)
model_DistilBert.compile(loss='categorical_crossentropy',optimizer=Adam_name,metrics=['accuracy'])
history = model_DistilBert.fit(X_train,y_train,batch_size=32,epochs=10)
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_test.dropna(inplace=True)
df_clean_test = clean(df_test)

X_test = fast_encode(df_clean_test.text.values.astype(str), fast_tokenizer, maxlen=128)
y_test = df_clean_test.sentiment
y_preds = model_DistilBert.predict(X_test)
y_predictions = pd.DataFrame(y_preds, columns=['negative','neutral','positive'])
y_predictions_final = y_predictions.idxmax(axis=1)
accuracy = accuracy_score(y_test,y_predictions_final)
print(f"The final model shows {accuracy:.2f} accuracy on the test set.")