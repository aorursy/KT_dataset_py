! pip install nlplot
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re

import string

from tqdm import tqdm, tqdm_notebook

tqdm.pandas()



import emoji

import nlplot

import nltk

from nltk.util import ngrams

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

stop=set(stopwords.words('english'))

from collections import defaultdict

from collections import  Counter



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report,accuracy_score



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D,BatchNormalization,TimeDistributed,Dropout,Bidirectional,Flatten,GlobalMaxPool1D

from keras.initializers import Constant

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



import matplotlib.pyplot as plt

%matplotlib inline



pd.set_option('display.max_columns', 300)

pd.set_option('display.max_rows', 300)

pd.set_option('display.max_colwidth', 300)

pd.options.display.float_format = '{:.3f}'.format
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
print('There are {} rows and {} columns in train'.format(train.shape[0],train.shape[1]))

print('There are {} rows and {} columns in test'.format(test.shape[0],test.shape[1]))
display(train.head(), test.head())
npt = nlplot.NLPlot(train, target_col='text')
stopwords = npt.get_stopword(top_n=30, min_freq=0)

stopwords
npt.bar_ngram(

    title='uni-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=1,

    top_n=30,

    height=700,

    stopwords=stopwords,

)
npt.bar_ngram(

    title='bi-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=2,

    top_n=30,

    height=700,

    stopwords=stopwords,

)
npt.bar_ngram(

    title='tri-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=3,

    top_n=30,

    height=700

)
# ビルド（データ件数によっては処理に時間を要します）

npt.build_graph(stopwords=stopwords, min_edge_frequency=22)
npt.co_network(

    title='Co-occurrence network',

    width=1000

)
npt.sunburst(

    title='sunburst chart',

    colorscale=True,

    width=1000

)
df = pd.concat([train,test])

df.shape
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)





def remove_html(text):

    html = re.compile(r'<.*?>')

    return html.sub(r'',text)





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





def remove_punct(text):

    table = str.maketrans('','',string.punctuation)

    return text.translate(table)





def remove_number(text):

    num = re.compile(r'\w*\d\w*')

    return num.sub(r'',text)





def acronyms(text):

    # cf. https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#4.-Embeddings-and-Text-Cleaning

    text = re.sub(r"MH370", "Malaysia Airlines Flight 370", text)

    text = re.sub(r"mÌ¼sica", "music", text)

    text = re.sub(r"okwx", "Oklahoma City Weather", text)

    text = re.sub(r"arwx", "Arkansas Weather", text)

    text = re.sub(r"gawx", "Georgia Weather", text)

    text = re.sub(r"scwx", "South Carolina Weather", text)

    text = re.sub(r"cawx", "California Weather", text)

    text = re.sub(r"tnwx", "Tennessee Weather", text)

    text = re.sub(r"azwx", "Arizona Weather", text)

    text = re.sub(r"alwx", "Alabama Weather", text)

    text = re.sub(r"wordpressdotcom", "wordpress", text)

    text = re.sub(r"usNWSgov", "United States National Weather Service", text)

    text = re.sub(r"Suruc", "Sanliurfa", text)



    # Grouping same words without embeddings

    text = re.sub(r"Bestnaijamade", "bestnaijamade", text)

    text = re.sub(r"SOUDELOR", "Soudelor", text)

    

    return text





def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = acronyms(text)

    text = text.lower()

    text = remove_URL(text)

    text = remove_html(text)

    text = remove_emoji(text)

    text = remove_punct(text)

    text = remove_number(text)

    

    return text
df['text'] = df['text'].progress_apply(lambda x: clean_text(x))
df.head()
# curpusの生成

def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus
corpus = create_corpus(df)
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN = 100

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences = tokenizer_obj.texts_to_sequences(corpus)



tweet_pad = pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec
train_tweet = tweet_pad[:train.shape[0]]

test_tweet = tweet_pad[train.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train_tweet, train['target'].values,test_size=0.1)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
def build_BLSTM():

    model = Sequential()

    model.add(Embedding(input_dim=embedding_matrix.shape[0], 

                        output_dim=embedding_matrix.shape[1], 

                        weights = [embedding_matrix], 

                        input_length=MAX_LEN))

    model.add(SpatialDropout1D(0.3))

    model.add(Bidirectional(LSTM(MAX_LEN, return_sequences = True, recurrent_dropout=0.2)))

    model.add(GlobalMaxPool1D())

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(MAX_LEN, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(MAX_LEN, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model
model = build_BLSTM()

model.summary()
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor = 'val_loss', 

    verbose = 1, 

    save_best_only = True

)



reduce_lr = ReduceLROnPlateau(

    monitor = 'val_loss', 

    factor = 0.2, 

    verbose = 1, 

    patience = 5,                        

    min_lr = 0.001

)
history = model.fit(

    X_train, 

    y_train, 

    epochs = 15,

    batch_size = 64,

    validation_data = [X_test, y_test],

    verbose = 1,

    callbacks = [reduce_lr, checkpoint]

)
def plot(history, arr):

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    for idx in range(2):

        ax[idx].plot(history.history[arr[idx][0]])

        ax[idx].plot(history.history[arr[idx][1]])

        ax[idx].legend([arr[idx][0], arr[idx][1]],fontsize=18)

        ax[idx].set_xlabel('A ',fontsize=16)

        ax[idx].set_ylabel('B',fontsize=16)

        ax[idx].set_title(arr[idx][0] + ' X ' + arr[idx][1],fontsize=16)

        

        

def metrics(pred_tag, y_test):

    print("F1-score: ", f1_score(pred_tag, y_test))

    print("Precision: ", precision_score(pred_tag, y_test))

    print("Recall: ", recall_score(pred_tag, y_test))

    print("Acuracy: ", accuracy_score(pred_tag, y_test))

    print("-"*50)

    print(classification_report(pred_tag, y_test))
plot(history, [['loss', 'val_loss'],['accuracy', 'val_accuracy']])
loss, accuracy = model.evaluate(X_test, y_test)

print('Loss:', loss)

print('Accuracy:', accuracy)
preds = model.predict_classes(X_test)

metrics(preds, y_test)
y_pred = model.predict(test_tweet)

y_pred = np.round(y_pred).astype(int).reshape(3263)

sub = pd.DataFrame({'id':submission['id'].values.tolist(),'target':y_pred})
sub
sub['target'].hist()
sub.to_csv('submission.csv',index=False)