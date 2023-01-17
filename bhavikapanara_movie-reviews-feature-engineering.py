#basic 

import pandas as pd
import numpy as np
pd.set_option('max_colwidth',400)
import string

#Graph
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
%matplotlib inline

#machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import hstack


#Deep Learning
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

#NLP
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
import textblob
import re
train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv' , sep="\t")
sub =pd.read_csv('../input/sampleSubmission.csv')

train.shape , test.shape
train.head()
train.loc[train['SentenceId'] == 2]
def check_missing(df):
    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
    

missing_data_df = check_missing(train)
missing_data_test = check_missing(test)
print('Missing data in train set: \n' , missing_data_df.head())
print('\nMissing data in test set: \n'  ,missing_data_test.head())

temp = train['Sentiment'].value_counts()

trace = go.Bar(
    x = temp.index,
    y = temp.values,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of Class Label in train dataset",
    xaxis=dict(
        title='Class Label',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Occurance of Class label',
        titlefont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Sentiment')
length = len(train)
df = pd.concat([train, test], axis=0)
# generate clean text from Phrase 
def review_to_words(raw_review): 
    review =raw_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))

df['Clean_text'] = df['Phrase'].apply(lambda x : review_to_words(x))

df['Clean_text'].replace('', str('something'), inplace=True)

df['char_count'] = df['Phrase'].apply(len)
df['word_count'] = df['Phrase'].apply(lambda x: len(x.split()))
df['word_density'] = df['char_count'] / (df['word_count']+1)
df['punctuation_count'] = df['Phrase'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
df['title_word_count'] = df['Phrase'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
df['upper_case_word_count'] = df['Phrase'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
df["stopword_count"] = df['Phrase'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
positive = df['Clean_text'][df['Sentiment']== 4 ]

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, max_features = 6877)

pos_words = vectorizer.fit_transform(positive)
pos_words = pos_words.toarray()
pos= vectorizer.get_feature_names()
print ("Total number of positive words : " ,len(pos))

dist = np.sum(pos_words, axis=0)
postive_new= pd.DataFrame(dist)
postive_new.columns=['word_count']
postive_new['word'] = pd.Series(pos, index=postive_new.index)
top = postive_new.sort_values(['word_count'] , ascending = False )

negative=df['Clean_text'][df['Sentiment']== 0]

neg_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, max_features = 6891)

neg_words = neg_vectorizer.fit_transform(negative)
neg_words = neg_words.toarray()
neg= neg_vectorizer.get_feature_names()
print ("Total number of negative words :",len(neg))

dist = np.sum(neg_words, axis=0)
negative_new= pd.DataFrame(dist)
negative_new.columns=['word_count']
negative_new['word'] = pd.Series(neg, index=negative_new.index)
top_neg = negative_new.sort_values(['word_count'] , ascending = False )

def count_word(x , pos_tag):
    cnt = 0
    if pos_tag:
        for e in x.split():
            if e in pos:
                cnt = cnt + 1
    else:
        for e in x.split():
            if e in neg:
                cnt = cnt + 1
    return cnt
    
df['pos_cnt'] = df['Clean_text'].apply(lambda x : count_word(x , pos_tag = True))
df['neg_cnt'] = df['Clean_text'].apply(lambda x : count_word(x, pos_tag = False))

df['Ratio'] = df['pos_cnt'] / (df['neg_cnt']+0.0001)

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    wiki = textblob.TextBlob(x)
    for tup in wiki.tags:
        ppo = list(tup)[1]
        if ppo in pos_family[flag]:
            cnt += 1

    return cnt

df['noun_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count'] = df['Phrase'].apply(lambda x: check_pos_tag(x, 'pron'))

def getSentFeat(s , polarity):
    sent = textblob.TextBlob(s).sentiment
    if polarity:
        return sent.polarity
    else :
        return sent.subjectivity
    
df['polarity'] = df['Phrase'].apply(lambda x: getSentFeat(x , polarity=True))
df['subjectivity'] = df['Phrase'].apply(lambda x: getSentFeat(x , polarity=False))

#separate train and test data
train = df[:length]
test = df[length:]
train.shape, test.shape
train.describe()

plt.figure(figsize=(12,6))
plt.subplot(121)
sns.violinplot(y='pos_cnt',x='Sentiment', data=train,split=True)
plt.xlabel('Class Label', fontsize=12)
plt.ylabel('# of Positive words ', fontsize=12)
plt.title("Number of Positive word in each review", fontsize=15)

plt.subplot(122)
sns.violinplot(y='neg_cnt',x='Sentiment', data=train,split=True)
plt.xlabel('Class label', fontsize=12)
plt.ylabel('# of negative words', fontsize=12)
plt.title("Number of Negative words in each review", fontsize=15)

plt.show()
f,ax = plt.subplots(figsize=(15,15))    #correlation between numerical values' maps
sns.heatmap(train.corr() , annot = True, linewidths = .5, fmt= '.1f', ax=ax , vmin=-1, vmax=1)
plt.legend()
plt.show() 
# Standardize numeric feature

ss = StandardScaler()
num_col = [ 'pos_cnt', 'neg_cnt' , 'Ratio','polarity','subjectivity' ,
            'char_count' , 'word_count' , 'word_density' , 'punctuation_count','title_word_count' ,
           'upper_case_word_count' ,'stopword_count' ,
            'adv_count' ,'verb_count','adj_count', 'pron_count' , 'noun_count']

X_num = ss.fit_transform(train[num_col].fillna(-1).clip(0.0001 , 0.99999))

y = train['Sentiment']
# vectorization of text data

count_vect = CountVectorizer(analyzer='word', ngram_range=(1,2))

X_txt = count_vect.fit_transform(train['Phrase'])

X_train, X_val, Y_train, Y_val = train_test_split(X_num, y, test_size=0.10, random_state=1234)

clf = LogisticRegression(C=3)

clf.fit(X_train,Y_train)
clf.score(X_val,Y_val)

plt.figure(figsize=(16,22))
plt.suptitle("Feature importance",fontsize=20)
gridspec.GridSpec(3,2)
plt.subplots_adjust(hspace=0.4)
plt.subplot2grid((3,2),(0,0))
sns.barplot(num_col,clf.coef_[0],color=color[0])
plt.title("class : Negative",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)

plt.subplot2grid((3,2),(0,1))
sns.barplot(num_col,clf.coef_[1] , color=color[1])
plt.title("class : Somewhat negative",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)

plt.subplot2grid((3,2),(1,0))
sns.barplot(num_col,clf.coef_[2],color=color[2])
plt.title("class : Neutral",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


plt.subplot2grid((3,2),(1,1))
sns.barplot(num_col,clf.coef_[3],color=color[3])
plt.title("class : Somewhat positive",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


plt.subplot2grid((3,2),(2,0))
sns.barplot(num_col,clf.coef_[4],color=color[4])
plt.title("class : Positive",fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)


plt.show()
x = hstack((X_num,X_txt)).tocsr()

X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.10, random_state=1234)

clf1 = LogisticRegression(C = 3)

clf1.fit(X_train,Y_train)
clf1.score(X_val,Y_val)
#pre-processing of data for keras model

train_DL = train.drop(['PhraseId' , 'SentenceId' , 'Sentiment'] , axis =1)
y = train['Sentiment']

ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))

X_train, X_val, Y_train, Y_val = train_test_split(train_DL, y_ohe, test_size=0.10, random_state=1234)

tk = Tokenizer(lower = True, filters='', num_words= 15000)
tk.fit_on_texts(train_DL['Phrase'])

train_tokenized = tk.texts_to_sequences(X_train['Phrase'])
valid_tokenized = tk.texts_to_sequences(X_val['Phrase'])

max_len = 80
X_train_txt = pad_sequences(train_tokenized, maxlen = max_len)
X_valid_txt = pad_sequences(valid_tokenized, maxlen = max_len)

X_num_train = ss.transform(X_train[num_col].fillna(-1).clip(0.0001 , 0.99999))
X_num_valid = ss.transform(X_val[num_col].fillna(-1).clip(0.0001 , 0.99999))

inp = Input(shape = (max_len,))
input_num = Input((len(num_col), ))

x = Embedding(15000 , 100 ,mask_zero=True)(inp)
x = LSTM(128, dropout=0.4, recurrent_dropout=0.4,return_sequences=True)(x)
x = LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False)(x)

x_num = Dense(64, activation="relu")(input_num)   
X_num = Dropout(0.2)(x_num)
X_num = Dense(32, activation = "relu")(X_num)

xx = concatenate([x_num, x])
xx = BatchNormalization()(xx)
xx = Dropout(0.1)(Dense(20, activation='relu') (xx))

outp = Dense(5, activation = "softmax")(xx)

model = Model(inputs = [inp,input_num], outputs = outp)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([X_train_txt,X_num_train], Y_train, validation_data=([X_valid_txt,X_num_valid], Y_val),
         epochs=6, batch_size=128, verbose=1)

accuracy = model.evaluate([X_valid_txt,X_num_valid], Y_val )[1]
accuracy

