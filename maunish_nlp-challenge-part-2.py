import pandas as pd
import numpy as np

#tqdm for progress bars
from tqdm import tqdm

#scikit-learn library
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#gradient boosting
import xgboost as xgb

#keras library
from keras.models import Sequential
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D,Conv1D,MaxPooling1D,Flatten,Bidirectional,SpatialDropout1D,Embedding
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

#nltk library
from nltk import word_tokenize
from nltk.corpus import stopwords
#ignore the warnings
import warnings
warnings.filterwarnings('ignore')

#gensim library allow us to access pre trained embeddings
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
#download the dataset and return it as object
model_twitter_glove = api.load("glove-twitter-25") #here 25 is dimenssion of the data
#let's see which are top 10 most similar words to apple.
model_twitter_glove.wv.most_similar("apple",topn=10)
#Let's get fruit this time
model_twitter_glove.wv.most_similar("pineapple",topn=5)
model_twitter_glove.wv.most_similar("politics",topn=5)
model_twitter_glove.wv.doesnt_match(["car","truck","bike","orange"])
# now let's try that king and queen example
model_twitter_glove.wv.most_similar(positive=['king','woman'],negative=['man'])
#loading data
PATH = '../input/spooky-author-identification'
train = pd.read_csv(f'{PATH}/train.zip')
test = pd.read_csv(f'{PATH}/test.zip')
sample = pd.read_csv(f'{PATH}/sample_submission.zip')


#data preprocssing
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(train["author"].values)

#data split
X_train, X_test, y_train, y_test = train_test_split(train.text.values,y,random_state=42,test_size=0.1,shuffle=True)
#our loss function
def multiclass_logloss(actual,predicted,eps=1e-15):
    
    #converting the 'actual' values to binary values if it's 
    #not binary values
    
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0],predicted.shape[1]))
        
        for i, val in enumerate(actual):
            actual2[i,val] = 1
        actual = actual2
    
    #clip function truncates the number between
    #a max number and min number
    clip = np.clip(predicted,eps,1-eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0/ rows * vsota 
#we need to word, vec dictionary before fitting it to models
embedding_index = {}

all_words = list(model_twitter_glove.wv.vocab.keys())
#words in gensim model is stored as "key":vector object pair

for word in all_words:
    embedding_index[word] = model_twitter_glove.wv.get_vector(word)

print('Total words in embeddings %d' % len(embedding_index))
#getting stop words from nltk library
stop_words = stopwords.words('english')

def sen2vec(s):
    # lowe the letters, tokenize them , remove stop_words, remove numbers
    words = str(s).lower()
    words = word_tokenize(s)
    words = [w for w in words if w not in stop_words]
    words = [w for w in words if w.isalpha()]
    
    M = []
    for w in words:
        #try because word might not present in index.
        try:
            M.append(embedding_index[w])
        except:
            continue
    
    M = np.array(M)
    v = M.sum(axis=0)
    
    if type(v) != np.ndarray:
        #25 because that is dimension of out word embedding
        return np.zeros(25)
    
    return v/np.sqrt((v** 2).sum())
#tqdm is libray which shows you progress bar
#you can write this code without it

#converting every sentence to word embedding
X_train_glove = [sen2vec(s) for s in tqdm(X_train)]
X_test_glove = [sen2vec(s) for s in tqdm(X_test)]
X_train_glove = np.array(X_train_glove)
X_test_glove = np.array(X_test_glove)
clf = xgb.XGBClassifier(n_estimators=200,nthread=10,silent=False)
clf.fit(X_train_glove, y_train)

predictions = clf.predict_proba(X_test_glove)

print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
#scaling data
scl = preprocessing.StandardScaler()

X_train_glove_scl = scl.fit_transform(X_train_glove)
X_test_glove_scl = scl.transform(X_test_glove)

y_train_enc = np_utils.to_categorical(y_train)
y_test_enc = np_utils.to_categorical(y_test)
model = Sequential()

model.add(Dense(25,input_dim=25,activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(25,input_dim=25,activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam')
#Just 5 epochs for testing
model.fit(X_train_glove_scl,y=y_train_enc,batch_size=50,epochs=5,verbose=1,validation_data=(X_test_glove_scl,y_test_enc))
#use keras tokenizer
token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(X_train)+list(X_test))
X_train_sec = token.texts_to_sequences(X_train)
X_test_sec = token.texts_to_sequences(X_test)

X_train_pad = sequence.pad_sequences(X_train_sec,maxlen=max_len)
X_test_pad = sequence.pad_sequences(X_test_sec,maxlen=max_len)

word_index = token.word_index
model = Sequential()
# we are not using pretrainde embedding yet.
model.add(Embedding(len(word_index)+1,25,input_length=max_len))
model.add(SimpleRNN(100))
model.add(Dense(3))
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(X_train_pad,y=y_train_enc,epochs=5,batch_size=100,validation_data=(X_test_pad,y_test_enc))
embedding_matrix = np.zeros((len(word_index)+1,25)) #25 because we have word vector of dim 25

for word,i in word_index.items():
    #we use get() so it returns None if word is not found
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
model = Sequential()
# we are not using pretrainde embedding yet.
model.add(Embedding(len(word_index)+1,25,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False))
#as the weight are predefined trainable is False
model.add(SimpleRNN(100))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(X_train_pad,y=y_train_enc,epochs=5,batch_size=100,validation_data=(X_test_pad,y_test_enc))
model = Sequential()

model.add(Embedding(len(word_index)+1,25,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False))

model.add(SpatialDropout1D(0.3))
model.add(LSTM(100,dropout=0.3,recurrent_dropout=0.3))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train_pad,y=y_train_enc,batch_size=100,epochs=10,verbose=1,
          validation_data=(X_test_pad,y_test_enc))
model = Sequential()

model.add(Embedding(len(word_index)+1,25,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False))

model.add(SpatialDropout1D(0.3))
model.add(GRU(100))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(X_train_pad,y=y_train_enc,batch_size=100,epochs=5,verbose=1,validation_data=(X_test_pad,y_test_enc))
model = Sequential()

model.add(Embedding(len(word_index)+1,25,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False))

model.add(Bidirectional(LSTM(25, dropout=0.3, recurrent_dropout=0.3)))
    
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train_pad,y=y_train_enc,batch_size=100,epochs=5,verbose=1,validation_data=(X_test_pad,y_test_enc))