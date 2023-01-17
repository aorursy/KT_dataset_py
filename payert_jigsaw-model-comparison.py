import numpy as np # linear algebra

import pandas as pd

import os

import matplotlib.pyplot as plt

import gc

import time

%matplotlib inline



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text  import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score

from sklearn import metrics
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

print(train.shape)

print(test_df.shape)

print(test_df.head())
train.head()
test_df.head()
train=train.iloc[:,1:3]

print(train.head())

y_trainAll =  np.where(train['target']>=0.5, 1, 0)

X_trainAll=train.drop('target',axis=1)
sum(y_trainAll)/len(y_trainAll)
print(sum(y_trainAll))

print(len(y_trainAll))

(len(y_trainAll) - sum(y_trainAll))/sum(y_trainAll)
import seaborn as sns

sns.countplot(y_trainAll)
import re, string, timeit, datetime



def clean(train_clean):

    tic = datetime.datetime.now()

    train_clean['comment_text']=train_clean['comment_text'].str.replace('[0-9]+',' ') ### remove numbers

    train_clean['comment_text']=train_clean['comment_text'].apply(lambda x : x.lower()) ### to lower case

    train_clean['comment_text']=train_clean['comment_text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    train_clean['comment_text']=train_clean['comment_text'].str.replace('[0-9]',' ') ### remove numbers

    tac = datetime.datetime.now(); time = tac - tic; print("To lower time" + str(time))

    print("remove punct time" + str(time))

    gc.collect()

    return(train_clean)





train_cl=clean(X_trainAll)

train_cl.head()
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True, strip_accents='unicode',  analyzer='word',

     stop_words='english', ngram_range=(1, 2),token_pattern=r'(?u)\b[A-Za-z]+\b',  #erhoehen auf 2

     max_features=50000) 



tfidf_train = word_vectorizer.fit_transform(train_cl['comment_text'])

print(word_vectorizer.get_feature_names()[:10])

print( len( word_vectorizer.get_feature_names() ))



gc.collect()
out=word_vectorizer.vocabulary_ ; list(out)[1:10]
n_size=len(y_trainAll); print(n_size/10)

sub_sample = np.random.choice(range(0, n_size), size=180487, replace=False).tolist()

#sub_sample[:20]
zw_df=pd.concat([train_cl, pd.DataFrame(y_trainAll)], axis=1)

zw_df.columns=['comment_text', 'target']

#print(zw_df.shape)

zw_df=zw_df.iloc[sub_sample,:]

#print(zw_df.shape)



toxic_comments = zw_df[zw_df['target'] >= .5]['comment_text'].values

toxic_comments = ' '.join(toxic_comments)



non_toxic_comments = zw_df[zw_df['target'] < .5]['comment_text'].values

non_toxic_comments = ' '.join(non_toxic_comments)

del zw_df, test_df, out ; gc.collect() 
import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))



from wordcloud import WordCloud

wordcloud_toxic = WordCloud(max_font_size=100, max_words=100, background_color="white",  stopwords=stop_words).generate(toxic_comments)

plt.figure(figsize=[15,5])    

# Display the generated image:

plt.title("Wordcloud: Toxic comments")

plt.imshow(wordcloud_toxic, interpolation='bilinear')

plt.axis("off")

plt.show()

del wordcloud_toxic,X_trainAll  , toxic_comments,   train_cl, word_vectorizer, train  ; 

gc.collect()
wordcloud_non_toxic = WordCloud(max_font_size=100, max_words=100, background_color="white",  stopwords=stop_words).generate(non_toxic_comments)

plt.figure(figsize=[15,5])

plt.title("Wordcloud: Non-Toxic comments")

plt.imshow(wordcloud_non_toxic, interpolation='bilinear')

plt.axis("off")

plt.show()



del wordcloud_non_toxic,  non_toxic_comments, stop_words, 

gc.collect()
from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB, ComplementNB

from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score

from sklearn.model_selection import StratifiedKFold
def train_and_predictNB(alpha, train_x,train_y, valid_x,valid_y):

    # Instantiate the classifier: nb_classifier

    nb_classifier = MultinomialNB(alpha)# count_train: best auc=0.769 bei alpha = 0.0 on count_train

    nb_classifier.fit(train_x,train_y)

    pred = nb_classifier.predict_proba(valid_x); pred=pd.DataFrame(pred); #print(pred[:3])

    auc = roc_auc_score(valid_y, pred[1]); #print(auc);# print(pred[1]) #print('AUC: %.3 f' % auc)

    pred = nb_classifier.predict(valid_x)

    score = metrics.accuracy_score(valid_y, pred)

    del nb_classifier, pred

    return [round(score,5), round(auc,5)]
X_train_tf, X_valid_tf, y_train, y_valid = train_test_split(tfidf_train, y_trainAll, test_size = 0.2, random_state = 53)
alpha=1.2

print('Alpha: ', alpha)

out=train_and_predictNB(alpha, X_train_tf, y_train, X_valid_tf, y_valid)

print('Accuracy: ', out[0])                             

print('AUC: ',out[1])
acc_out=[]; auc_out=[]

nfold = 5

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=123)

i = 1

for train_index, valid_index in skf.split(tfidf_train, y_trainAll):

    print("\nFold {}".format(i)); i+=1

    print(len(train_index));print(len(valid_index))

    out=train_and_predictNB(alpha, tfidf_train[train_index], y_trainAll[train_index], tfidf_train[valid_index], y_trainAll[valid_index])

    print(out)

    acc_out.append(out[0]); auc_out.append(out[1])

print(acc_out)   ; print(auc_out) 

print("Mean-Acc: ", round(np.mean(acc_out),5) )

print("Mean-AUC: ", round(np.mean(auc_out),5) )
np.random.seed(seed=234)

i_class0 = np.where(y_trainAll == 0)[0] ; i_class1 = np.where(y_trainAll == 1)[0]

n_class0 = len(i_class0) ; n_class1 = len(i_class1)

i_class0_downsampled = np.random.choice(i_class0, size=n_class1, replace=False)

ds_index=np.concatenate((i_class1,i_class0_downsampled))

print(n_class1); print(n_class0); print(len(ds_index))



y_train_ds=y_trainAll[ds_index]; tfidf_train_ds =tfidf_train[ds_index]
def downsample(x_orig, y_orig):

    np.random.seed(seed=234)

    i_class0 = np.where(y_orig == 0)[0] ; i_class1 = np.where(y_orig == 1)[0]

    n_class0 = len(i_class0) ; n_class1 = len(i_class1)

    if n_class0 > n_class1:

        i_class0_downsampled = np.random.choice(i_class0, size=n_class1, replace=False);

        ds_index=np.concatenate((i_class1,i_class0_downsampled))

    else: 

        i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False);

        ds_index=np.concatenate((i_class0,i_class1_downsampled)) 

    #print(n_class1); print(n_class0); print(len(ds_index))



    y_ds=y_orig[ds_index]; X_ds =x_orig[ds_index]

    return X_ds, y_ds

    
acc_out=[]; auc_out=[]

nfold = 5

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=123)

i = 1

for train_index, valid_index in skf.split(tfidf_train, y_trainAll):

    tfidf_train_ds, y_train_ds = downsample(tfidf_train[train_index], y_trainAll[train_index])

    out=train_and_predictNB(alpha, tfidf_train_ds, y_train_ds, tfidf_train[valid_index], y_trainAll[valid_index])

    acc_out.append(out[0]); auc_out.append(out[1])

print(acc_out)   ; print(auc_out) 

print("Mean-Acc: ", round(np.mean(acc_out),5) )

print("Mean-AUC: ", round(np.mean(auc_out),5) )
from sklearn.linear_model import LogisticRegression
def train_and_predictLogR(cl,c_weight=None):                                                      #c=0.8; l1 =0.9444   l2 ist schlechter

    logreg = LogisticRegression(C=cl,penalty='l1',class_weight=c_weight, solver='liblinear')    #class_weight : dict or ‘balanced’, optional (default=None)

    logreg.fit(X_train_tf, y_train)

    pred = logreg.predict_proba(X_valid_tf);pred=pd.DataFrame(pred)

    

    auc = roc_auc_score(y_valid, pred[1]); print('auc: ',auc)     

    pred = logreg.predict(X_valid_tf)

    score = metrics.accuracy_score(y_valid, pred)

    del logreg, pred

    return score



#print('Score: ', train_and_predictLogR(1))   

#classos = np.arange(0.001,3,.2)



classos =[.4,.6,.8 ]



for classo in classos:

    print('classo: ', classo)

    print('Score: ', train_and_predictLogR(classo))                              #0.8782946199369265

    print()
def train_and_predictLogR(c_par, train_x,train_y, valid_x,valid_y, c_weight=None):

    logreg = LogisticRegression(C=c_par,penalty='l1', solver='liblinear' , class_weight=c_weight)   

    logreg.fit(train_x, train_y)

    pred = logreg.predict_proba(valid_x);pred=pd.DataFrame(pred)        

    auc = roc_auc_score(valid_y, pred[1]); #print(auc);# print(pred[1]) #print('AUC: %.3 f' % auc)

    pred = logreg.predict(valid_x)

    score = metrics.accuracy_score(valid_y, pred)

    return [round(score,5), round(auc,5)]
c_par=0.6

start = time.time()

acc_out=[]; auc_out=[]   ;         

nfold = 5

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=123)

i = 1

for train_index, valid_index in skf.split(tfidf_train, y_trainAll):

    #print("\nFold {}".format(i)); i+=1  #print(len(train_index));print(len(valid_index))

    out=train_and_predictLogR(c_par, tfidf_train[train_index], y_trainAll[train_index], tfidf_train[valid_index], y_trainAll[valid_index])    #print(out)

    acc_out.append(out[0]); auc_out.append(out[1])

    

    

print(acc_out)   ; print(auc_out) 

print("Mean-Acc: ", round(np.mean(acc_out),5) )

print("Mean-AUC: ", round(np.mean(auc_out),5) )   ;end = time.time(); print((end - start)/60)
start = time.time()

acc_out=[]; auc_out=[]   ;       

nfold = 5

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=123)

i = 1

for train_index, valid_index in skf.split(tfidf_train, y_trainAll):

    tfidf_train_ds, y_train_ds = downsample(tfidf_train[train_index], y_trainAll[train_index])

    out=train_and_predictLogR(c_par, tfidf_train_ds, y_train_ds, tfidf_train[valid_index], y_trainAll[valid_index])    #print(out)

    acc_out.append(out[0]); auc_out.append(out[1])

    

    

print(acc_out)   ; print(auc_out) 

print("Mean-Acc: ", round(np.mean(acc_out),5) )

print("Mean-AUC: ", round(np.mean(auc_out),5) )   ;end = time.time(); print((end - start)/60)
start = time.time()

acc_out=[]; auc_out=[]   ;       

nfold = 5

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=123)

i = 1

for train_index, valid_index in skf.split(tfidf_train, y_trainAll):

    out=train_and_predictLogR(c_par, tfidf_train[train_index], y_trainAll[train_index], tfidf_train[valid_index], y_trainAll[valid_index],c_weight='balanced')    #print(out)

    acc_out.append(out[0]); auc_out.append(out[1])

    

    

print(acc_out)   ; print(auc_out) 

print("Mean-Acc: ", round(np.mean(acc_out),5) )

print("Mean-AUC: ", round(np.mean(auc_out),5) )  ;end = time.time(); print((end - start)/60)
import time

import lightgbm as lgb

train_data = lgb.Dataset(X_train_tf, y_train)

valid_data = lgb.Dataset(X_valid_tf, y_valid ) #tfidf_test, reference=train_data)



param = {

    'num_trees':5000,   #0.942217   ;   0.94268   #0.94338  (20; 0.1);0.94316 (30,0.1);  0.9434 (25,0.1/32min); 0.943271 (25;0.05/55)

    'learning_rate':0.1,

    "objective": "binary",

    'num_leaves':25,

    'metric': ['auc'],

    "num_threads": -1,

    "early_stopping_rounds":20,

    "verbose":1,

    'boost_from_average': False,    

}



start = time.time()

bdt = lgb.train(param, train_data, valid_sets=[valid_data], verbose_eval=100)  

end = time.time(); print((end - start)/60)
def train_and_predictLGBM01(train_x,train_y, valid_x,valid_y, num_trees=1):                  #[1126]	valid_0's auc: 0.943421

    param = {

    'num_trees':num_trees,    'learning_rate':0.1,  "objective": "binary",  'num_leaves':25,

    'metric': ['auc'],   "num_threads": -1,   # "early_stopping_rounds":20,

    "verbose":1,'boost_from_average': False,     #'is_unbalance': True,                       

     #'scale_pos_weight': ch_weights,                        

     }

    train_data = lgb.Dataset(train_x, train_y)

    bdt = lgb.train(param, train_data,  verbose_eval=500) 

    pred = bdt.predict(valid_x)  ;         

    auc = roc_auc_score(valid_y, pred); #print(auc);# print(pred[1]) #print('AUC: %.3 f' % auc)

    pred_dichotom=np.where(pred >=0.5, 1, 0); pred=pd.DataFrame(pred)

    #pred = bdt.predict(valid_x)

    score = metrics.accuracy_score(valid_y, pred_dichotom)

    return [round(score,5), round(auc,5)]
#train_and_predictLGBM01(X_train_tf, y_train, X_valid_tf, y_valid , num_trees=120)#

del train_data, X_train_tf, X_valid_tf,bdt,  valid_data, out, tfidf_train, y_trainAll

gc.collect()
for name in dir():

    if not name.startswith('_'):

        del globals()[name]



for name in dir():

    if not name.startswith('_'):

        del locals()[name]



%reset -f        

import gc        

gc.collect()        
import pandas as pd

import numpy as np

from tqdm import tqdm

tqdm.pandas()

import time



from keras.preprocessing import text, sequence

from keras import backend as K

from keras.models import load_model

import keras

import pickle

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
def clean_text(x):

    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x



train["comment_text"] = train["comment_text"].progress_apply(lambda x: clean_text(x))
train_data = train["comment_text"]

label_data = train.target.apply(lambda x: 0 if x < 0.5 else 1)

train_data.shape, label_data.shape
MAX_LEN = 200

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'



tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(train_data) )



train_data = tokenizer.texts_to_sequences(train_data)

train_data = sequence.pad_sequences(train_data, maxlen=MAX_LEN)
x_train, x_val, y_train, y_val = train_test_split(train_data, label_data, test_size = 0.35, random_state = 53)
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix



EMBEDDING_FILES = [ '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec']



start = time.time()



embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)



end = time.time(); elapsed = end - start; print(elapsed/60)

gc.collect()

embedding_matrix.shape
from keras.models import Sequential, Model

from keras.optimizers import  Adam

from keras.layers import Flatten, Dense, Embedding, Dropout, Bidirectional, Input, add #,  CuDNNLSTM,

from keras.layers import concatenate,  SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM, CuDNNLSTM

from keras.utils import plot_model

import matplotlib.pyplot as plt

%matplotlib inline



#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import roc_auc_score

import tensorflow as tf

import timeit



def auroc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
n_layers=64



def build_model(embedding_matrix):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=True)(words)

    x = SpatialDropout1D(0.2)(x)

    x = CuDNNLSTM(n_layers, return_sequences=True)(x)

    x = CuDNNLSTM(n_layers, return_sequences=True)(x)

    x = GlobalMaxPooling1D()(x)

    



    x = Dense(n_layers, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    result = Dense(1, activation='sigmoid')(x)

    

    model = Model(inputs=words, outputs=[result])

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=["accuracy",auroc])

    return model



#dir()
del tokenizer, train, train_data, label_data

gc.collect()
start = time.time()

model = build_model(embedding_matrix)

history = model.fit(x_train, y_train,

                    epochs=6,

                    batch_size=1024,

                    validation_data=(x_val, y_val))



end = time.time(); elapsed = end - start; print(elapsed/60)
def plot_accuracy(acc,val_acc):

  # Plot training & validation accuracy values

  plt.figure()

  plt.plot(acc)

  plt.plot(val_acc)

  plt.title('Model accuracy')

  plt.ylabel('Accuracy')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Test'], loc='upper left')

  plt.show()



def plot_loss(loss,val_loss):

  plt.figure()

  plt.plot(loss)

  plt.plot(val_loss)

  plt.title('Model loss')

  plt.ylabel('Loss')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Test'], loc='upper right')

  plt.show()



def plot_auc(auroc,val_auroc):

  plt.figure()

  plt.plot(auroc)

  plt.plot(val_auroc)

  plt.title('Model AUC')

  plt.ylabel('AUC')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Test'], loc='upper right')

  plt.show()
plot_loss(history.history['loss'], history.history['val_loss'])

plot_accuracy(history.history['acc'], history.history['val_acc'])

plot_auc(history.history['auroc'], history.history['val_auroc'])