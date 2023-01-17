#importing libraries



import pandas as pd

import numpy as np

import xgboost as xgb

from tqdm import tqdm

from sklearn.svm import SVC

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import model_selection, pipeline, metrics, preprocessing, decomposition

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from keras.layers import GlobalMaxPooling1D, MaxPooling1D, Conv1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

train = pd.read_csv("../input/spooky-author-identification/train.csv")

test = pd.read_csv("../input/spooky-author-identification/test.csv")

sample = pd.read_csv("../input/spooky-author-identification/sample_submission.csv")
train.head()

train.shape, test.shape
test.head()
train['id'].unique().shape
sample.head()
##multiclass_logloss

def multiclass_logloss(actual, predicted, eps = 1e-15):

    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes

    :param predicted: Matrix with class predictions, one probability per class

    """

    #converting actual to a binary array 

    if(len(actual.shape)==1):

        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))

        for i, val in enumerate(actual):

            actual2[i,val] = 1

        actual = actual2

        

    clip = np.clip(predicted, eps, 1-eps)

    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(clip))

    return -1.0 / rows *vsota

        
# Label encoder to convert text labels to 0,1,2

lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(train['author'].values)
## train test split



xtrain, xvalid, ytrain, yvalid = train_test_split(train['text'].values, y , 

                                                 stratify = y, shuffle = True,

                                                 test_size = 0.1, random_state = 42)
xtrain.shape, xvalid.shape
## 1st approach

#tfidfvectorizer - creating features

# Convert a collection of raw documents to a matrix of TF-IDF features.

tfv = TfidfVectorizer(min_df = 3, #When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. cut-off

                      strip_accents = 'unicode',#Remove accents and perform other character normalization during the preprocessing step. ‘ascii’ is a fast method that only works on characters that have an direct ASCII mapping. ‘unicode’ is a slightly slower method that works on any characters. None (default) does nothing. 

                      stop_words = 'english',

                      ngram_range = (1,3),

                      analyzer = 'word',

                      token_pattern = r'\w{1,}', # pattern for word, 1 or more words

                      max_features = None, #If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

                      use_idf = 1, # Enable inverse-document-frequency reweighting.

                      smooth_idf = 1, #Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.

                      sublinear_tf = 1 #Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

                     )



tfv.fit_transform(list(xtrain) + list(xvalid))

xtrain_tfv = tfv.transform(xtrain)

xvalid_tfv = tfv.transform(xvalid)
#Logistic regression

clf = LogisticRegression(C=1.0 ) #Inverse of regularization strength..small value means high regularization

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict_proba(xvalid_tfv)



print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## 2nd approach - using word count as features using CountVectorizer



ctv = CountVectorizer(analyzer = 'word', token_pattern =r'\w{1,}', stop_words='english',ngram_range=(1,3))

ctv.fit(list(xtrain) + list(xvalid))

xtrain_ctv = ctv.transform(xtrain)

xvalid_ctv = ctv.transform(xvalid)
## Logistic regression on count features

clf = LogisticRegression(C=1.0)

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict_proba(xvalid_ctv)

print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## 3rd approach - Naive Bayes on tfidf



clf = MultinomialNB()

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict_proba(xvalid_tfv)



print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## 4th approach - Naive Bayes on count vectorizer



clf = MultinomialNB()

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict_proba(xvalid_ctv)



print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## 5th approach - SVM

# SVM takes lot of time so, we need to reduce number of features from tfidf using SVD 

# Need to standardise 



#choosing 120 components for svd, 120 to 200 components are good to go for.



svd = decomposition.TruncatedSVD(n_components= 120)

svd.fit(xtrain_tfv)

xtrain_svd = svd.transform(xtrain_tfv)

xvalid_svd = svd.transform(xvalid_tfv)



scl = preprocessing.StandardScaler()

scl.fit(xtrain_svd)

xtrain_svd_scl = scl.transform(xtrain_svd)

xvalid_svd_scl = scl.transform(xvalid_svd)
##Fitting svm



clf = SVC(C=1.0, #Penalty parameter C of the error term.

          probability= True) # since we need probability

clf.fit(xtrain_svd_scl, ytrain)

predictions = clf.predict_proba(xvalid_svd_scl)

print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## 6th approach - xgboost on tfidf

clf = xgb.XGBClassifier(max_depth = 7,

                       n_estimators = 200, 

                       colsample_bytree = 0.8,

                       subsample = 0.8,

                       nthread = 10,

                       learning_rate = 0.1)

clf.fit(xtrain_tfv.tocsc(), ytrain)

predictions = clf.predict_proba(xvalid_tfv.tocsc())



print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## 7th approach - XGB on count vectorizer



clf = xgb.XGBClassifier(max_depth = 7,

                       n_estimators = 200, 

                       colsample_bytree = 0.8,

                       subsample = 0.8,

                       nthread = 10,

                       learning_rate = 0.1)

clf.fit(xtrain_ctv.tocsc(), ytrain)

predictions = clf.predict_proba(xvalid_ctv.tocsc())



print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## 8th approach - xgb with tfidf - svd features



clf = xgb.XGBClassifier(max_depth = 7,

                       n_estimators = 200, 

                       colsample_bytree = 0.8,

                       subsample = 0.8,

                       nthread = 10,

                       learning_rate = 0.1)

clf.fit(xtrain_svd, ytrain)

predictions = clf.predict_proba(xvalid_svd)



print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## 9th approach - simple xgboost - tfidf - svd



clf = xgb.XGBClassifier( nthread = 10)

                       

clf.fit(xtrain_svd, ytrain)

predictions = clf.predict_proba(xvalid_svd)



print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
## hyperparameter tuning - 10th approach



## Grid Search



##creating scoring function

mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better = False, needs_proba = True)

##pipeline construction with svd, standardscalar, logisticregression



svd = TruncatedSVD()

scl = preprocessing.StandardScaler()

lr_model = LogisticRegression()



clf = pipeline.Pipeline([('svd',svd),

                        ('scl',scl),

                        ('lr_model',lr_model)])



## grid parameters



param_grid = {'svd__n_components' : [120,180],

             'lr_model__C':[0.1,1.0,10],

             'lr_model__penalty':['l1','l2']}
## Grid search model

model = GridSearchCV(estimator = clf, scoring= mll_scorer, param_grid = param_grid,

                    verbose = 10, #Controls the verbosity: the higher, the more messages.

                    n_jobs = -1, #Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

                    iid = True, #If True, return the average score across folds, weighted by the number of samples in each test set.

                    refit = True, #Refit an estimator using the best found parameters on the whole dataset.

                    cv = 2 #Determines the cross-validation splitting strategy.  integer, to specify the number of folds in a (Stratified)KFold,

                    )



model.fit(xtrain_tfv, ytrain)

print("Best score: ", model.best_score_)

print("Best parameter set: ")

best_parameters = model.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):

    print("\t %s : %r " %(param_name, best_parameters[param_name]))
## hyperparameter tuning in Naive bayes - 11th approach

nb_model = MultinomialNB()



# create pipeline

clf = pipeline.Pipeline([('nb',nb_model)])

param_grid = {'nb__alpha' : [0.001, 0.01, 0.1, 1, 10, 100]}



model = GridSearchCV(estimator = clf, scoring = mll_scorer, param_grid = param_grid,

                    verbose = 10, n_jobs = -1, iid = True, refit = True, cv = 2)



model.fit(xtrain_tfv, ytrain)

print("Best score: %0.3f" %model.best_score_)

print("Best parameter set:")

best_parameters = model.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):

    print("\t %s : %r " %(param_name, best_parameters[param_name]))
# word vectors 

#Loading glove embeddings

embedding_index = {}

f = open('../input/glove840b300dtxt/glove.840B.300d.txt')

for line in tqdm(f):

    values = line.split()

    word = values[0]

    try:

        coefs = np.asarray(values[1:], dtype = 'float32')

    except ValueError as v:

        pass

    embedding_index[word] = coefs

f.close()

print("Found %s vectors" %(len(embedding_index)))
## function for creating normalized vectors for sentence



def sent2vec(sent):

    words = str(sent).lower()

    word = word_tokenize(words)

    word = [w for w in word if w not in stop_words]

    word = [w for w in word if w.isalpha()]

    M=[]

    for w in word:

        try:

            M.append(embedding_index[w])

        except:

            continue

    M = np.array(M)

    v = M.sum(axis = 0)

    if type(v) != np.ndarray:

        return np.zeros(300)

    return v / np.sqrt((v**2).sum())
#creating sentence vectors for train and test dataset



xtrain_glove = [ sent2vec(x) for x in tqdm(xtrain)]

xvalid_glove = [sent2vec(x) for x in tqdm(xvalid)]

xtrain_glove = np.array(xtrain_glove)

xvalid_glove = np.array(xvalid_glove)
# - 11th approach -simple xgboost on glove embeddings



clf = xgb.XGBClassifier(nthread = 10, silent = False)

clf.fit(xtrain_glove, ytrain)

predictions = clf.predict_proba(xvalid_glove)

print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
# 12th approach -  Fitting xgboost with parameters to glove embeddings

clf = xgb.XGBClassifier(max_depth =7, n_estimators = 200, colsample_bytree= 0.8, subsample=0.8,

                       nthread =10, learning_rate = 0.1, silent = False)

clf.fit(xtrain_glove, ytrain)

predictions = clf.predict_proba(xvalid_glove)

print("Logloss : %0.3f" %multiclass_logloss(yvalid, predictions))
# Deep learning 

#scale data before neural network



# 13th approach - Dense neural network



scl = preprocessing.StandardScaler()

xtrain_glove_scl = scl.fit_transform(xtrain_glove)

xvalid_glove_scl = scl.fit_transform(xvalid_glove)

##Binarize the output for neural network

ytrain_enc = np_utils.to_categorical(ytrain)

yvalid_enc = np_utils.to_categorical(yvalid)
model = Sequential()



model.add(Dense(300, input_dim = 300, activation='relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Dense(300, activation ='relu'))

model.add(Dropout(0.3))

model.add(BatchNormalization())



model.add(Dense(3))

model.add(Activation('softmax'))



model.compile(loss ='categorical_crossentropy', optimizer='adam')

model.fit(xtrain_glove_scl, y = ytrain_enc, batch_size = 64, epochs = 5, verbose =1,

              validation_data= (xvalid_glove_scl, yvalid_enc))
# 14th approach - LSTM with keras tokenizer



#using keras tokenizer

token = text.Tokenizer(num_words= None)

max_len = 70



token.fit_on_texts(list(xtrain)+list(xvalid))

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)



xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen = max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen = max_len)



word_index = token.word_index
## create embedding matrix

embedding_matrix  = np.zeros((len(word_index)+1, 300))

for word, i in tqdm(word_index.items()):

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
##LSTM with glove embeddings



model = Sequential()

model.add(Embedding(len(word_index)+1 , 300, 

                    weights = [embedding_matrix],

                    trainable = False,

                    input_length = max_len

                   ))

model.add(SpatialDropout1D(0.3))

model.add(LSTM(100, dropout=0.3, recurrent_dropout = 0.3))



model.add(Dense(1024, activation ='relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation ='relu'))

model.add(Dropout(0.8))



model.add(Dense(3))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam')
# commenting this as this will take lot of time,..as next step adding early stopping to stop the iteration if not improvement in validation loss.

#model.fit(xtrain_pad, y = ytrain_enc, batch_size = 512, epochs = 100, verbose = 1, validation_data = (xvalid_pad, yvalid_enc))
## Adding earlystopping in the same model



model = Sequential()

model.add(Embedding(len(word_index)+1 , 300,

                   weights = [embedding_matrix],

                   input_length = max_len,

                   trainable = False))

model.add(SpatialDropout1D(0.3))

model.add(LSTM(300, dropout = 0.3, recurrent_dropout = 0.3))



model.add(Dense(1024, activation ='relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.8))



model.add(Dense(3))

model.add(Activation('softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')



## Fit the model with earlystopping callback



earlystop = EarlyStopping(monitor ='val_loss', min_delta = 0, patience = 3, verbose= 0, mode = 'auto')



model.fit(xtrain_pad, ytrain_enc, batch_size = 512, epochs = 100, validation_data = (xvalid_pad, yvalid_enc),

         callbacks = [earlystop])
## 15th approach- Bidirectional LSTM with glove embeddings 



model = Sequential()

model.add(Embedding(len(word_index)+1, 300,

                   weights = [embedding_matrix],

                   trainable = False, 

                   input_length = max_len))

model.add(SpatialDropout1D(0.3))



model.add(Bidirectional(LSTM(300, dropout = 0.3, recurrent_dropout = 0.3)))

model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation ='relu'))

model.add(Dropout(0.8))



model.add(Dense(3))

model.add(Activation('softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')



earlystop = EarlyStopping(monitor ='val_loss', min_delta = 0, patience =3, verbose =0, mode = 'auto')



model.fit(xtrain_pad, ytrain_enc, batch_size = 512, epochs = 100, verbose =0, callbacks = [earlystop])
## 16th approach - 2 layers of GRU

model = Sequential()

model.add(Embedding(len(word_index)+1, 300,

                   weights = [embedding_matrix],

                   trainable = False, 

                   input_length = max_len))

model.add(SpatialDropout1D(0.3))



model.add(GRU(300, dropout = 0.3, recurrent_dropout = 0.3, return_sequences = True))

model.add(GRU(300, dropout =0.3, recurrent_dropout = 0.3))



model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation ='relu'))

model.add(Dropout(0.8))



model.add(Dense(3))

model.add(Activation('softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')



earlystop = EarlyStopping(monitor ='val_loss', min_delta = 0, patience =3, verbose =0, mode = 'auto')



model.fit(xtrain_pad, ytrain_enc, batch_size = 512, epochs = 100, verbose =0, callbacks = [earlystop])
