import numpy as np
import pandas as pd
import zipfile

train_archive = zipfile.ZipFile('../input/spooky-author-identification/train.zip', 'r')
test_archive = zipfile.ZipFile('../input/spooky-author-identification/test.zip', 'r')
train = pd.read_csv(train_archive.open("train.csv"))
test = pd.read_csv(test_archive.open("test.csv"))
train.head()
test_id = test['id']
train.info()
train['author'].unique()
print("EAP ",len(train[train['author']=='EAP']))
print("HPL ",len(train[train['author']=='HPL']))
print("MWS ",len(train[train['author']=='MWS']))
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english',decode_error='ignore')

label_enc = LabelEncoder()
y = label_enc.fit_transform(train['author'])

from sklearn.model_selection import train_test_split


xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv = tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)


xtest_tfv = tfv.transform(test['text'])
ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                    ngram_range=(1,3),stop_words='english')

ctv.fit(list(xtrain)+list(xvalid))
xtrain_ctv = ctv.transform(xtrain)
xvalid_ctv = ctv.transform(xvalid)
# Accuracy Mertic
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

model = DummyClassifier()
model.fit(xtrain_tfv,ytrain)
pred = model.predict_proba(xvalid_tfv)

print ("Dummy Classifier TF-IDF logloss: %0.3f " % multiclass_logloss(yvalid, pred))
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=1.0,max_iter=500)
lr_model.fit(xtrain_tfv, ytrain)
prediction = lr_model.predict(xvalid_tfv)
predictions = lr_model.predict_proba(xvalid_tfv) # Results probabilty percentage for each target value.
predictions[:5,:]
print ("Logistic Regression TF-IDF logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
lr_model.fit(xtrain_ctv,ytrain)

ctv_prediction = lr_model.predict_proba(xvalid_ctv)
print ("Logistic Regression CountVec logloss: %0.3f " % multiclass_logloss(yvalid, ctv_prediction))
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()

nb_model.fit(xtrain_tfv,ytrain)
tfv_nb_predictions = nb_model.predict_proba(xvalid_tfv)
print ("Naive Bayes TF-IDF logloss: %0.3f " % multiclass_logloss(yvalid, tfv_nb_predictions))
nb_model.fit(xtrain_ctv, ytrain)
ctv_nb_predictions = nb_model.predict_proba(xvalid_ctv)
print ("Naive Bayes CountVec logloss: %0.3f " % multiclass_logloss(yvalid, ctv_nb_predictions))
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=500, max_depth=7, colsample_bytree=0.8,
                         sumbsample=0.8, n_thread=10, learning_rate=0.1)
xgb_model.fit(xtrain_tfv.tocsc(), ytrain)
xbg_tfv_predictions = xgb_model.predict_proba(xvalid_tfv.tocsc())
print ("XGBoost TF-IDF logloss: %0.3f " % multiclass_logloss(yvalid, xbg_tfv_predictions))
xgb_model.fit(xtrain_ctv.tocsc(), ytrain)
xbg_ctv_predictions = xgb_model.predict_proba(xvalid_ctv.tocsc())
print ("XGBoost CountVec logloss: %0.3f " % multiclass_logloss(yvalid, xbg_ctv_predictions))
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB(alpha=0.1)

nb_model.fit(xtrain_tfv,ytrain)
tfv_nb_predictions = nb_model.predict_proba(xvalid_tfv)
print ("Naive Bayes TF-IDF logloss: %0.3f " % multiclass_logloss(yvalid, tfv_nb_predictions))
final_pred = nb_model.predict_proba(xtest_tfv)
x = {'id':test_id, 'EAP':final_pred[:,0], 'HPL':final_pred[:,1], 'MWS':final_pred[:,2]}
dd = pd.DataFrame(x)
dd.to_csv('spooky_author_NB_submission.csv',index=False)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv) # Truncated SVD works on TF-IDF

xtrain_tfv_svd = svd.transform(xtrain_tfv)
xvalid_tfv_svd = svd.transform(xvalid_tfv)
# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
from sklearn.preprocessing import StandardScaler
scl = StandardScaler()

scl.fit(xtrain_tfv_svd)
xtrain_svd_scl = scl.transform(xtrain_tfv_svd)
xtrain_svd_scl = scl.transform(xvalid_tfv_svd)
xgb_model.fit(xtrain_tfv_svd,ytrain)
predictions = xgb_model.predict_proba(xvalid_tfv_svd)
print ("XGBoost SVD logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print(xtrain_svd_scl.shape)
print(ytrain.shape)
# On scaler datail
xgb_model.fit(xtrain_svd_scl,ytrain)
predictions = xgb_model.predict_proba(xtrain_svd_scl)
print ("XGBoost SVD-SCL logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

model_scorer = make_scorer(score_func = multiclass_logloss,
                           greater_is_better = False,
                           needs_proba=True)

# Defining Pipeline
from sklearn.pipeline import Pipeline

svd = TruncatedSVD()
scl = StandardScaler() 
lr_model = LogisticRegression()

model = Pipeline(steps=[('svd',svd),
                        ('scl',scl),
                         ('model',lr_model)])

hyper_parameters = {
                    'svd__n_components':[120,180],
                    'model__C':[0.1, 1.0, 10],
                    'model__penalty':['l1','l2']
                    }
                
model = GridSearchCV(estimator=model,
                     param_grid=hyper_parameters,
                     scoring=model_scorer,
                     verbose=10,
                     n_jobs= -1,
                     iid = True,
                     refit=True,
                     cv = 2) # Cv = Cross fit split strategy

model.fit(xtrain_tfv, ytrain)

print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")                     
best_hyper_params = model.best_estimator_.get_params()
for param_name in model.param_grid.keys():
  print(f"\t{param_name}: {best_hyper_params[param_name]} ")
nb_model = MultinomialNB()
model = Pipeline([('model',nb_model)])

hyper_parameters = {
                    'model__alpha':[0.001, 0.01, 0.1, 1.0, 10, 100],
                    }
                
grid_nb_model = GridSearchCV(estimator=model,
                     param_grid=hyper_parameters,
                     scoring=model_scorer,
                     verbose=10,
                     n_jobs= -1,
                     iid = True,
                     refit=True,
                     cv = 2) # Cv = Cross fit split strategy

grid_nb_model.fit(xtrain_tfv, ytrain)
print("Best score: %0.3f" % grid_nb_model.best_score_)
print("Best parameters set:")                     
best_hyper_params = grid_nb_model.best_estimator_.get_params()
for param_name in grid_nb_model.param_grid.keys():
  print(f"\t{param_name}: {best_hyper_params[param_name]} ")
xg_model = XGBClassifier()

model = Pipeline(steps=[('svd',svd),
                        ('scl',scl),
                         ('model',xg_model)])

hyper_parameters = {
                    'svd__n_components':[180],
                    'model__max_depth':[3,7,10],
                    'model__learning_rate':[0.01, 0.03, 0.05],
                    'model__n_estimators':[100,300 ,600],
                    'model__n_jobs':[-1],
                    'model__colsample_bytree': [0.3,0.5,0.8],
                    # 'model__subsample':[0.3,0.5,0.8],
                    'model__nthread':[10]

                    }
                
model = GridSearchCV(estimator=model,
                     param_grid=hyper_parameters,
                     scoring=model_scorer,
                     verbose=10,
                     n_jobs= -1,
                     iid = True,
                     refit=True,
                     cv = 2) # Cv = Cross fit split strategy

model.fit(xtrain_tfv, ytrain)

# Load Glove embedding matrix
import pickle
with open(r'../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', 'rb') as handle:
    glove_embeddings = pickle.load(handle)

print('Found %s word vectors.' % len(glove_embeddings))

# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(glove_embeddings[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# create sentence vectors using the above function for training and validation set
xtrain_glove_list = [sent2vec(x) for x in xtrain]
xvalid_glove_list = [sent2vec(x) for x in xvalid]
len(xtrain_glove_list)
# Converting into numpy array 
xtrain_glove = np.array(xtrain_glove_list)
xvalid_glove = np.array(xvalid_glove_list)
type(xtrain_glove)
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=500, max_depth=7, colsample_bytree=0.8,
                         sumbsample=0.8, n_thread=10, learning_rate=0.1)

xgb_model.fit(xtrain_glove,ytrain)
xgb_vector_pred = xgb_model.predict_proba(xvalid_glove)
print ("XGBoost Vector logloss: %0.3f " % multiclass_logloss(yvalid, xgb_vector_pred))
#nb_model = MultinomialNB(alpha=0.1)
#nb_model.fit(xtrain_glove,ytrain)
#nb_vector_pred = nb_model.predict_proba(xvalid_glove)
#print ("Naive Bayes Vector logloss: %0.3f " % multiclass_logloss(yvalid, nb_vector_pred))
# Scaler data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain_glove_sc = sc.fit_transform(xtrain_glove)
xvalid_glove_sc = sc.transform(xvalid_glove)
# Numerical encoding target value
from keras.utils import np_utils
ytrain_enc = np_utils.to_categorical(ytrain)
yvalid_enc = np_utils.to_categorical(yvalid)
ytrain_enc[:5,:]
xtrain_glove_sc.shape
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization


model = Sequential()

model.add(Dense(300, input_dim=300, activation='relu')) # First Layer
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(300, activation='relu')) # Second Layer
model.add(Dropout(0.3))
model.add(BatchNormalization())


model.add(Dense(3)) # Final Layer
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
model.fit(xtrain_glove_sc,ytrain_enc,
          batch_size=64, epochs=20, verbose=1,use_multiprocessing=True,
          validation_data=(xvalid_glove_sc,yvalid_enc))
from keras.preprocessing import sequence,text

tokenizer = text.Tokenizer(num_words=None)
max_len = 70 # max length of sentences we want to keep

tokenizer.fit_on_texts(list(xtrain) + list(xvalid))

xtrain_seq = tokenizer.texts_to_sequences(xtrain) # convert text into numbers
xvalid_seq = tokenizer.texts_to_sequences(xvalid)
xtrain_seq[1]
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_tokens = tokenizer.word_index
embedding_matrix = np.zeros( (len(word_tokens)+1,300) )

for word,i in tqdm(word_tokens.items()):
  word_vector = glove_embeddings.get(word)
  if word_vector is not None:
    embedding_matrix[i] = word_vector

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import SpatialDropout1D
from keras.callbacks import EarlyStopping

model = Sequential()

model.add(Embedding(len(word_tokens)+1,
                    300,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False
                    ))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(300, dropout=0.3,recurrent_dropout=0.3))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')

model.summary()
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(xtrain_pad,y=ytrain_enc,use_multiprocessing=True ,
          batch_size=512, epochs=100, verbose=1,
          validation_data=(xvalid_pad,yvalid_enc),
          callbacks=[earlystop])
from keras.layers import Bidirectional

model = Sequential()

model.add(Embedding(len(word_tokens)+1,
                    300,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False
                    ))

model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(xtrain_pad,y=ytrain_enc,use_multiprocessing=True ,
          batch_size=512, epochs=100, verbose=1,
          validation_data=(xvalid_pad,yvalid_enc),
          callbacks=[earlystop])
from keras.layers.recurrent import GRU

model = Sequential()
model.add(Embedding(len(word_tokens) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(300, dropout=0.8, recurrent_dropout=0.8, return_sequences=True))
model.add(GRU(300, dropout=0.8, recurrent_dropout=0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(xtrain_pad,y=ytrain_enc,use_multiprocessing=True ,
          batch_size=512, epochs=100, verbose=1,
          validation_data=(xvalid_pad,yvalid_enc),
          callbacks=[earlystop])
