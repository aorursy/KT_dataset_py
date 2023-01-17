import numpy as np
import pandas as pd 
import os
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,ENGLISH_STOP_WORDS 
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D, LSTM, MaxPool1D, MaxPooling1D
#from tensorflow.python.keras.layers.embeddings import Embedding
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
Train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
Test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

Train.head()
steam = PorterStemmer()
lem = WordNetLemmatizer()


Train.text = Train.text.apply(lambda text: text.translate(str.maketrans('', '', string.punctuation )))
Test.text = Test.text.apply(lambda text: text.translate(str.maketrans('', '', string.punctuation )))


Train.text = Train.text.apply(lambda text: text.translate(str.maketrans('', '', string.digits )))
Test.text = Test.text.apply(lambda text: text.translate(str.maketrans('', '', string.digits )))

Train.text = Train.text.apply(lambda text: text.lower())
Test.text = Test.text.apply(lambda text: text.lower())


def remove_w(text, filter_list):
    sentence=[]
    for word in text.split(' '):
        if not word in str(filter_list):#ENGLISH_STOP_WORDS
            sentence.append(word)
    
    return ' '.join(sentence)

Train.text = Train.text.apply(lambda text: remove_w(text ,ENGLISH_STOP_WORDS))
Test.text = Test.text.apply(lambda text: remove_w(text ,ENGLISH_STOP_WORDS))



def steaming(text):
    sentence=[]
    for word in text.split(' '):
        sentence.append(steam.stem(word))
    
    return ' '.join(sentence)


#Train.text = Train.text.apply(steaming)
#Test.text = Test.text.apply(steaming)


def lemming(text):
    sentence=[]
    for word in text.split(' '):
        sentence.append(lem.lemmatize(word))
    
    return ' '.join(sentence)


Train.text = Train.text.apply(lemming)
Test.text = Test.text.apply(lemming)

def top_10_freq(text):
    cnt = Counter()
    for sentence in text:
        for word in sentence.split(' '):
            cnt[word]+=1
    
    return cnt.most_common(10)
Top_10_popul=top_10_freq(Train.text)
print(Top_10_popul)

def top_10_unfreq(text):
    cnt = Counter()
    for sentence in text:
        for word in sentence.split(' '):
            cnt[word]+=1
    
    return cnt.most_common()[-10:]
Top_10_unpopul=top_10_unfreq(Train.text)
print(Top_10_unpopul)
        


Train.text = Train.text.apply(lambda text: remove_w(text ,Top_10_popul))
Test.text = Test.text.apply(lambda text: remove_w(text ,Top_10_popul))

Train.text = Train.text.apply(lambda text: remove_w(text ,Top_10_unpopul))
Test.text = Test.text.apply(lambda text: remove_w(text ,Top_10_unpopul))
        
print(sum(Train.keyword.isna())/len(Train.keyword))
print(sum(Train.location.isna())/len(Train.location))
X_train, X_test, y_train, y_test = train_test_split(Train.drop(['id',"target"], axis=1),Train.target,  test_size=0.1, random_state=42)
m_f_imputer = SimpleImputer(strategy = "most_frequent", )
one_hot_loc = OneHotEncoder(sparse = 0, handle_unknown = 'ignore')
one_hot_key = OneHotEncoder(sparse = 0, handle_unknown = 'ignore')
#one_hot_loc.fit(np.array(Train.location.dropna()).reshape(-1,1))
#one_hot_key.fit(np.array(Train.keyword.dropna()).reshape(-1,1))
#
m_f_keyword_train = one_hot_key.fit_transform(m_f_imputer.fit_transform(np.array(X_train.keyword).reshape(-1,1)))
m_f_location_train = one_hot_loc.fit_transform(m_f_imputer.fit_transform(np.array(X_train.location).reshape(-1,1)))


m_f_keyword_pretest = one_hot_key.transform( m_f_imputer.transform(np.array(X_test.keyword).reshape(-1,1)))
m_f_location_pretest = one_hot_loc.transform(m_f_imputer.transform(np.array(X_test.location).reshape(-1,1)))

m_f_keyword_test = one_hot_key.transform( m_f_imputer.transform(np.array(Test.keyword).reshape(-1,1)))
m_f_location_test = one_hot_loc.transform(m_f_imputer.transform(np.array(Test.location).reshape(-1,1)))


print(m_f_location_pretest.shape, m_f_keyword_pretest.shape, m_f_location_train.shape, m_f_keyword_train.shape)
loc_df_train = pd.DataFrame(m_f_keyword_train , columns = ['loc'+str(i) for i in range(m_f_keyword_train.shape[1])])
key_df_train = pd.DataFrame(m_f_location_train , columns = ['key'+str(i) for i in range(m_f_location_train.shape[1])])

DF_train = pd.concat([loc_df_train, key_df_train], axis =1)

loc_df_pretest = pd.DataFrame(m_f_keyword_pretest , columns = ['loc'+str(i) for i in range(m_f_keyword_pretest.shape[1])])
key_df_pretest = pd.DataFrame(m_f_location_pretest , columns = ['key'+str(i) for i in range(m_f_location_pretest.shape[1])])

DF_pretest = pd.concat([loc_df_pretest, key_df_pretest], axis =1)

loc_df_test = pd.DataFrame(m_f_keyword_test , columns = ['loc'+str(i) for i in range(m_f_keyword_test.shape[1])])
key_df_test = pd.DataFrame(m_f_location_test , columns = ['key'+str(i) for i in range(m_f_location_test.shape[1])])

DF_test = pd.concat([loc_df_test, key_df_test], axis =1)
tok = Tokenizer()
tok.fit_on_texts(X_train.text)

text_X_train = tok.texts_to_matrix(X_train.text, mode='count')

text_X_pretest = tok.texts_to_matrix(X_test.text, mode='count')

text_X_test = tok.texts_to_matrix(Test.text, mode='count')
text_X_train.shape, text_X_pretest.shape, text_X_test.shape
DF_train = pd.concat([DF_train, pd.DataFrame(text_X_train)], axis =1)

DF_pretest = pd.concat([DF_pretest, pd.DataFrame(text_X_pretest)], axis =1)

DF_test = pd.concat([DF_test, pd.DataFrame(text_X_test)], axis =1)

DF_train.info()

# random_forest_cls = RandomForestClassifier()
# xg_cls = XGBClassifier()
#print("xg_cls + count",cross_val_score(xg_cls, DF_train, y_train, cv=3, scoring="f1"))
#print("random_forest_cls + count",cross_val_score(random_forest_cls, Proba_count, Train_target_train, cv=3, scoring="f1"))
#print("xg_cls + tfid",cross_val_score(xg_cls, Proba_tfid, Train_target_train, cv=3, scoring="f1"))
#print("random_forest_cls + tfid",cross_val_score(random_forest_cls, Proba_tfid, Train_target_train, cv=3, scoring="f1"))
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
#mirrored_strategy = tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():
   
Model = Sequential()

#     Model.add(Embedding(DF_train.shape[1],5 ,input_length=DF_train.shape[1]))
#     Model.add(Conv1D(32 , 5, activation = 'elu'))
#     Model.add(MaxPool1D(5))
#     Model.add(Dropout(0.6))

#     Model.add(Flatten())

Model.add(Dense(64, input_shape=(DF_train.shape[1],), activation='elu'))

Model.add(Dropout(0.3))

Model.add(Dense(32, activation='elu'))
Model.add(Dropout(0.3))

Model.add(Dense(2, activation='softmax'))
print(Model.summary())
Model.compile(loss="categorical_crossentropy", metrics=['acc', f1], optimizer = Adam(lr=0.000001))

filepath = "best.hdf5"

checkpoint = ModelCheckpoint(filepath,
                            monitor='val_f1',
                            verbose=1,
                            save_best_only=True, 
                            mode='max')
Model.fit(np.array(DF_train).reshape(len(DF_train),-1), to_categorical(y_train), validation_data=(np.array(DF_pretest).reshape(len(DF_pretest),-1), to_categorical(y_test)), epochs=350, callbacks=[checkpoint], batch_size=32)
Model.load_weights(filepath)
pred=[]
for i in Model.predict(np.array(DF_pretest).reshape(len(DF_pretest),-1)):
    pred.append(np.argmax(i))
f1_score(pred, y_test)
Test_id = Test.id
pred=[]
for i in Model.predict(DF_test):
    pred.append(np.argmax(i))
Test_pred = pred
Df_answer= pd.DataFrame()
Df_answer["id"] = Test_id
Df_answer["target"] = pred
Df_answer.to_csv("out2.csv",index=False)