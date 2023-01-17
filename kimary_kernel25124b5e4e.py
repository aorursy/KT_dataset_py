!pip install pymorphy2
import numpy as np 
import pandas as pd 
import pymorphy2
import gensim
import itertools
import tensorflow as tf
import xgboost as xgb
from collections import Counter,defaultdict
from pymorphy2 import MorphAnalyzer
from gensim.utils import tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from string import punctuation, digits
import seaborn as sns
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
data.head()
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
valid.head()
valid.lang.unique()
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
test.head()
!pip install googletrans
#from googletrans import Translator
#translator = Translator()

#translated = []
#for idx in range(len(test)):
#     translated.append(translator.translate(test.content[idx], src=test.lang[idx]))
        
#test['translated'] = translated
#test.head()
#test.to_csv('test_translated.csv')
test_translated = pd.read_csv('../input/translated/test_translated.csv')[['content', 'lang', 'translated']]
test_translated.head()
valid_translated = pd.read_csv('../input/translated/validation_with_translation.csv')[['comment_text', 'lang', 'toxic', 'translation']]
valid_translated.head()
morph = MorphAnalyzer()
stops = set(stopwords.words('english'))

def normalize(sent):
    tokens = list(tokenize(sent))
    norm_tokens = [morph.parse(word)[0].normal_form for word in tokens if word and word not in stops]
    return norm_tokens
data['comment_norm'] = [normalize(x) for x in data.comment_text]
test_translated['translated_norm'] = [normalize(x) for x in test_translated.translated]
valid_translated['translated_norm'] = [normalize(x) for x in valid_translated.translation]
data.head()
fast_text = gensim.models.FastText(data.comment_norm, size=50, min_n=4, max_n=8) 
w2v = gensim.models.Word2Vec(data.comment_norm, size=50, sg=1)
def get_embedding(text, model, dim):
    words = Counter(text)
    total = len(text)
    vectors = np.zeros((len(words), dim))
    
    for i,word in enumerate(words):
        try:
            v = model[word]
            vectors[i] = v*(words[word]/total)
        except (KeyError, ValueError):
            continue
    
    if vectors.any():
        vector = np.average(vectors, axis=0)
    else:
        vector = np.zeros((dim))
    
    return vector
def vectorize(data, model, dim=50):
    X = np.zeros((len(data), dim))
    for i, text in enumerate(data.values):
        X[i] = get_embedding(text, model, dim)

    return X
train_X, valid_X, train_y, valid_y = train_test_split(vectorize(data.comment_norm, w2v), data.toxic, random_state=1)
clf = LogisticRegression(C=1000, max_iter=500, class_weight='balanced')
clf.fit(train_X, train_y)
preds = clf.predict_proba(valid_X)
pred = [x[1] for x in preds]
print(roc_auc_score(valid_y, pred))

train_X, valid_X, train_y, valid_y = train_test_split(vectorize(data.comment_norm, fast_text), data.toxic, random_state=1)
clf = LogisticRegression(C=1000,max_iter=500, class_weight='balanced')
clf.fit(train_X, train_y)
preds = clf.predict_proba(valid_X)
pred = [x[1] for x in preds]
print(roc_auc_score(valid_y, pred))
val_vec = vectorize(valid_translated.translated_norm, w2v)
preds_valid = clf.predict_proba(val_vec)
pred = [x[1] for x in preds_valid]
print(roc_auc_score(list(valid_translated.toxic), pred))
val_vec2 = vectorize(valid_translated.translated_norm, fast_text)
preds_valid = clf.predict_proba(val_vec2)
pred = [x[1] for x in preds_valid]
print(roc_auc_score(list(valid_translated.toxic), pred))
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

num_est = [3, 5, 7, 10]
labels = ['AdaBoost (n_est=3)', 'AdaBoost (n_est=5)', 'AdaBoost (n_est=7)', 'AdaBoost (n_est=10)']
for n_est, label in zip(num_est, labels):   
    boosting = AdaBoostClassifier(base_estimator=clf, n_estimators=n_est)   
    boosting.fit(train_X, train_y)
    preds_valid = boosting.predict_proba(valid_X)
    pred = [x[1] for x in preds_valid]
    print(label, roc_auc_score(valid_y, pred))
preds_valid = boosting.predict_proba(val_vec2)
pred = [x[1] for x in preds_valid]
print(roc_auc_score(list(valid_translated.toxic), pred))
clf = xgb.XGBClassifier(objective='binary:logistic')
clf.fit(train_X,  train_y)
preds_valid = clf.predict_proba(valid_X)
pred = [x[1] for x in preds_valid]
print(roc_auc_score(valid_y, pred))
preds_valid = clf.predict_proba(val_vec2)
pred = [x[1] for x in preds_valid]
print(roc_auc_score(list(valid_translated.toxic), pred))
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
lr = LogisticRegression(C=1000, max_iter=500, class_weight='balanced')

clf1 = xgb.XGBClassifier(objective='binary:logistic')
clf2 = AdaBoostClassifier(base_estimator=clf, n_estimators=8)

sclf = StackingClassifier(classifiers=[clf1, clf2], meta_classifier=lr)
labels = ['XGboost', 'AdaBoost', 'Stacking Classifier']
clf_list = [clf1, clf2, sclf]

for clf, label in zip(clf_list, labels):
        
    scores = cross_val_score(clf, train_X, train_y, cv=3, scoring='roc_auc')
    print ("Roc Auc: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

    clf.fit(train_X, train_y)
    
    preds_valid = clf.predict_proba(valid_X)
    pred = [x[1] for x in preds_valid]
    print(label, roc_auc_score(valid_y, pred))
def prepareEmbeddings(texts):
    vocab = Counter()
    for text in texts:
        vocab.update(text)
        
    filtered_vocab = set()
    for word in vocab:
        if vocab[word] > 5:
            filtered_vocab.add(word)
    
    word2id = {'UNK':1, 'PAD':0}
    for word in filtered_vocab:
        word2id[word] = len(word2id)
    
    id2word = {i:word for word, i in word2id.items()}

    X = []
    for text in texts:
        tokens = text
        ids = [word2id.get(token, 1) for token in tokens]
        X.append(ids)
    
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=200)
    return X, word2id
X_train, word2id = prepareEmbeddings(data.comment_norm)
y_train = np.array(data.toxic)
X_valid, valword2id = prepareEmbeddings(valid_translated.translated_norm)
y_valid = np.array(valid_translated.toxic)
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.weights', # названия файла 
                                                monitor='val_auc', # за какой метрикой следить
                                                verbose=1, # будет печатать что происходит
                                                save_weights_only=True, # если нужно только веса сохранить
                                                save_best_only=True, # сохранять только лучшие
                                                mode='max', # если метрика должна расти, то тут max и min если наоборот
                                                save_freq='epoch' # как часто вызывать
                                               )

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', 
                                              min_delta=0.01, # какая разница считается как улучшение
                                              patience=2, # сколько эпох терпеть отсутствие улучшений
                                              verbose=1, 
                                              mode='max',
                                              )
inputs = tf.keras.layers.Input(shape=(200,))
embeddings = tf.keras.layers.Embedding(input_dim=len(word2id), output_dim=100)(inputs, )

drop1 = tf.keras.layers.Dropout(0.4)(embeddings)
conv1 = tf.keras.layers.Conv1D(kernel_size=3, filters=32, strides=1, kernel_regularizer='l2', activation='relu')(drop1)
conv2 = tf.keras.layers.Conv1D(kernel_size=5, filters=32, strides=2, kernel_regularizer='l2', activation='relu')(conv1)
pool = tf.keras.layers.AveragePooling1D()(conv2)
drop2 = tf.keras.layers.Dropout(0.3)(pool)

flatten = tf.keras.layers.Flatten()(drop2)
dense = tf.keras.layers.Dense(50, activation='relu')(flatten)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics='AUC')
model.fit(X_train, y_train, 
          validation_data=(X_valid, y_valid),
          batch_size=2000,
          epochs=5,
          callbacks=[checkpoint, early_stop])
print(model.history.history.keys())
plt.plot(model.history.history['auc'])
plt.plot(model.history.history['val_auc'])
plt.title('model f1')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
inputs = tf.keras.layers.Input(shape=(200,))
embeddings = tf.keras.layers.Embedding(input_dim=len(word2id), output_dim=100)(inputs, )

# kernel_size = 3
pad1 = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [0, 0]], mode='REFLECT'))(embeddings)

conv1 = tf.keras.layers.Conv1D(kernel_size=3, filters=32, strides=1)(pad1)
drop1 = tf.keras.layers.Dropout(0.3)(conv1)

pad2 = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [0, 0]], mode='REFLECT'))(drop1)

conv2 = tf.keras.layers.Conv1D(kernel_size=3, filters=32,strides=1, kernel_regularizer='l2', activation='relu')(pad2)
pool1 = tf.keras.layers.AveragePooling1D()(conv2)
conv3 = tf.keras.layers.Conv1D(kernel_size=3, filters=32,strides=1, kernel_regularizer='l2', activation='relu')(pool1)

#kernel_size = 5
pad3 = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0], [2,2], [0, 0]],mode='REFLECT'))(embeddings)

conv4 = tf.keras.layers.Conv1D(kernel_size=5, filters=32, strides=1)(pad3)
drop2 = tf.keras.layers.Dropout(0.3)(conv4)

pad4 = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0], [2,2], [0, 0]],mode='REFLECT'))(drop2)

conv5 = tf.keras.layers.Conv1D(kernel_size=5, filters=32,strides=1, kernel_regularizer='l2', activation='relu')(pad4)
pool2 = tf.keras.layers.AveragePooling1D()(conv5)
conv6 = tf.keras.layers.Conv1D(kernel_size=3, filters=32,strides=1, kernel_regularizer='l2', activation='relu')(pool2)


concat = tf.keras.layers.concatenate([conv3, conv6])
drop3 = tf.keras.layers.Dropout(0.5)(concat)

conv_global = tf.keras.layers.Conv1D(kernel_size=5, filters=32, strides=1)(drop3)
flatten = tf.keras.layers.Flatten()(conv_global)
dense = tf.keras.layers.Dense(50, activation='relu')(flatten)
dense2 = tf.keras.layers.Dense(25, activation='relu')(dense)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics='AUC')
model.fit(X_train, y_train, 
          validation_data=(X_valid, y_valid),
          batch_size=2000,
          epochs=10,
          callbacks=[checkpoint, early_stop])
print(model.history.history.keys())
plt.plot(model.history.history['auc'])
plt.plot(model.history.history['val_auc'])
plt.title('model f1')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
test_dataset, word2idtest = prepareEmbeddings(test_translated.translated)
test_dataset
idx = list(test_translated.index)
s = model.predict(test_dataset, verbose=1)
s = [float(x) for x in s]
s
sub = pd.DataFrame(zip(idx, s), columns=['id', 'toxic'])
sub
sub.to_csv('submission.csv', index=False)