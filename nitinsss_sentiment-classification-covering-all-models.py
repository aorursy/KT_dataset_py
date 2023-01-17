import requests



url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

target_path = 'aclImdb_v1.tar.gz'



response = requests.get(url, stream=True)

if response.status_code == 200:

    with open(target_path, 'wb') as f:

        f.write(response.raw.read())
import tarfile

tar = tarfile.open("aclImdb_v1.tar.gz")

tar.extractall()

tar.close()
import glob 

classes = {'pos':1, 'neg':0}



def read_txt(file_path):

  with open(file_path, 'r') as file:

    text = file.read()

  file.close()

  return text



def populate(main_folder):

  all_txts, all_sentiments = [], []

  for class_ in classes:

    directory = "aclImdb/{}/{}".format(main_folder, class_)

    file_paths = glob.glob(directory + '/*.txt')

    txts = [read_txt(path) for path in file_paths]

    sentiments = [classes[class_] for _ in range(len(txts))]

    all_txts.extend(txts)

    all_sentiments.extend(sentiments)

  return all_txts, all_sentiments
X_train, y_train = populate('train')

X_test, y_test = populate('test')
print(len(X_train))

print(len(X_test))
print(X_train[5])
import re



extra_chars = re.compile("[0-9.;:!\'?,%\"()\[\]]")

html_tags = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")



def clean(texts):

    texts = [extra_chars.sub("", text.lower()) for text in texts]

    texts = [html_tags.sub(" ", text) for text in texts]

    return texts
X_train = clean(X_train)

X_test = clean(X_test)
import random



def shuffle_set(X, y):

  all_data = list(zip(X, y))

  random.shuffle(all_data)

  X_shuff, y_shuff = [list(item) for item in zip(*all_data)]

  return X_shuff, y_shuff



X_train, y_train = shuffle_set(X_train, y_train)

X_test, y_test = shuffle_set(X_test, y_test)
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english')) 



def filter_text(text):

  words = text.split()

  return ' '.join([w for w in words if w not in stop_words])
X_train = [filter_text(text) for text in X_train]

X_test = [filter_text(text) for text in X_test]
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer() 



def lemmatize(text):

  words = text.split()

  return ' '.join([lemmatizer.lemmatize(w) for w in words])
X_train = [lemmatize(text) for text in X_train]

X_test = [lemmatize(text) for text in X_test]
print(X_train[0])
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(binary=True)



vectorizer.fit(X_train)



X_train_onehot = vectorizer.transform(X_train)



X_test_onehot = vectorizer.transform(X_test)
print(X_train_onehot.shape)

print(X_test_onehot.shape)
word_dict = vectorizer.vocabulary_



print({k: word_dict[k] for k in list(word_dict)[:20]})
from sklearn.metrics import accuracy_score, f1_score
def fit_and_test(classifier, X_train, y_train, X_test, y_test, only_return_accuracy=False):



  classifier.fit(X_train, y_train)



  y_hat = classifier.predict(X_test)



  print('accuracy:', accuracy_score(y_test, y_hat))



  if not only_return_accuracy:

    print('f1_score:', f1_score(y_test, y_hat))
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

fit_and_test(mnb, X_train_onehot, y_train, X_test_onehot, y_test)
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

fit_and_test(bnb, X_train_onehot, y_train, X_test_onehot, y_test)
from sklearn.linear_model import LogisticRegression
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:



  lr = LogisticRegression(C=c, max_iter=1000)



  print (f'At C = {c}:-', end=' ')



  fit_and_test(lr, X_train_onehot, y_train, X_test_onehot, y_test, True)
lr_best = LogisticRegression(C=0.05, max_iter=1000)

fit_and_test(lr_best, X_train_onehot, y_train, X_test_onehot, y_test)
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()



fit_and_test(sgd, X_train_onehot, y_train, X_test_onehot, y_test)
from sklearn.neighbors import KNeighborsClassifier



neighbours = [10, 20, 50, 100, 500]



for k in neighbours:



  knn = KNeighborsClassifier(n_neighbors=k)



  print (f'At K = {k}:-', end=' ')



  fit_and_test(knn, X_train_onehot, y_train, X_test_onehot, y_test, True)
knn_best = KNeighborsClassifier(n_neighbors=50)



fit_and_test(knn_best, X_train_onehot, y_train, X_test_onehot, y_test)
from sklearn.svm import LinearSVC
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:



  svc = LinearSVC(C=c, max_iter=5000)



  print (f'At C = {c}:-', end=' ')



  fit_and_test(svc, X_train_onehot, y_train, X_test_onehot, y_test, True)
from sklearn.metrics import classification_report
svc_best = LinearSVC(max_iter=5000, C=0.01)



svc_best.fit(X_train_onehot, y_train)

y_hat = svc_best.predict(X_test_onehot)



report = classification_report(y_test, y_hat, output_dict=True)



print('positive: ', report['1'])

print('negative: ', report['0'])
vectorizer = CountVectorizer(binary=False)



vectorizer.fit(X_train)



X_train_wc = vectorizer.transform(X_train)



X_test_wc = vectorizer.transform(X_test)
mnb = MultinomialNB()

fit_and_test(mnb, X_train_wc, y_train, X_test_wc, y_test)
lr = LogisticRegression(C=0.05, max_iter=1000)

fit_and_test(lr, X_train_wc, y_train, X_test_wc, y_test)
svc = LinearSVC(max_iter=5000, C=0.01)

fit_and_test(svc, X_train_wc, y_train, X_test_wc, y_test)
vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))



vectorizer.fit(X_train)



X_train_2gram = vectorizer.transform(X_train)



X_test_2gram = vectorizer.transform(X_test)
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:



  lr = LogisticRegression(C=c, max_iter=1000)



  print (f'At C = {c}:-', end=' ')



  fit_and_test(lr, X_train_2gram, y_train, X_test_2gram, y_test, True)
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:



  svc = LinearSVC(C=c, max_iter=5000)



  print (f'At C = {c}:-', end=' ')



  fit_and_test(svc, X_train_2gram, y_train, X_test_2gram, y_test, True)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()

vectorizer.fit(X_train)



X_train_tf = vectorizer.transform(X_train)



X_test_tf = vectorizer.transform(X_test)
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:



  lr = LogisticRegression(C=c, max_iter=1000)



  print (f'At C = {c}:-', end=' ')

  

  fit_and_test(lr, X_train_tf, y_train, X_test_tf, y_test, True)
for c in [0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1]:



  svc = LinearSVC(C=c, max_iter=5000)



  print (f'At C = {c}:-', end=' ')



  fit_and_test(svc, X_train_tf, y_train, X_test_tf, y_test, True)
from tensorflow.keras.preprocessing.text import Tokenizer



MAX_NUM_WORDS = 5000



tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

X_train.extend(X_test)

tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index



print([(w, i) for w, i in word_index.items()] [:20])
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
batch_size = 128

l = len(X_train)



i = 0

while (i <= l-1):



  if (i + batch_size) >= (l-1):

    X_train[i:] = tokenizer.texts_to_sequences(X_train[i:])

  

  else:

    X_train[i:i+batch_size] = tokenizer.texts_to_sequences(X_train[i:i+batch_size])

  

  i += batch_size



X_train, X_test = X_train[:l//2], X_train[l//2:]
print(X_train[10])
import matplotlib.pyplot as plt



seq_lengths = [len(seq) for seq in X_train]



plt.figure(figsize=(10, 4))

plt.hist(seq_lengths)

plt.show()
from tensorflow.keras.preprocessing.sequence import pad_sequences



MAX_SEQ_LEN = 120



X_train = pad_sequences(X_train, maxlen=MAX_SEQ_LEN, padding='pre', truncating='pre')



X_test = pad_sequences(X_test, maxlen=MAX_SEQ_LEN, padding='pre', truncating='pre')
print(X_train.shape)

print(X_test.shape)
print(X_train)
import numpy as np

y_train = np.array(y_train)

y_test = np.array(y_test)
#Embedding matrix first dimension

V = num_words



#Embedding matrix second dimension

D = 50



#Hidden state length

M = 100



#Number of steps

T = MAX_SEQ_LEN
from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.layers import Embedding, LSTM

from tensorflow.keras.models import Model
i = Input(shape=(T,))

x = Embedding(V, D)(i)

x = LSTM(M)(x)

x = Dense(1, activation='sigmoid')(x)



model = Model(i, x)

model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import load_model
def train_and_test(model, label, batch_size, epochs):



  save_at = label + ".hdf5"



  save_best = ModelCheckpoint(save_at, monitor='val_loss', verbose=1, 

                              save_best_only=True, save_weights_only=False, mode='min')



  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



  s = len(X_test)//2



  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 

            validation_data=(X_test[:s], y_test[:s]), callbacks=[save_best])



  trained_model = load_model(save_at)

  y_hat = model.predict(X_test[s:])

  y_hat  = (y_hat > 0.5)*1



  print('\n')

  print('-'*100)

  print(f"Test Results for '{save_at}' model")

  print('accuracy:', accuracy_score(y_test[s:], y_hat))

  print('f1_score:', f1_score(y_test[s:], y_hat))
train_and_test(model, 'simple_lstm', batch_size=128, epochs=3)
from tensorflow.keras.layers import GlobalAveragePooling1D
i = Input(shape=(T,))

x = Embedding(V, D)(i)

x = LSTM(M, return_sequences=True)(x)



x = GlobalAveragePooling1D()(x)

x = Dense(1, activation='sigmoid')(x)



model = Model(i, x)

model.summary()
train_and_test(model, 'lstm_all_hidden_states', batch_size=128, epochs=5)
from tensorflow.keras.layers import Bidirectional
i = Input(shape=(T,))

x = Embedding(V, D)(i)



x = Bidirectional(LSTM(M, return_sequences=True))(x)



x = GlobalAveragePooling1D()(x)

x = Dense(1, activation='sigmoid')(x)



model = Model(i, x)

model.summary()
train_and_test(model, 'lstm_bidirectional', batch_size=128, epochs=5)
from tensorflow.keras.layers import Conv1D

from tensorflow.keras.layers import MaxPooling1D

from tensorflow.keras.layers import BatchNormalization
i = Input(shape=(T,))

x = Embedding(V, D)(i)



x = Conv1D(16, 3, activation='relu')(x)

x = MaxPooling1D()(x)

x = BatchNormalization()(x)



x = Conv1D(16, 3, activation='relu')(x)

x = MaxPooling1D()(x)

x = BatchNormalization()(x)



x = Conv1D(16, 3, activation='relu')(x)

x = MaxPooling1D()(x)

x = BatchNormalization()(x)



x = GlobalAveragePooling1D()(x)

x = Dense(1, activation='sigmoid')(x)



model = Model(i, x)

model.summary()
train_and_test(model, 'cnn', batch_size=128, epochs=8)
! mkdir glove
# import zipfile, io



# data_url = 'http://nlp.stanford.edu/data/glove.6B.zip'

# r = requests.get(data_url)

# z = zipfile.ZipFile(io.BytesIO(r.content))

# z.extractall('glove/')
! conda install -y gdown
import gdown



url = "https://drive.google.com/uc?id=18WgSks6St7KVDgY4Y2e29dHhEcD-9SWK"



output = 'glove/glove.6B.100d.txt'



gdown.download(url, output, quiet=False)
EMBEDDING_DIM = 100



embeddings_index = {}

with open('glove/glove.6B.100d.txt') as f:

  for line in f:

    word, coeff = line.split(maxsplit=1)

    coeff = np.fromstring(coeff, 'f', sep=' ')

    embeddings_index[word] = coeff
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))  #(num_words, length of each word embedding)



for word, i in word_index.items():

  if i >= num_words:

    continue

  embedding_vector = embeddings_index.get(word)

  if embedding_vector is not None:               # words not found in embedding index will be all-zeros.

    embedding_matrix[i] = embedding_vector
from tensorflow.keras.initializers import Constant
#Embedding matrix second dimension

D = EMBEDDING_DIM
i = Input(shape=(T,))



x = Embedding(V, D, 

              embeddings_initializer=Constant(embedding_matrix),

              trainable=False)(i)



x = LSTM(M)(x)

x = Dense(1, activation='sigmoid')(x)



model = Model(i, x)

model.summary()
train_and_test(model, 'lstm_glove', batch_size=128, epochs=10)
i = Input(shape=(T,))



x = Embedding(V, D, 

              embeddings_initializer=Constant(embedding_matrix),

              trainable=True)(i)



x = LSTM(M)(x)

x = Dense(1, activation='sigmoid')(x)



model = Model(i, x)

model.summary()
train_and_test(model, 'lstm_glove_trainable', batch_size=128, epochs=10)