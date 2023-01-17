import pandas as pd
import numpy as np
train = pd.read_json("../input/whats-cooking-kernels-only/train.json")
#test  = pd.read_json("../input/whats-cooking-kernels-only/test.json")
sample_sub = pd.read_csv("../input/whats-cooking-kernels-only/sample_submission.csv")
#common_ingredients = ['salt', 'onions', 'olive oil', 'water', 'garlic', 'sugar','garlic cloves', 'butter', 'ground black pepper', 'all-purpose flour']
# def remove_common(list):
#   new_words = [word for word in list if word not in common_ingredients]
#   return new_words
def sort(list):
  return sorted(list)
#test['ingredients'] = test[["ingredients"]].apply(lambda x: remove_common(*x), axis=1)
y_train = train["cuisine"]
X_train = train.drop(["cuisine","id"],axis =1)
#X_train['ingredients'] = X_train[["ingredients"]].apply(lambda x: remove_common(*x), axis=1)
X_train['ingredients'] = X_train[["ingredients"]].apply(lambda x: sort(*x), axis=1)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
from keras.utils import to_categorical
# y_train = to_categorical(y_train)
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score
from keras.optimizers import Adam, RMSprop, Adagrad,Adadelta,SGD
from keras.layers import Flatten
from keras.layers.merge import concatenate
from sklearn.model_selection import StratifiedKFold
MAX_SEQUENCE_LENGTH = 200
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.20
BATCH_SIZE = 256
EPOCHS = 20
import numpy as np
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    word2vec = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        word2vec[word] = embedding
    print ("Done.",len(word2vec)," words loaded!")
    return word2vec
word2vec = loadGloveModel("../input/glove6b200d/glove.6B.200d.txt")
X_train['liststring'] = X_train['ingredients'].apply(lambda x: ' '.join(map(str, x)))
X_train.loc[X_train['liststring'].str.len() ==0,'liststring'] = "DUMMY_VALUE"
sentences = X_train['liststring'].fillna("DUMMY_VALUE").values
print("max sequence length:", max(len(s) for s in sentences))
print("min sequence length:", min(len(s) for s in sentences))
s = sorted(len(s) for s in sentences)
print("median sequence length:", s[len(s) // 2])

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word2idx = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# test['liststring'] = test['ingredients'].apply(lambda x: ' '.join(map(str, x)))
# test.loc[test['liststring'].str.len() ==0,'liststring'] = "DUMMY_VALUE"
# sentences_test = test["liststring"].fillna("DUMMY_VALUE").values
# tokenizer.fit_on_texts(sentences_test)
# sequences = tokenizer.texts_to_sequences(sentences_test)
# data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=True
)
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(data, y_train):
#     y_train = to_categorical(y_train)
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    #1
    embedding1 = embedding_layer(input_)
    conv1 = Conv1D(filters=200, kernel_size=1, activation='relu')(embedding1)
    #drop1 = Dropout(0.8)(conv1)
    pool1 = GlobalMaxPooling1D()(conv1)
    #flat1 = Flatten()(pool1)
    #2
    #embedding2 = embedding_layer(input_)
    conv2 = Conv1D(filters=200, kernel_size=2, activation='relu')(embedding1)
    #drop2 = Dropout(0.8)(conv2)
    pool2 = GlobalMaxPooling1D()(conv2)
    #flat2 = Flatten()(pool2)
    #3
    #embedding3 = embedding_layer(input_)
    conv3 = Conv1D(filters=200, kernel_size=3, activation='relu')(embedding1)
    #drop3 = Dropout(0.8)(conv3)
    pool3 = GlobalMaxPooling1D()(conv3)
    #flat3 = Flatten()(pool3)
    #merge
    merged = concatenate([pool1, pool2, pool2])
    drop = Dropout(0.8)(merged)
    #interpretation
    dense1 = Dense(60, activation='relu')(drop)
    output = Dense(20, activation='softmax')(dense1)

    model = Model(input_, output)
    adam = Adam(lr = 0.0001)
    model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer="Adam",
      metrics=['accuracy']
    )

    r = model.fit(
      data[train],
      y_train[train],
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      validation_split=VALIDATION_SPLIT,
      verbose=0
    )
    
    scores = model.evaluate(data[test], y_train[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()
test  = pd.read_json("../input/whats-cooking-kernels-only/test.json")
test['ingredients'] = test[["ingredients"]].apply(lambda x: sort(*x), axis=1)
test['liststring'] = test['ingredients'].apply(lambda x: ' '.join(map(str, x)))
test.loc[test['liststring'].str.len() ==0,'liststring'] = "DUMMY_VALUE"
sentences_test = test["liststring"].fillna("DUMMY_VALUE").values
tokenizer.fit_on_texts(sentences_test)
sequences = tokenizer.texts_to_sequences(sentences_test)
data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
p_test = model.predict(data_test)
p_test[0]
test["cuisine"] = pd.DataFrame(le.inverse_transform(np.argmax(p_test, axis=1)))
submition = test.drop(["ingredients","liststring"],axis = 1)
submition.to_csv('submission.csv', index=False)
