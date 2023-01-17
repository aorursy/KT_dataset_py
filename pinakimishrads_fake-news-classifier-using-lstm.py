import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_data = train.copy()
test_data = test.copy()
train_data.head()
train_data = train_data.set_index('id', drop = True)
print(train_data.shape)
train_data.head()
print(test_data.shape)
test_data.head()
# checking for missing values
train_data.isnull().sum()
# dropping missing values from text columns alone. 
train_data[['title', 'author']] = train_data[['title', 'author']].fillna(value = 'Missing')
train_data = train_data.dropna()
train_data.isnull().sum()
length = []
[length.append(len(str(text))) for text in train_data['text']]
train_data['length'] = length
train_data.head()
min(train_data['length']), max(train_data['length']), round(sum(train_data['length'])/len(train_data['length']))
len(train_data[train_data['length'] < 50])
train_data['text'][train_data['length'] < 50]
# dropping the outliers
train_data = train_data.drop(train_data['text'][train_data['length'] < 50].index, axis = 0)
min(train_data['length']), max(train_data['length']), round(sum(train_data['length'])/len(train_data['length']))
##importing tensorflow and looking into the version of it
import tensorflow as tf
tf.__version__
## import all necessaries
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout, RNN, SpatialDropout1D
## voc size
voc_size =  4500
tokenizer = Tokenizer(num_words = voc_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
tokenizer.fit_on_texts(texts = train_data['text'])
X = tokenizer.texts_to_sequences(texts = train_data['text'])
# now applying padding to make them even shaped.
X = pad_sequences(sequences = X, maxlen = voc_size, padding = 'pre')
print(X.shape)
y = train_data['label'].values
print(y.shape)
# splitting the data training data for training and validation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)
##creating model
embdding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embdding_vector_features, input_length = 20))
model.add(LSTM(100))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics =  ['accuracy'])
print(model.summary())
## model training 
model_fit = model.fit(X_train, y_train, epochs = 1)
print(test.shape)
test_data = test.copy()
print(test_data.shape)
test_data = test_data.set_index('id', drop = True)
test_data.shape
test_data = test_data.fillna(' ')
print(test_data.shape)
test_data.isnull().sum()
tokenizer.fit_on_texts(texts = test_data['text'])
test_text = tokenizer.texts_to_sequences(texts = test_data['text'])
test_text = pad_sequences(sequences = test_text, maxlen = voc_size, padding = 'pre')

pred = model.predict_classes(test_text)
pred

pred = pred[:, 0]
submission = pd.DataFrame({'id':test_data.index, 'label':pred})
submission.shape
pred
submission.head()
submission.to_csv('submission.csv', index = False)
