import sys, re ,csv, codecs
import numpy as np # linear algebra
import pandas as pd # data processing (CSV)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Dense, Flatten, LSTM, Embedding, Activation
from keras.layers import Input, GlobalMaxPool1D, Dropout, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential

#read data
train_csv = pd.read_csv('../input/kag-data/train.csv')
test_csv = pd.read_csv('../input/kag-data/test.csv')
train_csv.columns
col = train_csv.iloc[:, 2:].sum() # Use of pandas iloc to see the distribution of all x and y values from 2 and forward
row=train_csv.iloc[:,2:].sum(axis=1) # iloc sums x val on axis 1
train_csv['clean'] = (row == 0) # if row == 0 then it is equal to 'clean'
appender = {True : 1, False : 0} # append 1 if true else 0
train_csv["clean"] = train_csv["clean"].map(appender) #mapper for hver x værdi
print(col)
train_csv = train_csv.drop(['toxic'], axis = 1) # drop 'toxic'
train_csv.sample(10)
# Checking null vals in set
print(train_csv.isnull().any())
#label x and y values
X = train_csv['comment_text'] #will be used to train our model on
y = train_csv[['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
X_testv2 = test_csv['comment_text'] #will be used to predict the output labels to see how well our model has trained

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Used because of wierd labeleing in the test set

# Check shape is corresponding to the dataset (x comments, y column)
y_test.shape

max_features = 2000
tokens = Tokenizer(num_words=max_features)
tokens.fit_on_texts(list(X_train))
tokenized_train = tokens.texts_to_sequences(X_train) # Converting to ints
tokenized_test = tokens.texts_to_sequences(X_test)
tokenized_testv2 = tokens.texts_to_sequences(X_testv2)
total = [len(comment) for comment in tokenized_test]
plt.hist(total,bins = np.arange(0,410,10))
plt.show()
print(X_train[4]) # Print comment at index x
print(110 * '_')#print(tokenized_train[2]) # Index x comment tokenized and converted to ints
print(tokenized_train[4])
# Since each vector representation in comments are different, pad sequences because the model is expecting input data of equal size
max_len = 200 # padded squence max length (Changed from 300 to 200)
padded_train = pad_sequences(tokenized_train, maxlen = max_len) # Post padding with zeros
padded_test = pad_sequences(tokenized_test, maxlen = max_len)
padded_testv2 = pad_sequences(tokenized_testv2, maxlen = max_len)
padded_train[:10] # padding starts at the length of sentence and contiunes for max_len
total = [len(comment) for comment in tokenized_test]
plt.hist(total,bins = np.arange(0,410,10))
plt.show()
input_model = Input(shape = (max_len, ))
embed_size = 128
x = Embedding(max_features, embed_size)(input_model)
x = LSTM(60, return_sequences = True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation = 'relu')(x)
x = Dropout(0.1)(x)
x = Dense(6, activation = 'sigmoid')(x)

# Balance weights
class_weight = {0: 1,
                1: 1,
                2: 1.75,
                3: 1,
                4: 1.75,
                5: 1,}
model = Model(inputs = input_model, outputs = x)
model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy']) #compiling the model
model.summary()
# Fit model to dataset
batch_size = 32
epochs = 2
History = model.fit(padded_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1, class_weight = class_weight)
pred = model.predict(padded_test,verbose=1)
metrics = model.evaluate(x = padded_test, y = y_test, verbose = 1)
new_pred = model.predict([padded_testv2],verbose=1)
print(X_testv2[1836])
print(new_pred[1836])
print(X_testv2[48])
print(new_pred[48])
print(X_testv2[1841])
print(new_pred[1841])
print(X_testv2[9424])
print(new_pred[9424])
print(X_testv2[152539])
print(new_pred[152539])
print(X_testv2[152569])
print(new_pred[152569])
types = ['severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate', 'clean']

train_csv[types].describe()
import seaborn as sns
cnt = []
for i in types:
    cnt.append(train_csv[i].sum())
    
y_pos = np.arange(len(cnt))

plt.bar(y_pos, cnt)
plt.xticks(y_pos, types)
plt.show()
print(cnt)
print(train_csv)
print(train_csv.iloc[:, 2:].sum())