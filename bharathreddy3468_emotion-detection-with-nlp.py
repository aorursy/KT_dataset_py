def fff(x):

  with open(x) as file: 

    line = []

    for lines in file.readlines():

      line.append(lines)

    return line

line = fff('../input/emotions-dataset-for-nlp/train.txt')
line[0:5]
import pandas as pd
import re
def csv(line):

  list1,list2 = [],[]

  for lines in line:

    x,y = lines.split(';')

    y = y.replace('\n','')

    list1.append(x)

    list2.append(y)

  df = pd.DataFrame(list(list1),columns=['sentence'])

  df['emotion'] = list2

  return df
df = csv(line)
df
df.emotion.value_counts()
df.isnull().sum()
import nltk

nltk.download('wordnet')

nltk.download('stopwords')

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords
wn = WordNetLemmatizer()
def lem(x):

  corpus = []

  i=1

  for words in x:

    words = words.split()

    y = [wn.lemmatize(word) for word in words if not word in stopwords.words('english')]

    y =  ' '.join(y)

    corpus.append(y)

  return corpus

x = lem(df['sentence'])
x[:5]
test_line = fff('../input/emotions-dataset-for-nlp/train.txt') 
test_df = csv(test_line)
test_df[:5]
x_test = lem(test_df['sentence'])
all = x + x_test
len(all)
y = df.iloc[:,1].values
y.shape
y_test = test_df.iloc[:,1].values
y_test.shape
from tensorflow.keras.layers import Embedding,LSTM,Dense

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
y_train = pd.DataFrame(y)
tokenizer = Tokenizer(nb_words=10000, split=' ')

tokenizer.fit_on_texts(all)

X1 = tokenizer.texts_to_sequences(all)

X1 = pad_sequences(X1,maxlen=20,padding='post',truncating='post')

Y1 = pd.get_dummies(y_train).values
X_train = X1[:16000]

X_test = X1[16000:]
Y_train = Y1
Y_test = pd.get_dummies(y_test).values
model = Sequential()

model.add(Embedding(input_dim=10000,output_dim = 64,input_length=20))

model.add(LSTM(64))

model.add(Dense(6,activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=32,epochs=10,verbose=2,validation_split=0.2)
loss,acc = model.evaluate(X_test,Y_test)
preds = model.predict(X_test,Y_test)