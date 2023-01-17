import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

from tqdm import tqdm

%matplotlib inline
df = pd.read_csv('../input/first-gop-debate-twitter-sentiment/Sentiment.csv')

df.head()
data = df[['text','sentiment']]

data = data[data.sentiment != 'Neutral']

data.head()
data['text'] = data['text'].apply(lambda x: x.lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-z0-9\s]','',x)))

data.head()
for idx,row in tqdm(data.iterrows()):

    row[0] = row[0].replace('rt ',' ')



data.head()   
print('Positive: ',data[ data['sentiment'] == 'Positive'].size)

print('Negative: ',data[ data['sentiment'] == 'Negative'].size)

x = data['text']

y = data['sentiment']

plt.hist(y,bins=3)

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score





y = LabelEncoder().fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42)
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.8)

v_train=vectorizer.fit_transform(x_train) 

v_test=vectorizer.transform(x_test)
pac=PassiveAggressiveClassifier(C=0.01,random_state=42)

pac.fit(v_train,y_train)



y_pred=pac.predict(v_test)

score=accuracy_score(y_test,y_pred)

print(f'Accuracy: {round(score*100,2)}%')
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for i in tqdm(range(len(x_test))):

   

    if y_pred[i] == y_test[i]:

        if y_test[i] == 1:

            neg_correct += 1

        else:

            pos_correct += 1

       

    if y_pred[i] == 1:

        neg_cnt += 1

    else:

        pos_cnt += 1







print(f"Positive Accuracy {round(pos_correct/pos_cnt*100,2)} %")

print(f"Negative Accuracy {round(neg_correct/neg_cnt*100,2)} %")
cmnt = ["I hate summer because it is so sweaty",

        "I love walking in the park at sunset"]

cmnt = vectorizer.transform(cmnt)

pred = pac.predict(cmnt)

for p in pred:

    if p == 0:

        print('Negative')

    else:

        print('Positive')
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.utils.np_utils import to_categorical
max_fatures = 3000

tokenizer = Tokenizer(num_words=max_fatures)

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
embed_dim = 128

lstm_out = 128



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(LSTM(lstm_out))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.summary()
batch_size = 64

epochs = 10

model.fit(X_train, Y_train, epochs = epochs, batch_size=batch_size)
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc*100),"%")
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for i in tqdm(range(len(X_test))):

    

    result = model.predict(X_test[i].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]

   

    if round(result[0]) == Y_test[i]:

        if Y_test[i] == 0:

            neg_correct += 1

        else:

            pos_correct += 1

       

    if Y_test[i] == 0:

        neg_cnt += 1

    else:

        pos_cnt += 1







print(f"Positive Accuracy {round(pos_correct/pos_cnt*100,2)} %")

print(f"Negative Accuracy {round(neg_correct/neg_cnt*100,2)} %")
cmnt = ["I hate summer because it is so sweaty",

        "I love walking in the park at sunset"]



cmnt = tokenizer.texts_to_sequences(cmnt)

cmnt = pad_sequences(cmnt, maxlen=29, dtype='int32', value=0)



for p in cmnt:

    sentiment = model.predict(np.expand_dims(p,axis=0),batch_size=1,verbose = 2)[0]

    if(np.round(sentiment[0]) == 0):

        print("negative")

    elif (np.round(sentiment[0]) == 1):

        print("positive")