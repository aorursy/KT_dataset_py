# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM,GRU,SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re
data.head()
data.rename(columns = {'text':'tweet','airline_sentiment':'senti'},inplace = True)

data = data[['tweet','senti']]

data['tweet'] = data['tweet'].apply(lambda x: x.lower())

data['tweet'] = data['tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
data.head()
data['senti'].value_counts()
data['pos_neg'] = 0

data['pos_neu'] = 0

data['neu_neg'] = 0

for i,v in data.iterrows():

    if v['senti'] == 'neutral':

        data.loc[i,'pos_neg'] = 1

        data.loc[i,'pos_neu'] = 0

        data.loc[i,'neu_neg'] = 1

    if v['senti'] == 'positive':

        data.loc[i,'pos_neg'] = 1

        data.loc[i,'pos_neu'] = 1

        data.loc[i,'neu_neg'] = 1

    if v['senti'] == 'negative':

        data.loc[i,'pos_neg'] = 0

        data.loc[i,'pos_neu'] = 0

        data.loc[i,'neu_neg'] = 0
data.head()
max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data['tweet'].values)

X = tokenizer.texts_to_sequences(data['tweet'].values)

X = pad_sequences(X)





embed_dim = 128

lstm_out = 196

from keras.layers import Bidirectional



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(SpatialDropout1D(0.4))

model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(3,activation='sigmoid'))

#model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
X.shape
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state= 3)

Y = data[['pos_neg','pos_neu','neu_neg']]

#Y = pd.get_dummies(data['senti']).values

#X,Y = sm.fit_resample(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
batch_size = 32

model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)
validation_size = 1500

X_validate = X_test[-validation_size:]

Y_validate = Y_test[-validation_size:]

X_test = X_test[:-validation_size]

Y_test = Y_test[:-validation_size]

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))
twt = [''' 





this is the best show ever



''']

#vectorizing the tweet by the pre-fitted tokenizer instance

twt = tokenizer.texts_to_sequences(twt)

print(twt)

#padding the tweet to have exactly the same shape as `embedding_2` input

twt = pad_sequences(twt, maxlen=32, dtype='int32', value=0)

print(twt)

sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]

print(sentiment)

# if(np.argmax(sentiment) == 0):

#     print("negative")

# elif (np.argmax(sentiment) == 1):

#     print("positive")

if sentiment[0] >=0.5 and sentiment[1] >= 0.5 and sentiment[2]>=0.5:

    print("Positive")

elif sentiment[0] <=0.5 and sentiment[1] <= 0.5 and sentiment[2]<=0.5:

    print("Negative")

else:

    print("Neutral")
data.to_csv('data.csv')