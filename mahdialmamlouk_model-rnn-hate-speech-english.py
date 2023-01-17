import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

%matplotlib inline
df = pd.read_csv('../input/hate-speech-twitter-train-and-test/train_E6oV3lV.csv',delimiter=',',encoding='latin-1')

df
df.drop(['id'],axis=1,inplace=True)

df.info()
sns.countplot(df.label)

plt.xlabel('Label')

plt.title('Number of ham and spam messages')
def eval_fun(labels, preds):

    labels = label.split(' ')

    preds = tweet.split(' ')

    rr = (np.intersect1d(label, tweet))

    precision = np.float(len(rr)) / len(tweet)

    recall = np.float(len(rr)) / len(label)

    try:

        f1 = 2 * precision * recall / (precision + recall)

    except ZeroDivisionError:

        return (precision, recall, 0.0)

    return (precision, recall, f1)

print(1)
import numpy as np

print("Hatred labeled: {}\nNon-hatred labeled: {}".format(

    (df.label == 1).sum(),

    (df.label == 0).sum()

))
hashtags = df['tweet'].str.extractall('#(?P<hashtag>[a-zA-Z0-9_]+)').reset_index().groupby('level_0').agg(lambda x: ' '.join(x.values))

df.loc[:, 'hashtags'] = hashtags['hashtag']

df['hashtags'].fillna('', inplace=True)



df.loc[:, 'mentions'] = df['tweet'].str.count('@[a-zA-Z0-9_]+')



df.tweet = df.tweet.str.replace('@[a-zA-Z0-9_]+', '')
df.tweet = df.tweet.str.replace('[^a-zA-Z]', ' ')
from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

from nltk import pos_tag, FreqDist, word_tokenize



stemmer = SnowballStemmer('english')

lemmer = WordNetLemmatizer()



part = {

    'N' : 'n',

    'V' : 'v',

    'J' : 'a',

    'S' : 's',

    'R' : 'r'

}



def convert_tag(penn_tag):

    if penn_tag in part.keys():

        return part[penn_tag]

    else:

        return 'n'





def tag_and_lem(element):

    sent = pos_tag(word_tokenize(element))

    return ' '.join([lemmer.lemmatize(sent[k][0], convert_tag(sent[k][1][0]))

                    for k in range(len(sent))])

    



df.loc[:, 'tweet'] = df['tweet'].apply(lambda x: tag_and_lem(x))

df.loc[:, 'hashtags'] = df['hashtags'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
X = df.tweet

Y = df.label

le = LabelEncoder()

Y = le.fit_transform(Y)

Y = Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)
max_words = 1000

max_len = 100

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
def RNN():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(64,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#history = model.fit(sequences_matrix,Y_train,batch_size=64,epochs=10,

#          validation_split=0.2)
history = model.fit(sequences_matrix,Y_train,batch_size=64,epochs=10,

          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()