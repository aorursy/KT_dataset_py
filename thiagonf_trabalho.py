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

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

%matplotlib inline

import os

import re

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



print(os.listdir("../input"))
file_train = '../input/train.csv'

file_valid = '../input/valid.csv'

file_exemplo = '../input/sample_su.csv'

data_train = pd.read_csv(file_train)

data_valid = pd.read_csv(file_valid)

print(len(data_train), len(data_valid))
init_notebook_mode(connected=True)



sarcams_percent = data_train['is_sarcastic'].value_counts()

labels = ['True', 'Sarcastic']

sizes = (np.array((sarcams_percent / sarcams_percent.sum())*100))

colors = ['#58D68D', '#9B59B6']



trace = go.Pie(labels=labels, values=sizes, opacity = 0.8, hoverinfo='label+percent',

               marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))

layout = go.Layout(

    title='Sarcastic Vs True'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="Sa_Ac")
Y_dummies = pd.get_dummies(data_train['is_sarcastic']).values

Y = data_train['is_sarcastic']

data_train_format = data_train.drop('is_sarcastic', axis=1)

data_all = data_train_format.append(data_valid)

data_all = data_all.drop('article_link', axis=1)

data_all = data_all.drop('ID', axis=1)

data_all['headline'] = data_all['headline'].apply(lambda x: x.lower())

data_all['headline'] = data_all['headline'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(len(data_train['headline']), len(data_valid['headline']))

print(len(data_all))
all_words = data_all['headline'].str.split(expand=True).unstack().value_counts()

data = [go.Bar(

            x = all_words.index.values[2:50],

            y = all_words.values[2:50],

            marker= dict(colorscale='Viridis',

                         color = all_words.values[2:100]

                        ),

            text='Word counts'

    )]



layout = go.Layout(

    title='Frequent Occuring word (unclean) in Headlines'

)



fig = go.Figure(data=data, layout=layout)



iplot(fig, filename='basic-bar')
max_fatures = 5000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data_all['headline'].values)

X = tokenizer.texts_to_sequences(data_all['headline'].values)

X = pad_sequences(X)
data_train_split = X[:len(data_train)]

data_valid_split = X[len(data_train):]



print(len(data_train_split), len(data_valid_split))
embed_dim = 64

lstm_out = 196



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = data_train_split.shape[1])) # Size of the vocabulary,  Dimension of matrix embeding

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)) # Dimensionality of the output space, Fraction continuous linear value and recurrent value

model.add(Dense(2,activation='softmax')) # Dimensionality of the output space(0,1), Function activation

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# Softmax this function will calculate the probabilities of each target class over all possible target classes
batch_size = 32

history = model.fit(data_train_split, Y_dummies, epochs = 60, batch_size=batch_size, verbose = 2)
# summarize history for accuracy

plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.savefig('model_accuracy.png')

# summarize history for loss

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.savefig('model_loss.png')
Y_sarcasm = model.predict(data_valid_split,batch_size=1,verbose = 2)
list_sarcasm = list()

for y in Y_sarcasm:

    if(np.argmax(y) == 0):

        list_sarcasm.append(0)

    elif (np.argmax(y) == 1):

        list_sarcasm.append(1)

print(len(list_sarcasm))
data_to_submit = pd.DataFrame({

    'ID': data_valid['ID'],

    'is_sarcastic':list_sarcasm

})



data_to_submit.to_csv('csv_to_submit.csv', index = False)