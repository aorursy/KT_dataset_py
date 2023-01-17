import re

import os

import nltk

import string

import warnings



import numpy as np

import pandas as pd



from plotly import tools

import plotly.offline as pyo

import plotly.graph_objs as go

import plotly.figure_factory as ff



import matplotlib.pyplot as plt



from collections import Counter
warnings.filterwarnings('ignore')



pyo.offline.init_notebook_mode(connected= True)
df = pd.read_json('../input/Sarcasm_Headlines_Dataset.json', lines= True)
df.head()
df.describe()
df.info()
for col in df.columns:

    print(sum(df.isnull()[col]))
x = 'https://www.github.com'
re.findall('\w+', x)
df['source'] = df['article_link'].apply(lambda x: re.findall('\w+', x)[2])
df.head()
labels = ['Sarcasm', 'Not Sarcasm']

values = [df['is_sarcastic'].sum(), len(df) - df['is_sarcastic'].sum()]



trace0 = go.Bar(x = [labels[1]], y = [values[1]], name = 'No Sarcasm')

trace1 = go.Bar(x = [labels[0]], y = [values[0]], marker= {'color': '#00cc66'} , name = 'Sarcasm')





data = [trace0, trace1]



layout = go.Layout(title = 'Number of Sarcastic Articles',

                   width = 800,

                   height = 500,

                  yaxis= dict(title = 'Number of articles'),)



fig = go.Figure(data, layout)



pyo.offline.iplot(fig)
df.head(10)
list(df['source'].value_counts())
trace0 = go.Bar(x = df['source'].unique(), y = list(df['source'].value_counts()) , marker = {'color': '#9900e6'}, name = 'Sarcasm')



data = [trace0]



layout = go.Layout(title = 'Sources of Articles',

                   width = 800,

                   height = 500,

                  yaxis= dict(title = 'Number of articles'),)



fig = go.Figure(data, layout)



pyo.offline.iplot(fig)
x = 'F*&!@#$%&*(U)+>"{}C<>!@#$K%&*()+>"{}<> T%&!@H#$%&*()+E>"{}<S>!@#$%&*()E+>"{}<> *&P(($UN#$$@!CT%$&UA&&*T()I$%@(O*&N!S'

print(x)

print('\n'+re.sub('[^a-zA-z0-9\s]','',x))
df['headlineNoPunc'] = df['headline'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
print(df['headlineNoPunc'][456])

df['headline'][456]
allWords_df = df['headlineNoPunc'].str.split(expand = True).unstack().value_counts()

allWords_df = pd.DataFrame(allWords_df).reset_index()
trace0 = go.Bar(x = allWords_df['index'][0:50] , y = allWords_df[0][0:50], marker = dict(color = allWords_df[0]

                                                                                   , colorscale = 'Portland')

                                                                                , name = 'Top 50 Frequent Headline Words')



data = [trace0]



layout = go.Layout(title = 'Most Frequent Words',

                   width = 950,

                   height = 600,

                   xaxis = dict(title = 'Words', nticks = 50),

                  yaxis= dict(title = 'Count'),)



fig = go.Figure(data, layout)



pyo.offline.iplot(fig)
nltk.download('stopwords')
from nltk.corpus import stopwords
word_counts = df['headlineNoPunc'].str.split(expand = True).unstack().value_counts()

word_counts = pd.DataFrame(word_counts).reset_index()
stopwords = stopwords.words('english') # let's save us some hassle
noSarcasmWords_all = df[df['is_sarcastic'] == 0]['headlineNoPunc'].str.split(expand = True).unstack()

sarcasmWords_all = df[df['is_sarcastic'] == 1]['headlineNoPunc'].str.split(expand = True).unstack()
noSarcasmWords = [wo for wo in noSarcasmWords_all if wo not in stopwords]

sarcasmWords = [wo for wo in sarcasmWords_all if wo not in stopwords]
from collections import Counter
noSarcasmWordCount = pd.DataFrame(list(Counter(noSarcasmWords).items()), columns= ['word','count'])

sarcasmWordCount = pd.DataFrame(list(Counter(sarcasmWords).items()), columns= ['word','count'])
temp1 = noSarcasmWordCount.sort_values(by = ['count'], ascending= False)[1:31]

temp2 = sarcasmWordCount.sort_values(by = ['count'], ascending= False)[1:31]
trace0 = go.Bar(y = temp1['word'] , x = temp1['count'], marker = dict(color = '#ff3333'), opacity= 0.6, orientation = 'h' ,name = 'Frequent words in Non Sarcastic articles')

trace1 = go.Bar(y = temp2['word'] , x = temp2['count'], marker = dict(color = '#33cc33'), opacity= 0.6, orientation = 'h' ,name = 'Frequent words in Sarcastic articles')



fig = tools.make_subplots(rows= 1, cols= 2)

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=600, width=1200, title = 'Frequent words', yaxis = dict(nticks = 30))

                     

pyo.offline.iplot(fig)
nltk.download('punkt')

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()



sentence = "i am trying, he is frying, she learned shit, i mowed the lawn"



sentence_words = sentence.split()



sentence_words = [re.sub('[^a-zA-z0-9\s]','',x) for x in sentence_words]

print(sentence_words)

sentence_words

print("{0:20}{1:20}".format("Word","Lemma"))

for word in sentence_words:

    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word, pos = 'v')))
noSarcasmWords_lemm = []

for word in noSarcasmWords:

    if word != None:

        noSarcasmWords_lemm.append(wordnet_lemmatizer.lemmatize(word, pos = 'v'))



sarcasmWords_lemm = []

for word in sarcasmWords:

    if word != None:

        sarcasmWords_lemm.append(wordnet_lemmatizer.lemmatize(word, pos = 'v'))
print('finding' in noSarcasmWords)



print('finding' in noSarcasmWords_lemm)

print('find' in noSarcasmWords_lemm)
from wordcloud import WordCloud
sar_cloud1 = WordCloud(background_color= 'white', colormap= 'viridis').generate(" ".join(noSarcasmWords_lemm))

sar_cloud2 = WordCloud(background_color= 'white', colormap= 'viridis').generate(" ".join(sarcasmWords_lemm))





fig = plt.figure(figsize=(20,20))



ax1 = fig.add_subplot(1, 2, 1)

ax1.tick_params(length=0, width=0)

ax1.set_xticklabels([])

ax1.set_yticklabels([])

ax1.set_title("In Non Sarcastic Articles")

ax1.imshow(sar_cloud1)



ax2 = fig.add_subplot(1, 2, 2)

ax2.tick_params(length=0, width=0)

ax2.set_xticklabels([])

ax2.set_yticklabels([])

ax2.set_title("In Sarcastic Articles")

ax2.imshow(sar_cloud2)



plt.show()
bigrams = list(nltk.bigrams(sarcasmWords_lemm))

bigram_counts = Counter(bigrams)



bigram_name = []

bigram_freq = []

for key in bigram_counts.keys():

    if bigram_counts[key] > 1:

        bigram_name.append(str(key))

        bigram_freq.append(bigram_counts[key])

        

bigram_df = pd.DataFrame({'bigram_names': bigram_name, 'bigram_freq': bigram_freq})





# our values in the first columns look like '('doctor', 'man')'

# Let's clean it up to 'doctor man'



bigram_df['bigram_names'] = [re.sub('[^a-zA-z0-9\s]','',x) for x in bigram_df['bigram_names']]



bigram_df.head()
temp = bigram_df.sort_values(by = 'bigram_freq')[-20:]

trace0 = go.Bar(y = temp['bigram_names'] , x = temp['bigram_freq'], orientation= 'h', marker = dict(color = '#00997a')

               , opacity = 0.6, showlegend = False)



fig = tools.make_subplots(1,2, subplot_titles = ['in Sarcasm Articles','in Non Sarcasm Articles'] ,horizontal_spacing = 0.05, shared_xaxes = False)

fig.append_trace(trace0, 1, 1)
bigrams = list(nltk.bigrams(noSarcasmWords_lemm))

bigram_counts = Counter(bigrams)



bigram_name = []

bigram_freq = []

for key in bigram_counts.keys():

    if bigram_counts[key] > 1:

        bigram_name.append(str(key))

        bigram_freq.append(bigram_counts[key])

        

bigram_df = pd.DataFrame({'bigram_names': bigram_name, 'bigram_freq': bigram_freq})



bigram_df['bigram_names'] = [re.sub('[^a-zA-z0-9\s]','',x) for x in bigram_df['bigram_names']]



bigram_df.head()
temp = bigram_df.sort_values(by = 'bigram_freq')[-20:]

trace1 = go.Bar(y = temp['bigram_names'] , x = temp['bigram_freq'], orientation= 'h', marker = dict(color = '#00e6b8')

                , showlegend = False, opacity = 0.6)

fig.append_trace(trace1, 1, 2)

fig['layout'].update( height=600, width=1100, title = 'Bigrams', yaxis = dict(nticks = 20))

pyo.offline.iplot(fig)
# Finding the number of words in the headline

df['num_words'] = df['headlineNoPunc'].apply(lambda x: len(x.split()))



# the unique number of words in the headline

df['unique_num_words'] = df['headlineNoPunc'].apply(lambda x: len(set(x.split())))



# the number of characters in the headline

df['num_chars'] = df['headline'].apply(lambda x: len(x))



# number of genuine words with no stopwords  

df['num_words_nostop'] = df['headlineNoPunc'].apply(lambda xx: len([x for x in xx.split() if x not in stopwords]))



# number of stopwords

df['num_stop'] = df['num_words'] - df['num_words_nostop']



# number of punctuations

df['num_punc'] = df['headline'].apply(lambda xx: len([x for x in xx if x in string.punctuation]))
trace0 = go.Box(y = df[df['is_sarcastic'] == 0]['num_words'], name = 'Non Sarcasm')

trace1 = go.Box(y = df[df['is_sarcastic'] == 1]['num_words'], name = 'Sarcasm')

data = [trace0, trace1]

layout = go.Layout(title = 'Number of Words',

                   width = 900,

                   height = 600)

fig = go.Figure(data, layout)

pyo.offline.iplot(fig)
trace0 = go.Box(y = df[df['is_sarcastic'] == 0]['unique_num_words'], name = 'Non Sarcasm')

trace1 = go.Box(y = df[df['is_sarcastic'] == 1]['unique_num_words'], name = 'Sarcasm')

data = [trace0, trace1]

layout = go.Layout(title = 'Number of Unique Words',

                   width = 900,

                   height = 600)

fig = go.Figure(data, layout)

pyo.offline.iplot(fig)
trace0 = go.Box(y = df[df['is_sarcastic'] == 0]['num_chars'], name = 'Non Sarcasm')

trace1 = go.Box(y = df[df['is_sarcastic'] == 1]['num_chars'], name = 'Sarcasm')

data = [trace0, trace1]

layout = go.Layout(title = 'Number of Characters',

                   width = 900,

                   height = 600)

fig = go.Figure(data, layout)

pyo.offline.iplot(fig)
trace0 = go.Box(y = df[df['is_sarcastic'] == 0]['num_punc'], name = 'Non Sarcasm')

trace1 = go.Box(y = df[df['is_sarcastic'] == 1]['num_punc'], name = 'Sarcasm')

data = [trace0, trace1]

layout = go.Layout(title = 'Number of Punctuations',

                   width = 900,

                   height = 600)

fig = go.Figure(data, layout)

pyo.offline.iplot(fig)
from sklearn.preprocessing import LabelEncoder
X = df['headlineNoPunc']

y = df['is_sarcastic']
enc = LabelEncoder()

y = enc.fit_transform(y)

y = y.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)
import keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence
max_words = 1000

max_len = 150

tok = Tokenizer(num_words= max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences, max_len)
import pickle

pickle.dump(tok,open('tokenizer.pkl','wb'))  
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.models import Model
def RNN():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.2)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
checkpoint_path = "model.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)



# Create checkpoint callback

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,

                                                 save_weights_only=True,

                                                 verbose=1)
from keras.callbacks import EarlyStopping

model.fit(sequences_matrix,y_train,batch_size=100,epochs=3,

          validation_split=0.1,callbacks=[cp_callback])
pickle.dump(model,open('model.pkl','wb'))
# model_2 = RNN()

# model_2.summary()

# model_2.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
# model_2.load_weights('model.ckpt')
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
model_loaded = pickle.load(open('model.pkl', 'rb'))

token = pickle.load(open('tokenizer.pkl', 'rb'))
def detect_sarcasm(sent):

    sent = re.sub('[^a-zA-z0-9\s]','',sent)

    inp_sent = pd.Series([sent])

    seq = token.texts_to_sequences(inp_sent)

    sequences_matrix = sequence.pad_sequences(seq, max_len)

    res = model_loaded.predict(sequences_matrix)

    if res[0][0] <= 0.5:

        print("\nNah! seems legit!")

    if res[0][0] > 0.5:

        print("\nThis shit is sarcastic bro!")
detect_sarcasm(input())