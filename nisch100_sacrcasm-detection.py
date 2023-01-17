import pandas as pd

path = '../input/'

sar_acc = pd.read_json(path+ 'Sarcasm_Headlines_Dataset.json',lines=True)

import re

print(sar_acc.head())

sar_acc['source'] = sar_acc['article_link'].apply(lambda x: re.findall(r'\w+', x)[2])

sar_acc.head()
import numpy as np

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



sar_acc_tar = sar_acc['is_sarcastic'].value_counts()

labels = ['Acclaim', 'Sarcastic']

sizes = (np.array((sar_acc_tar / sar_acc_tar.sum())*100))

colors = ['#58D68D', '#9B59B6']



trace = go.Pie(labels=labels, values=sizes, opacity = 0.8, hoverinfo='label+percent',

               marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))

layout = go.Layout(

    title='Sarcastic Vs Acclaim'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="Sa_Ac")
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

all_words = sar_acc['headline'].str.split(expand=True).unstack().value_counts()

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
sar_det = sar_acc[sar_acc.is_sarcastic==1]

sar_det.reset_index(drop=True, inplace=True)

acc_det = sar_acc[sar_acc.is_sarcastic==0]

acc_det.reset_index(drop=True, inplace=True)



# Tokenizing the Headlines of Sarcasm

sar_news = []

for rows in range(0, sar_det.shape[0]):

    head_txt = sar_det.headline[rows]

    head_txt = head_txt.split(" ")

    sar_news.append(head_txt)



#Converting into single list for Sarcasm

import itertools

sar_list = list(itertools.chain(*sar_news))



# Tokenizing the Headlines of Acclaim

acc_news = []

for rows in range(0, acc_det.shape[0]):

    head_txt = acc_det.headline[rows]

    head_txt = head_txt.split(" ")

    acc_news.append(head_txt)

    

#Converting into single list for Acclaim

acc_list = list(itertools.chain(*acc_news))
import nltk



stopwords = nltk.corpus.stopwords.words('english')

sar_list_restp = [word for word in sar_list if word.lower() not in stopwords]

acc_list_restp = [word for word in acc_list if word.lower() not in stopwords]



print("Length of original Sarcasm list: {0} words\n"

      "Length of Sarcasm list after stopwords removal: {1} words"

      .format(len(sar_list), len(sar_list_restp)))



print("=="*46)



print("Length of original Acclaim list: {0} words\n"

      "Length of Acclaim list after stopwords removal: {1} words"

      .format(len(acc_list), len(acc_list_restp)))
#Data cleaning for getting top 30

from collections import Counter

sar_cnt = Counter(sar_list_restp)

acc_cnt = Counter(acc_list_restp)



#Dictonary to Dataframe

sar_cnt_df = pd.DataFrame(list(sar_cnt.items()), columns = ['Words', 'Freq'])

sar_cnt_df = sar_cnt_df.sort_values(by=['Freq'], ascending=False)

acc_cnt_df = pd.DataFrame(list(acc_cnt.items()), columns = ['Words', 'Freq'])

acc_cnt_df = acc_cnt_df.sort_values(by=['Freq'], ascending=False)



#Top 30

sar_cnt_df_30 = sar_cnt_df.head(30)

acc_cnt_df_30 = acc_cnt_df.head(30)
#Plotting the top 30 Sarcasm Vs Acclaim

from plotly import tools

sar_tr  = go.Bar(

    x=sar_cnt_df_30['Freq'],

    y=sar_cnt_df_30['Words'],

    name='Sarcasm',

    marker=dict(

        color='rgba(155, 89, 182, 0.6)',

        line=dict(

            color='rgba(155, 89, 182, 1.0)',

            width=.3,

        )

    ),

    orientation='h',

    opacity=0.6

)



acc_tr  = go.Bar(

    x=acc_cnt_df_30['Freq'],

    y=acc_cnt_df_30['Words'],

    name='Acclaim',

    marker=dict(

        color='rgba(88, 214, 141, 0.6)',

        line=dict(

            color='rgba(88, 214, 141, 1.0)',

            width=.3,

        )

    ),

    orientation='h',

    opacity=0.6

)



fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Top 30 Most occuring words in Sarcasm Headlines',

                                                          'Top 30 Most occuring words in Acclaim Headlines'))



fig.append_trace(sar_tr, 1, 1)

fig.append_trace(acc_tr, 2, 1)





fig['layout'].update(height=1200, width=800)



iplot(fig, filename='sar_vs_acc')
stemmer = nltk.stem.SnowballStemmer("english", ignore_stopwords=True)



print("The stemmed form of dealing is: {}".format(stemmer.stem("dealing")))

print("The stemmed form of deals is: {}".format(stemmer.stem("deals")))

print("The stemmed form of deal is: {}".format(stemmer.stem("deal")))



print("The stemmed form of Leaves is: {}".format(stemmer.stem("leaves")))



from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()

print("The lemmatized form of leaves is: {}".format(lemm.lemmatize("leaves")))
#Sarcasm headline after Lemmatization

sar_wost_lem = []

for batch in sar_news:

    sar_list_restp = [word for word in batch if word.lower() not in stopwords]

    lemm = WordNetLemmatizer()

    sar_list_lemm =  [lemm.lemmatize(word) for word in sar_list_restp]

    sar_wost_lem.append(sar_list_lemm)



#Acclaim headline after Lemmatization

acc_wost_lem = []

for batch in acc_news:

    acc_list_restp = [word for word in batch if word.lower() not in stopwords]

    lemm = WordNetLemmatizer()

    acc_list_lemm =  [lemm.lemmatize(word) for word in acc_list_restp]

    acc_wost_lem.append(sar_list_lemm)
from sklearn.feature_extraction.text import CountVectorizer

vec = []

for block in sar_wost_lem:

    vectorizer = CountVectorizer(min_df=0)

    sentence_transform = vectorizer.fit_transform(block)

    vec.append(sentence_transform)

    

print("The features are:\n {}".format(vectorizer.get_feature_names()))

print("\nThe vectorized array looks like:\n {}".format(sentence_transform.toarray()))
sar_wost_lem_df = pd.DataFrame({'sarcasm':sar_wost_lem})

acc_wost_lem_df = pd.DataFrame({'acclaim':acc_wost_lem})



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    ngrams = zip(*[text[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



#Plotting the Bigram plot

from collections import defaultdict

freq_dict = defaultdict(int)

for sent in sar_wost_lem_df["sarcasm"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

sar_2 = horizontal_bar_chart(fd_sorted.head(50), '#280f52')





freq_dict = defaultdict(int)

for sent in acc_wost_lem_df["acclaim"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

acc_2 = horizontal_bar_chart(fd_sorted.head(50), '#854238')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=["Frequent bigrams of Sarcasm Headlines", 

                                          "Frequent bigrams of Acclaim Headlines"])

fig.append_trace(sar_2, 1, 1)

fig.append_trace(acc_2, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Plots Sarcasm Vs Acclaim after removing Stopwords")

iplot(fig, filename='word-plots')
#Getting X and Y ready

from sklearn.preprocessing import LabelEncoder

X = sar_acc.headline

Y = sar_acc.is_sarcastic

le = LabelEncoder()

Y = le.fit_transform(Y)

Y = Y.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

max_words = 1000

max_len = 150

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
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
from keras.callbacks import EarlyStopping

model.fit(sequences_matrix,Y_train,batch_size=100,epochs=10,

          validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))