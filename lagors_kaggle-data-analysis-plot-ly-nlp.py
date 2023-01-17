import numpy as np

import pandas as pd

#basic data science livraries for data manipulation



import warnings #Used to hide unnecessary warning messages



warnings.simplefilter('ignore')







import matplotlib.pyplot as plt 

import seaborn as sns 

#plotting libraries

sns.set(style='darkgrid', palette='Set3')







import plotly as py 

from plotly import subplots

from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objects as go  

#for interactive plotting

init_notebook_mode(connected=True)

py.offline.init_notebook_mode (connected = True)









from wordcloud import WordCloud

import nltk

import string

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

import random

import statistics

from nltk.classify import NaiveBayesClassifier

#text data manipulation, visualization and classification









from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping

#deep learning libraries for further application



languages=pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')

kernels=pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')

kernelTags=pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')

tags=pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')

versions=pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')
languages.tail()
languages= languages[['Id','DisplayName','IsNotebook']]
kernels.tail()
kernels=kernels[['Id','CurrentKernelVersionId','FirstKernelVersionId',

                'MadePublicDate','Medal','TotalViews','TotalComments','TotalVotes']]

kernelTags.tail()
tags.tail()
tags=tags[['Id','Name']]
versions.tail()
versions= versions[['ScriptId','ScriptLanguageId','AuthorUserId','VersionNumber','Title']]
#Starting by merging the data



versions.drop_duplicates(subset='ScriptId',keep='last',inplace=True)



versions =versions.join(languages.set_index('Id'), on='ScriptLanguageId', how='left')



kernels =versions.join(kernels.set_index('Id'), on='ScriptId')



kernelTags= kernelTags.join(tags.set_index('Id'),on='TagId', how='left')



#Stacking the Tags into rows and merging into a main dataframe



a=kernelTags[kernelTags.duplicated('KernelId')] [['KernelId','Name']]

c=kernelTags.drop_duplicates(subset='KernelId', keep='first').join(a.set_index('KernelId'),on='KernelId', how='left',rsuffix='_2')



a=c[c.duplicated('KernelId')] [['KernelId','Name_2']]

d=c.drop_duplicates(subset='KernelId', keep='first').join(a.set_index('KernelId'),on='KernelId', how='left',rsuffix='_3')



a=d[d.duplicated('KernelId')] [['KernelId','Name_2_3']]

e=d.drop_duplicates(subset='KernelId', keep='first').join(a.set_index('KernelId'),on='KernelId', how='left',rsuffix='_4')



a=e[e.duplicated('KernelId')] [['KernelId','Name_2_3_4']]

f=e.drop_duplicates(subset='KernelId', keep='first').join(a.set_index('KernelId'),on='KernelId', how='left',rsuffix='_5')



tags=f.drop_duplicates('KernelId',keep='first')

Names={'Name':'Tag1','Name_2':'Tag2','Name_2_3': 'Tag3','Name_2_3_4':'Tag4','Name_2_3_4_5':'Tag5'}

tags.rename(Names, axis=1, inplace=True)

tags.drop(columns=['TagId','Id'],inplace=True)
kernels =kernels.join(tags.set_index('KernelId'), on='ScriptId', how='left')

kernels.drop(columns=['ScriptLanguageId','CurrentKernelVersionId','FirstKernelVersionId'],inplace=True)

kernels.tail()
# We only take in consideration published works

kernels.dropna(axis=0,subset=['MadePublicDate'], inplace=True)



# we fill the missing data in Versions with 1 

kernels.loc[kernels['VersionNumber'].isna()].fillna(1, inplace=True)



# we fill the missing data in language with other and in Isnotbook as False

kernels['DisplayName'].fillna('Other', inplace=True)

kernels['IsNotebook'].fillna(False,inplace=True)
print(kernels['Medal'].isna().count())
kernels.drop('Medal',axis=1,inplace=True)

kernels.reset_index(drop=True,inplace=True)

kernels.head()
label=['notebook','script']

trace1=go.Pie(labels=kernels['DisplayName'],domain=dict(x=[0, 0.5]),name='Language',marker_colors=['#ADFFA2','darkred'])

trace2=go.Pie(labels=kernels['IsNotebook'], domain=dict(x=[0.5, 1.0]), name='Notebook', marker_colors=['#6ED1E9','grey'])



fig1 = subplots.make_subplots(rows = 1, cols = 2, specs=[[{"type": "pie"}, {"type": "pie"}]])

fig1.add_trace(trace1,row=1,col=1)

fig1.add_trace(trace2,row=1,col=2)



fig1.update_traces(hoverinfo='label+percent+name', textinfo='label',textfont_size=20,marker=dict(line=dict(color='#000000', width=2)))

fig1.update(layout_title_text='Language and notebook types',layout_showlegend=False)



iplot(fig1)
languagePlot= kernels.groupby('DisplayName').sum().reset_index()

notebookPlot= kernels.groupby('IsNotebook').sum().reset_index()



languageCount= kernels.groupby('DisplayName').count().reset_index()

notebookCount= kernels.groupby('IsNotebook').count().reset_index()



trace3= go.Bar(x=languagePlot['DisplayName'],y=languagePlot['TotalVotes']/languageCount['TotalVotes'], name='Votes',marker_color='lightgreen')

trace4= go.Bar(x=languagePlot['DisplayName'],y=languagePlot['TotalComments']/languageCount['TotalComments'], name ='Comments')



trace5= go.Bar(x=notebookPlot['IsNotebook'],y=notebookPlot['TotalVotes']/notebookCount['TotalVotes'], name='Votes',marker_color='#FF9700')

trace6= go.Bar(x=notebookPlot['IsNotebook'],y=notebookPlot['TotalComments']/notebookCount['TotalComments'], name ='Comments',marker_color='purple')

fig2= subplots.make_subplots(rows= 1 , cols = 2)



fig2.append_trace(trace3,row=1,col=1)

fig2.append_trace(trace4,row=1,col=1)

fig2.append_trace(trace5,row=1,col=2)

fig2.append_trace(trace6,row=1,col=2)



fig2.update(layout_title_text='Average popularity of languages',layout_showlegend=False)

fig2.update_traces(marker=dict(line=dict(color='#000000', width=2)))

fig2.update_yaxes(showticklabels=False)



iplot(fig2)
kernels['AuthorUserId']= kernels['AuthorUserId'].astype(str)



authorPlot=kernels.groupby(by=kernels['AuthorUserId']).count().reset_index()

authorPlot.sort_values(by='Title', ascending=False,inplace=True)





authorPlot['Title'].head()

authorPlot= authorPlot[1:]

print('Average number of kernels per user:',authorPlot['Title'].mean())

authorPlot= authorPlot[:30]





f=plt.subplots(figsize=(18,8))

f=sns.barplot(x=authorPlot['AuthorUserId'],y=authorPlot['Title'],color='orange')

f.set_title('Number of lernels by most active users')

f.set_xticklabels(f.get_xticklabels(), rotation=45)

a=f.set(xlabel='User Id', ylabel='Number of Kernels')
authorPlot=kernels.groupby(by=kernels['AuthorUserId']).sum().reset_index()

authorPlot.sort_values(by='TotalVotes', ascending=False,inplace=True)

authorPlot= authorPlot[:30]



f=plt.subplots(figsize=(18,8))

f=sns.barplot(x=authorPlot['AuthorUserId'],y=authorPlot['TotalVotes'],color='orange')

f.set_title('Number of votes by users')

f.set_xticklabels(f.get_xticklabels(), rotation=45)

a=f.set(xlabel='User Id', ylabel='Number of votes')
kernels['MadePublicDate']=pd.to_datetime(kernels['MadePublicDate'])

dates=kernels.groupby(kernels['MadePublicDate']).count().reset_index()



trace7=go.Line(y=dates['Title'],x=dates['MadePublicDate'], name='Published kernels')

trace8=go.Line(y=dates['Title'][-425:-365],x=dates['MadePublicDate'][-425:-365],name='Published kernels')



fig3 = subplots.make_subplots(rows = 2, cols = 1)

fig3.add_trace(trace7,row=1,col=1)

fig3.add_trace(trace8,row=2,col=1)



fig3.update(layout_title_text='Trends with a zoom on a month in 2019',layout_showlegend=False)

iplot(fig3)
#we use regular plotting here as the y axis is not important since we're doing a 6 months rolling mean to see the evolution

plt.figure(figsize=(20,6))

plt.plot(dates['MadePublicDate'],dates['Title'].rolling(180).mean(),linewidth=3, c='black')

a=plt.title('6 months rolled mean analysis of the Number of published kernels ')
trace9=go.Line(y=dates['TotalViews'],x=dates['MadePublicDate'],name='Views')

trace10=go.Line(y=dates['TotalVotes'],x=dates['MadePublicDate'],name='Votes')

trace11=go.Line(y=dates['TotalComments'],x=dates['MadePublicDate'], name='Comments')





fig4 = subplots.make_subplots(rows = 3, cols = 1)

fig4.add_trace(trace9,row=1,col=1)

fig4.add_trace(trace10,row=2,col=1)

fig4.add_trace(trace11,row=3,col=1)



fig4.update(layout_title_text='Views, votes and comments over time',layout_showlegend=False)

iplot(fig4)
#as we did earlier, 6 months roll are represented in basic plotting

dates=kernels.groupby(['MadePublicDate']).sum().reset_index()

f=plt.figure(figsize=(20,10))

ax1=f.add_subplot(2,1,1)

ax1.plot(dates['MadePublicDate'],dates['TotalViews'].rolling(180).mean(),label='Views', linewidth=3, c= 'blue')

ax1.set_title('6 months rolled mean analysis of the Views of published kernels ')

ax1.legend()



ax2=f.add_subplot(2,1,2)

ax2.plot(dates['MadePublicDate'],dates['TotalVotes'].rolling(180).mean(),label='Votes', linewidth=3, c= 'red')

ax2.plot(dates['MadePublicDate'],dates['TotalComments'].rolling(180).mean(), label='Comments', linewidth=3, c= 'green')

ax2.legend()

a=ax2.set_title('6 months rolled mean analysis of the Votes and comments of published kernels ')
percentageVotes=kernels['TotalVotes'].sum()/kernels['TotalViews'].sum() * 100

percentageComments=kernels['TotalComments'].sum()/kernels['TotalViews'].sum() * 100



print(float("{0:.4f}".format(percentageVotes)),'%   votes/views')

print(float("{0:.4f}".format(percentageComments)),'%   comments/views')





correlations=kernels[['TotalViews','TotalVotes','TotalComments']]



f=plt.figure(figsize=(10,8))

f=sns.heatmap(correlations.corr(),annot=True)

a=f.set_title('Matrix of correlation')
versionsPlot=kernels

versionsPlot = kernels[kernels['VersionNumber'] < kernels['VersionNumber'].quantile(0.9999)]

versionsPlot = versionsPlot[kernels['TotalVotes'] < versionsPlot['TotalVotes'].quantile(0.9999)]







trace12=go.Scattergl(x=versionsPlot['VersionNumber'],y=versionsPlot['TotalVotes'], mode='markers', name='Votes', marker_color='#3C8E6A')

trace13=go.Scattergl(x=versionsPlot['VersionNumber'],y=versionsPlot['TotalComments'], mode='markers', name='Comments',marker_color='#893636')



fig5= go.Figure()

fig5.add_trace(trace12)

fig5.add_trace(trace13)

fig5.update_layout(title='Distribution of votes/comments by version number')

fig5.update_xaxes(title_text='Version')

fig5.update_traces(marker=dict(line=dict(color='#000000', width=0.5)))





iplot(fig5)
tagsPlot= kernelTags.groupby(['Name']).count().reset_index()

tagsPlot.sort_values(by='KernelId',ascending=False, inplace=True)

tagsPlot=tagsPlot.head(12)





plt.figure(figsize=(20,7))

f=sns.barplot(x=tagsPlot['Name'],y=tagsPlot['KernelId'])

f.set_title('Most used tags')

f.set_xlabel('Tag')

a=f.set_xticklabels(f.get_xticklabels(), rotation=15)
kernelTags =kernels.join(kernelTags.set_index('KernelId'), on='ScriptId', how='left')

kernelTags= kernelTags[['ScriptId','TotalVotes','Name']]



tagsPlot= kernelTags.groupby(['Name']).sum().reset_index()

tagsPlot.sort_values(by='TotalVotes',ascending=False, inplace=True)

tagsPlot=tagsPlot.head(12)





plt.figure(figsize=(20,7))

f=sns.barplot(x=tagsPlot['Name'],y=tagsPlot['TotalVotes'])

f.set_title('Tags attracting most votes')

f.set_xlabel('Tag')

a=f.set_xticklabels(f.get_xticklabels(), rotation=15)
kernels.reset_index(drop=True, inplace=True)

kernels['Title']= kernels['Title'].astype(str)



kernels['Title'].replace('_', ' ', inplace=True)

kernels['Title'].replace('.', ' ', inplace=True)



more=[' ',"'s","''","``",'-','_','‘’',',']

stop= set(stopwords.words('english')+list(string.punctuation)+more)

words=list()

for i in range(0,len(kernels)):

    for w in word_tokenize(kernels['Title'][i]):

        if w.lower() not in stop:

            words.append(w.lower())

            

frequent=pd.DataFrame(pd.Series(nltk.FreqDist(words)))

frequent.reset_index(inplace=True)

frequent.sort_values(by=(0),ascending=False, inplace= True)



f,ax=plt.subplots(figsize=(20,8))

f=sns.barplot(frequent['index'][:12],frequent[0][:12])

a=ax.set(title='Barplot of the most used words',xlabel='Words', ylabel='')
#wordcloud

wordcloud = WordCloud(background_color = 'white',width=2000, height=900, max_words = 100).generate(' '.join(kernels['Title']))

f,ax=plt.subplots(figsize = (200, 120) ) 

plt.imshow(wordcloud) 

a=plt.axis('off')
kernelsBins=kernels.copy()

bins=[-1,0,100000]

labels=['1','2']

kernelsBins['TotalVotes']= pd.cut(kernelsBins["TotalVotes"], bins , labels=labels)



#We only take the most recent 100k as the old data might be misleading and/or out of date

recentKernels=kernelsBins[-100000:].copy()





#Sampling 15k data with same distribution as the population

sampleML=kernelsBins[320000:335000]

sampleML.reset_index(inplace=True, drop=True)



#Displaying the percentage

sampPercent=sampleML.loc[sampleML['TotalVotes']=='2'].count() [1]/sampleML['TotalVotes'].count() 

fullPercent=recentKernels.loc[recentKernels['TotalVotes']=='2'].count() [1] /recentKernels['TotalVotes'].count() 



print('General percentage' ,fullPercent*100,'%')

print('Percentage of the sample',sampPercent*100,'%')

#extracting the word frequence distribution to create features later on

words=list()

for i in range(0,len(sampleML)):

    for w in word_tokenize(sampleML['Title'][i]):

        if w.lower() not in stop:

            words.append(w)

            

freq=list(nltk.FreqDist(words).keys())[:10000]



#a function to find the features from the words

def find_feat(text):

    word=set(text)

    features={}

    for w in freq:

        features[w]=(w in word)

    return features



#a function to clean the data from stopwords after tokenizing

def clean(text):

    l=list()

    for w in word_tokenize(text):

        if w.lower() not in stop:

            l.append(w)

    return l

            

# preprocessing, extracting features and splitting the data into train/test 

documents=list()

for i in range(0,len(sampleML)):

     documents.append([clean(sampleML['Title'][i]), sampleML['TotalVotes'][i]])



doc=list()

for i in range(0,len(sampleML)):

    for w in clean(sampleML['Title'][i]):

        doc.append(w)

ratio=int(len(doc)/13)



random.shuffle(documents)

featuresets = [(find_feat(title), votes) for (title, votes) in documents]

random.shuffle(featuresets)



split=int(len(featuresets)-len(featuresets)/5)

train=featuresets[:split]

test=featuresets[split:]



clf_naiveBayes=nltk.NaiveBayesClassifier.train(train)

print("Classifier accuracy percent:",(nltk.classify.accuracy(clf_naiveBayes, test))*100)
clf_naiveBayes.show_most_informative_features(20)
#we use 100k data 

#we clean and prepare data before using it

recentKernels.reset_index(drop=True,inplace=True)

for i in range (0,len(recentKernels)):

    recentKernels['Title'][i]=clean(recentKernels['Title'][i])

for i in range (0,len(recentKernels)):

    space=' '

    recentKernels['Title'][i]=space.join(recentKernels['Title'][i])







max_features = 100000

tokenizer = Tokenizer(nb_words=max_features, split=' ')

tokenizer.fit_on_texts(recentKernels['Title'].values)

X = tokenizer.texts_to_sequences(recentKernels['Title'].values)

X = pad_sequences(X, maxlen=15)

Y = pd.get_dummies(recentKernels['TotalVotes']).values

x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state = 42,shuffle=True)



print(x_train.shape,y_train.shape)
#LSTM squential model

lstmModel = Sequential()

lstmModel.add(Embedding(max_features, 100,input_length=X.shape[1]))



lstmModel.add(LSTM(100, dropout=0.3, recurrent_dropout=0.2))



lstmModel.add(Dense(2,activation='softmax'))

lstmModel.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

#Using categorical_crossentropy for our classification problem



earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

#Define an earlystop as our data does not carry an obvious pattern and might give our model a hard time learninf



print(lstmModel.summary())

history=lstmModel.fit(x_train, y_train, validation_split=0.1 ,batch_size=256, epochs=100, callbacks=[earlystop])



score,accu = lstmModel.evaluate(x_test, y_test, batch_size = 64)

print((accu)*100)
f=plt.figure(figsize=(20,6))

plt.plot(history.history['accuracy'], label='train', c='green',linewidth=3)

plt.plot(history.history['val_accuracy'], label='test' , c='blue',linewidth=3)

plt.title('LSTM model Accuracy')

a=plt.legend()
# GRU with glove embeddings and two dense layers

gruModel = Sequential()

gruModel.add(Embedding(max_features, 100,input_length=X.shape[1]))



gruModel.add(GRU(50,dropout=0.1, recurrent_dropout=0.2, return_sequences=True))

gruModel.add(GRU(100, dropout=0.1, recurrent_dropout=0.2))





gruModel.add(Dense(2,activation='softmax'))

gruModel.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

#Using categorical_crossentropy for our classification problem



earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

#Define an earlystop as our data does not carry an obvious pattern and might give our model a hard time learninf



print(gruModel.summary())

history=gruModel.fit(x_train, y_train, validation_split=0.1, batch_size=256, epochs=100, callbacks=[earlystop])



score,acc = gruModel.evaluate(x_test, y_test, batch_size = 64)

print((acc)*100)
plt.figure(figsize=(20,6))

plt.plot(history.history['accuracy'], label='train', c='green',linewidth=3)

plt.plot(history.history['val_accuracy'], label='test' , c='blue',linewidth=3)

plt.title('gru model Accuracy')

a=plt.legend()
newTitles=[['There is only one thing we say to death: Not today'],

          ['How to choose a topic and layout for a new Kaggle kernel'],

          ['Kaggle meta data analysis to boost your kernel exposure'],

          ['What is dead may never die'],

          ['Attracting Kaggle users - Using plotly visualisations and DL NLP predictions'],#Using the words from the NaiveBayes suggestions

          ['You know nothing, Jon Snow']] 



labels=['0 votes','1 vote or more']



print ('LSTM model')

for i in range (0,len(newTitles)):

    title=newTitles[i]

    sequence=tokenizer.texts_to_sequences(title)

    padded=pad_sequences(sequence, maxlen=15)

    prediction=lstmModel.predict(padded)

    print( title,'   ==>   ',labels[np.argmax(prediction)],'\n')

    

print ('\n \ngru model')

for i in range (0,len(newTitles)):

    title=newTitles[i]

    sequence=tokenizer.texts_to_sequences(title)

    padded=pad_sequences(sequence, maxlen=15)

    prediction=gruModel.predict(padded)

    print( title,' ==> ',labels[np.argmax(prediction)],'\n')
