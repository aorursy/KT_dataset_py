import pandas as pd

import numpy as np



import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns



from bs4 import BeautifulSoup as bsp

import datetime as dt

from collections import Counter



from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud, STOPWORDS 



import warnings

warnings.filterwarnings('ignore')



import re





%matplotlib inline
data = pd.read_csv("../input/60k-stack-overflow-questions-with-quality-rate/data.csv")

data.head(10)
data.info()
data.shape
data.describe(include='object').transpose()
data['Body'][0]
data['Tags'][10]
data['CleanBody'] = data['Body'].apply(lambda x:bsp(x,'html.parser').text.replace('\\','').replace('\n','').replace('\t','').replace('\r',''))

data['Tags'] = data['Tags'].apply(lambda x:x.replace('<','').replace('>',' '))
data['CreationDate'] = pd.to_datetime(data['CreationDate'],errors='coerce')



data['CreationMonth'] = data['CreationDate'].dt.month

data['CreationDay'] = data['CreationDate'].dt.day

data['CreationYear'] = data['CreationDate'].dt.year

data['CreationQuater'] = data['CreationDate'].dt.quarter

data['CreationDate'] = data['CreationDate'].dt.date
fig = px.histogram(data,x='CreationDate',title='Question\'s Quality year by year',color='Y')



fig.update_layout(height = 700)

fig.update_xaxes(categoryorder='category descending',title='Date').update_yaxes(title='Number of Questions')



fig.show()
fig = px.histogram(data,x='CreationDate',title='Number of Question year by year')



fig.update_layout(height = 700)

fig.update_xaxes(categoryorder='category descending',title='Date').update_yaxes(title='Number of Questions')



fig.show()
fig = px.histogram(data,x='CreationDate',title='Question\'s Quality year by year',color='Y',nbins=6,barmode = 'group')



fig.update_layout(height = 700)

fig.update_xaxes(categoryorder='category descending',title='Date').update_yaxes(title='Number of Questions')



fig.show()
allTags = data['Tags'].apply(lambda x: x.lower())

allTags = allTags.values

allTags = list(allTags)

allTags = ''.join(allTags)

count = Counter(allTags.split())





count = pd.DataFrame(list(dict(count).items()),columns = ['Technology','Count'])

count.astype({'Count':'int64'})



count.sort_values('Count',axis =0,ascending = False,inplace = True)
fig = px.scatter(count[:20], x = 'Technology',y='Count',size='Count',color='Count')



fig.update_layout(title='Top 20 Technologies',xaxis=dict(title='Technology'),yaxis=dict(title='No. of Questions'))



fig.show()
title = data['Title'].apply(lambda x:len(x.split()))



fig = px.histogram(x=title.values,title='Length Distribution of  Question')





fig.update_xaxes(title='Length Of Question').update_yaxes(title='No. Of Question')

fig.show()
cleanBd = data['CleanBody'].apply(lambda x:len(x.split()))



fig = px.histogram(x=cleanBd.values,title = 'Length Distribution of Clean Body')



fig.show()
tags = data['Tags'].apply(lambda x:len(x.split()))



fig = px.histogram(x= tags.values,title= ' Length Distribution of Tags')

fig.show()
# Word Cloud with Stop words

stopwords = set(STOPWORDS) 

def WordCloudSW(values):

    wordcloud = WordCloud(width = 500, height = 300, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(values) 

    

    plt.figure(figsize=(19,9))



    plt.axis('off')

    plt.title("Without Stop Words")

    plt.imshow(wordcloud)

    plt.show()
# Word Cloud without Stop words



def word_cloud(values):

    wordcloud = WordCloud(width = 500, height = 300, 

                background_color ='white', 

                min_font_size = 10).generate(values) 

    

    plt.figure(figsize=(19,9))



    plt.axis('off')

    plt.title("With Stop Words")

    plt.imshow(wordcloud)

    plt.show()


js = data[data['Tags'].str.contains('javascript')]['CleanBody'].values



js = ' '.join(list(js))

word_cloud(js)# with stop words
WordCloudSW(js)# without stop words
python = data[data['Tags'].str.contains('python')]['CleanBody']

python = ''.join(list(python.values))

word_cloud(python)
WordCloudSW(python)# without stop words
# Unigrams before removing the stopwords

def getTopNWords(corpus,n=20):

    vec = CountVectorizer().fit(corpus)

    bow = vec.transform(corpus)

    bow = bow.sum(0)

    

    top = [(word,bow[0,idx]) for word,idx in vec.vocabulary_.items()]

    top = sorted(top,key = lambda x:x[1],reverse = True)

    top = pd.DataFrame(top[:n],columns = ['Words','Count'])

    return top
# Unigrams after removing the stopwords

def getTopNWordsSW(corpus , n = 20):

    vec = CountVectorizer(stop_words= stopwords).fit(corpus)

    bog = vec.transform(corpus)

    

    bog = bog.sum(axis=0)

    bog = [(word,bog[0,idx]) for word,idx in vec.vocabulary_.items()]

    

    bog = sorted(bog,key = lambda x:x[1],reverse = True)

    

    top = pd.DataFrame(bog[:n],columns = ['Words','Count'])

    return top


title = getTopNWords(data['Title'])





fig = px.bar(title, x='Words',y = 'Count')

fig.update_layout(title= 'Most Ocurring Words in Title Before Removing StopWords',

                 xaxis=dict(title='Words'),

                 yaxis=dict(title='Counts'))

fig.show()
title = getTopNWordsSW(data['Title'])





fig = px.bar(title, x='Words',y = 'Count')

fig.update_layout(title= 'Most Ocurring Words in Title After Removing StopWords',

                 xaxis=dict(title='Words'),

                 yaxis=dict(title='Counts'))

fig.show()
title = getTopNWords(data['CleanBody'])





fig = px.bar(title, x='Words',y = 'Count')

fig.update_layout(title= 'Most Ocurring Words in CleanBody Before Removing StopWords',

                 xaxis=dict(title='Words'),

                 yaxis=dict(title='Counts'))

fig.show()
title = getTopNWordsSW(data['CleanBody'])





fig = px.bar(title, x='Words',y = 'Count')

fig.update_layout(title= 'Most Ocurring Words in CleanBody After Removing StopWords',

                 xaxis=dict(title='Words'),

                 yaxis=dict(title='Counts'))

fig.show()
#before removing stop words

def bigramNWord(corpus,n = 20):

    vec = CountVectorizer(ngram_range=(2,2)).fit(corpus)

    bog = vec.transform(corpus)

    

    bog = bog.sum(0)

    

    bog = [(word,bog[0,idx]) for word,idx in vec.vocabulary_.items()]

    

    bog = sorted(bog,key = lambda x:x[1],reverse = True )

    

    bog = pd.DataFrame(bog[:n],columns = ['Word','Count'])

    

    return bog
#before removing stop words

def bigramNWordSW(corpus,n = 20):

    vec = CountVectorizer(ngram_range=(2,2),stop_words=stopwords).fit(corpus)

    bog = vec.transform(corpus)

    

    bog = bog.sum(0)

    

    bog = [(word,bog[0,idx]) for word,idx in vec.vocabulary_.items()]

    

    bog = sorted(bog,key = lambda x:x[1],reverse = True )

    

    bog = pd.DataFrame(bog[:n],columns = ['Word','Count'])

    

    return bog
title = bigramNWord(data['Title'])



fig= px.bar(title,x ='Word',y = 'Count')

fig.update_layout(title = 'Bigram of Title Before Removing Stop Words',

                 xaxis = dict(title='Bigram Words'),

                 yaxis = dict(title='Counts'))

fig.show()
title = bigramNWordSW(data['Title'])



fig= px.bar(title,x ='Word',y = 'Count')

fig.update_layout(title = 'Bigram of Title After Removing Stop Words',

                 xaxis = dict(title='Bigram Words'),

                 yaxis = dict(title='Counts'))

fig.show()
title = bigramNWord(data['CleanBody'])



fig= px.bar(title,x ='Word',y = 'Count')

fig.update_layout(title = 'Bigram of CleanBody Before Removing Stop Words',

                 xaxis = dict(title='Bigram Words'),

                 yaxis = dict(title='Counts'))

fig.show()
title = bigramNWordSW(data['CleanBody'])



fig= px.bar(title,x ='Word',y = 'Count')

fig.update_layout(title = 'Bigram of CleanBody After Removing Stop Words',

                 xaxis = dict(title='Bigram Words'),

                 yaxis = dict(title='Counts'))

fig.show()
# for NLP 



import emoji



from nltk import pos_tag

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet,stopwords

from nltk import word_tokenize,sent_tokenize

import nltk

from nltk.tokenize import regexp_tokenize 





# for Making Prediciton Models 

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense,Flatten,LSTM,Dropout,Embedding

from keras.models import Sequential,load_model

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score,accuracy_score,confusion_matrix,classification_report,recall_score

from sklearn.preprocessing import MultiLabelBinarizer



from collections import Counter
data['Y'] = data['Y'].map({'LQ_CLOSE':0,'LQ_EDIT':1,'HQ':2})





data['question'] = data['Title']+data['CleanBody']
slangs = {}

slangs_df = pd.read_csv('../input/allslangs/AllSlangs.csv')

slangs_df.drop('Unnamed: 0',1,inplace=True)

slangs_df.dropna(inplace=True)



for index,rows in slangs_df.iterrows():slangs[str(rows['Abbreviation'].replace(' ',''))] = str(rows['FullForm'].lower()).strip(' ')
CONTRACTION_MAP = {

"ain't": "is not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he he will have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"I'd": "I would",

"I'd've": "I would have",

"I'll": "I will",

"I'll've": "I will have",

"I'm": "I am",

"I've": "I have",

"i'd": "i would",

"i'd've": "i would have",

"i'll": "i will",

"i'll've": "i will have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she would",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as",

"that'd": "that would",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have",

"n't":'not' 

}
class PreprocessingData:

  # ----------------------------- Removing Slangs from Raw Text -----------------------------------------

  def removingSlangs(self,reviews):

    doc = regexp_tokenize(str(reviews), "[\w']+") 



    for token in doc:

      if(token in CONTRACTION_MAP):

        reviews = reviews.replace(token,CONTRACTION_MAP[token])

      elif(token in slangs):

        reviews = reviews.replace(token,slangs[token])

    return reviews



  

  # ----------------------------- Part of Speech Tagging -----------------------------------------

  def get_wordnet_pos(self,pos_tag):

    if pos_tag.startswith('J'):

        return wordnet.ADJ

    elif pos_tag.startswith('V'):

        return wordnet.VERB

    elif pos_tag.startswith('N'):

        return wordnet.NOUN

    elif pos_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN



        

  

  # ----------------------------- Removing StopWords and Lemmatizing Words -----------------------------------------

  def CleaningData(self,val):

    rex = re.sub(r'[^a-zA-Z0-9]+',' ',str(val))



    pos = pos_tag(word_tokenize(rex))



    filter = [WordNetLemmatizer().lemmatize(x[0],PreprocessingData.get_wordnet_pos(self,x[1])) for x in pos if x[0] not in stopwords.words('english')]



    filter = ' '.join(filter)



    return filter







#ppd = PreprocessingData()

#data['question']= data['question'].apply(lambda x:ppd.CleaningData(x))

#data['question']= data['question'].apply(lambda x:ppd.removingSlangs(x))

#data.to_csv('pp_data.csv',index = False)

data = pd.read_csv('../input/preprocessed-data/pp_data.csv')
allQuestions = data['question'].apply(lambda x:x.lower())

allQuestions = list(allQuestions.values)

allQuestions = ''.join(allQuestions)

count = Counter(allQuestions.split())

len(count) # unique vocabulary
vocab = 20000

embedding_dim = 500

text_length = 100

trunc_type = 'post'

padding_type = 'post'

oov_tok = '<OOV>' # OOV = Out of Vocabulary

training_portion = .8
token = Tokenizer(num_words=vocab,lower = True,oov_token =oov_tok )

token.fit_on_texts(data['question'])

seq = token.texts_to_sequences(data['question'])

padded = pad_sequences(seq,maxlen = text_length,padding = 'post',truncating='post')
y = to_categorical(data['Y'])

y
xTrain,xTest ,yTrain,yTest = train_test_split(padded,y,test_size = 0.2,random_state = 2)
model = Sequential()



model.add(Embedding(vocab,embedding_dim,input_length=xTrain.shape[1]))

model.add(Dropout(0.2))

model.add(LSTM(250))

model.add(Dropout(0.3))

model.add(Dense(20))

model.add(Dense(3,activation='softmax'))



model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()
call = EarlyStopping(monitor='val_loss',patience = 10,verbose = 0)



#model.fit(xTrain,yTrain,epochs = 20,verbose=1,callbacks=[call],validation_data=(xTest,yTest))
#model.save('multiclass.h5')



mm1 = load_model('../input/weights/multiclass.h5')
pred = mm1.predict_classes(xTest)



import numpy as np

rounded_labels=np.argmax(yTest, axis=1)

rounded_labels
confusion_matrix(rounded_labels,pred)
accuracy_score(rounded_labels,pred)
allTags = data['Tags'].apply(lambda x:x.lower())

allTags = list(allTags.values)

allTags = ''.join(allTags)

count = Counter(allTags.split())

len(count) # unique Tags
count = pd.DataFrame(list(dict(count).items()),columns = ['Technology','Count'])

count.astype({'Count':'int64'})



count.sort_values('Count',axis =0,ascending = False,inplace = True)
fig = px.scatter(count, x = 'Technology',y='Count',size='Count',color='Count')



fig.update_layout(title='All Technologie\'s-Tags and their Frequency',xaxis=dict(title='Technology'),yaxis=dict(title='No. of Questions'))



fig.show()
count = count[count['Count']>=1000]
fig = px.scatter(count, x = 'Technology',y='Count',size='Count',color='Count')



fig.update_layout(title='Technologie\'s have Frequency greater than 1000',xaxis=dict(title='Technology'),yaxis=dict(title='No. of Questions'))



fig.show()
tech = list(count['Technology'].values)

len(tech) # we have 23 multilabe which cover most of the questions
def FinalTags(tags):

  t = list()

  for tag in tags.split():

    if (tag.lower() in tech):

      t.append(tag)

  if(t==[]):

    return None

  else:

    return t

data['Tags'] = data['Tags'].apply(FinalTags)

data.isna().sum()
data.dropna(inplace= True,axis= 0)

data.isna().sum()
multi = MultiLabelBinarizer()

y = multi.fit_transform(data['Tags'])
token = Tokenizer(vocab,lower=True,oov_token=oov_tok)

token.fit_on_texts(data['question'])

seq = token.texts_to_sequences(data['question'])

padded = pad_sequences(seq,padding = 'post',truncating='post',maxlen=text_length)



padded.shape
xTrain,xTest,yTrain,yTest = train_test_split(padded,y,test_size = 0.2,random_state = 2)
model = Sequential()



model.add(Embedding(vocab,embedding_dim,input_length=xTrain.shape[1]))

model.add(Dropout(0.3))

model.add(LSTM(250))

model.add(Dropout(0.3))

model.add(Dense(100))

model.add(Dense(50))

model.add(Dense(yTrain.shape[1]))



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
call = EarlyStopping(monitor='val_loss',verbose=0,patience=6)



#model.fit(xTrain,yTrain,epochs=20,validation_data=(xTest,yTest),callbacks=[call],verbose = 1)
#model.save('Labels.h5')

mm2 = load_model('../input/weights/Labels.h5')
rounded_labels=np.argmax(yTest, axis=1)

pred = mm1.predict_classes(xTest)
confusion_matrix(rounded_labels,pred)