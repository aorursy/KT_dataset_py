# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import spacy
train=pd.read_csv(r'/kaggle/input/train.csv')
train.head()
train.shape
nlp=spacy.load('en_core_web_sm')

stopwords=spacy.lang.en.stop_words.STOP_WORDS #List of stop words.
train.isnull().sum()
train.dropna(subset=['text'],inplace=True) # We will make our model based on the news content only,that why we are only worry about the missing data in text columns.
train=train[['text','label']]

train.head()
doc=nlp(train.text[0]) 

#Split the whole news into chunks of words,means this is for tockenization, this instance 

#also have many functionality later on you get familiar with these all.

#We can access everything tockens,parts of peech tag, named entities by interating over doc instance, and that the thing we will use in all Function.

# Now i can hope you will able to understand everyting.

def only_alpha(text):  # Keep only alphabets,keep in mind that it will keep alphabets of any country.

    doc=nlp(text)   

    alpha=[w.text for w in doc if w.text.isalpha() and w not in stopwords] #W.text to access tocken

    return ' '.join(alpha)



def count_pos_tag(text): #Returns the count of noun,proper noun and pronouns for each news.

    doc=nlp(text)

    tags=[w.pos_ for w in doc] # W.pos_ gives us part of speech tag for that word.

    return tags.count('NOUN'),tags.count('PROPN'),tags.count('PRON')



def count_named_entity(text): # Returns the numbers of person,location,organisation,political groups ,mentioned in the news content.

    doc=nlp(text)

    entity=[w.label_ for w in doc.ents] # W.label_ tell us entity eg. person or place or organisation

    return entity.count('PERSON'),entity.count('GPE'),entity.count('NORP'),entity.count('ORG')



def lemma(text):      #Lemmatize the word

    doc=nlp(text)

    lemma=[w.lemma_ for w in doc] #W.lemma_ lemmatize the word.

    return " ".join(lemma)



def preprocessing(data): #A combined function to all the text mining works.

    data=pd.DataFrame(data,columns=['text'])

    data.text=data.text.apply(only_alpha)

    data.text=data.text.str.lower()

    print('wooh, have Only alphabetic char ')

    data['noun_count'],data['pnoun_count'],data['pron_count']=np.array(list(data.text.apply(count_pos_tag))).T

    print('Tagged pos')

    data['num_person'],data['num_places'],data['num_national_gr'],data['num_organisation']=np.array(list(data.text.apply(count_named_entity))).T

    print('lemmatisation starts')

    data.text=data.text.apply(lemma)

    return data

    

feature=train.text #On which we will performe text cleaning.

label=train.label
import timeit

%timeit preprocessing(feature[:1])
500/7*len(feature)/1000/60 #It will take approx half an hour to complete all steps.
feature=feature.apply(only_alpha) #Removes all non-alphabetic chars
# You can see all the punctuation have been removed.

feature[:5]
feature=feature.str.lower() #Converts to lower case.

feature[:5]
feature=pd.DataFrame(feature)

#Adding columns for number of noun,proper noun and pronouns.

feature['noun_count'],feature['pnoun_count'],feature['pron_count']=np.array(list(feature.text.apply(count_pos_tag))).T

feature.head()
# Adding columns for count of each entity(person,place,organisation etc.)

feature['num_person'],feature['num_places'],feature['num_national_gr'],feature['num_organisation']=np.array(list(feature.text.apply(count_named_entity))).T
feature.text=feature.text.apply(lemma)#Lemmatize the words(tockens)

feature.head()
#Lets see how these all steps could be done by a single of code using preprocessing function

preprocessing(train.text[:10])
train=feature.join(label)

train.to_csv(r'trainv2.0.csv',index=False)#keep a copy of preprocessed data
train.head()
train.isnull().sum()
train.dropna(inplace=True)
def compare_plot(feature,ax=None):

    Fake_mean=train.loc[train.label==1,feature].mean()

    Real_mean=train.loc[train.label==0,feature].mean()

    sns.barplot(x=['Fake','Real'],y=[Fake_mean,Real_mean],ax=ax)

_,ax=plt.subplots(nrows=1,ncols=3,figsize=(15,5)) 

ax=ax.flatten()

compare_plot('noun_count',ax[0])

compare_plot('pnoun_count',ax[1])

compare_plot('pron_count',ax[2],)

for end,title in enumerate(['noun','propernoun','pronoun']):

    ax[end].set(title=title)
_,ax=plt.subplots(nrows=1,ncols=3,figsize=(15,5)) 

ax=ax.flatten()

compare_plot('num_person',ax[0])

compare_plot('num_place',ax[1])

compare_plot('num_organisation',ax[2],)

for end,title in enumerate(['person','place','organisation']):

    ax[end].set(title=title)
train.index=list(range(len(train)))

feature=train.drop('label',axis=1)

label=train.label

feature.shape
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#We, are already familiar with CountVectorizer and its working ,below is a short note on Tfidf.

#tfidf also works ecaxtly like the count vectorixer but give more weightaage to the rare words,thats the difference.
tfidf_vectorizer=TfidfVectorizer(ngram_range=(1,1),stop_words='english')

vectors=tfidf_vectorizer.fit_transform(feature.text)
vectors 

# This is a sparce matrix of tfidf corpus,it contains a huge amount of data of shape approx (200680,155000),

#so a 8gb ram unable to allocate that huge memory.We can review this to make another tfidf vector on lesser data.

#lets review it
tfidf_vectorizer.transform(feature.text[:1]).A #we can see must of the columns having 0 values,mut definetlly not all.so the dimensality reduction becomes a import steps
from sklearn.decomposition import NMF

from sklearn.preprocessing import Normalizer

norm=Normalizer()

nmf=NMF(n_components=50)

vectors=norm.fit_transform(vectors)

vectors=nmf.fit_transform(vectors)
vectors=pd.DataFrame(vectors)

vectors=vectors.join(feature.drop('text',axis=1))

vectors.head()
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score



lgb=LGBMClassifier()

X_train,X_test,y_train,y_test=train_test_split(vectors,label)

lgb.fit(X_train,y_train)

accuracy_score(y_test,lgb.predict(X_test))
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=5)

pipeline=make_pipeline(tfidf_vectorizer,norm,nmf)
accuracy=np.array([])

for trainind,testind in skf.split(feature,label):

    X_train,X_test,y_train,y_test=feature.iloc[trainind],feature.iloc[testind],label[trainind],label[testind]

    vector_train=pipeline.fit_transform(X_train.text)

    

    vector_train=pd.DataFrame(vector_train)

    vector_train.join(X_train.drop('text',axis=1))

    

    vector_val=pipeline.transform(X_test.text)

    vector_val=pd.DataFrame(vector_val)

    X_test.index=list(range(len(X_test)))

    vector_val.join(X_test.drop('text',axis=1))

    

    lgb.fit(vector_train,y_train)

    

    y_pred=lgb.predict(vector_val)

    accuracy=np.append(accuracy,accuracy_score(y_test,y_pred))  
accuracy
training_data=vectors #As we know we stored tfidf data of whole dataset in vectors.
training_data.to_csv(r'trainv2.1.csv')
lgb.fit(training_data,label)