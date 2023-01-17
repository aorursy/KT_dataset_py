#libraries

import pandas as pd

import spacy

from spacy import displacy

import seaborn as sns



dtrain=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

sample=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

dtest=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
#inspection of train data

dtrain.info()
sns.countplot(y='target',data=dtrain,palette='Set3')#are there more tweets classified as 0 or as 1?
#from where are these tweets?

sns.countplot(y='location',data=dtrain,palette='Set3',order=dtrain['location'].value_counts().iloc[:6].index)
#this is for replace the cities by their country

dtrain['location']=dtrain['location'].replace(['United States','New York','Los Angeles','Los Angeles, CA', 'Washington, DC'],'USA')

dtrain['location']=dtrain['location'].replace(['London'],'UK')

dtrain['location']=dtrain['location'].replace(['Mumbai'],'India')
sns.countplot(y='location',data=dtrain,palette='Set3',order=dtrain['location'].value_counts().iloc[:6].index)
sns.countplot(y='keyword',data=dtrain,palette='Set2',order=dtrain['keyword'].value_counts().iloc[:3].index)#Top 3 of keywords more used
nlp = spacy.load("en_core_web_sm")
#First one: Entity Recognition

doc=nlp(dtrain['text'][58])

displacy.render(doc,style='ent')
doc=nlp(dtrain['text'][10])

displacy.render(doc,style='ent')
#linguistic annotations

tokenized_text = pd.DataFrame()

#describe the words in the sentence before

for i, token in enumerate(doc):

    tokenized_text.loc[i, 'text'] = token.text

    tokenized_text.loc[i, 'type'] = token.pos_

    tokenized_text.loc[i, 'lemma'] = token.lemma_,

    tokenized_text.loc[i, 'is_alphabetic'] = token.is_alpha

    tokenized_text.loc[i, 'is_stop'] = token.is_stop

    tokenized_text.loc[i, 'is_punctuation'] = token.is_punct

    tokenized_text.loc[i, 'sentiment'] = token.sentiment

    

    



tokenized_text[:30]
#dependency parser- see the relations between the words 

displacy.render(doc,style='dep',jupyter='true')
#if you don't understand a tag displayed

spacy.explain('ADP')
import string

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English
#A-FIRST STEP: TOKEN THE DATA. We are going to remove stopwords and puntuaction from each sentence.



# Create a list of punctuation marks

punctuations = string.punctuation



# Create a list of stopwords

stop_words = spacy.lang.en.stop_words.STOP_WORDS







# Load English tokenizer

tokenizer = English()



# Creating a tokenizer function with the ones defined before

def text_tokenizer(sentence):

    # Creating the token object

    tokens = tokenizer(sentence)



    # Lemmatizing each token if it is not a pronoun and converting each token into lowercase

    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens ]

    

    # Remove stop words

    tokens = [ word for word in tokens if word not in stop_words and word not in punctuations ]



    # return preprocessed list of tokens

    return tokens

#we want to clean more our data. For that, we will be creating a class predictors which inherits from sklearn TransformerMixin

from sklearn.base import TransformerMixin
# Custom transformer using spaCy

class CleanTextTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        # Cleaning Text

        return [clean_text(text) for text in X]



    def fit(self, X, y=None, **fit_params):

        return self



    def get_params(self, deep=True):

        return {}



# Basic function to clean the text

def clean_text(text):

    # Removing spaces and converting text into lowercase

    text = text.strip().replace("\n", " ").replace("\r", " ")

    text = text.lower()

    return text
#CountVectorizer converts a collection of text documents to a matrix of token counts

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

vectorizer = CountVectorizer(tokenizer = text_tokenizer, ngram_range=(1,1))
#Now we need to split our train dataset into train and validation data

X = dtrain['text'] 

y = dtrain['target'] 
from sklearn.model_selection import train_test_split

X_train, X_vald, y_train, y_vald = train_test_split(X, y, test_size=0.15)
from sklearn.pipeline import Pipeline

# we are going to use Linear Support Vector Classification

from sklearn.svm import LinearSVC

classifier = LinearSVC()



# Create a pipeline

pipeline = Pipeline([("cleaner", CleanTextTransformer()),

                 ('vectorizer', vectorizer),

                 ('classifier', classifier)])



# model generation

pipeline.fit(X_train,y_train)
from sklearn import metrics

# predict the X_vald

predictions = pipeline.predict(X_vald)



# model Accuracy

print("Linear Support Vector Classification Accuracy:",metrics.accuracy_score(y_vald, predictions))

#the code bellow is to create the submission file with the predictions made using the test dataset
predictionsFinal=pipeline.predict(dtest['text'])
sample['target'] = predictionsFinal
sample
sample.to_csv("submissionNLP.csv", index=False)