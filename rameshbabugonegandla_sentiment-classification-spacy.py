from IPython.display import Image

import os

Image("../input/spacyimage/Introduction.PNG")
# Importing the packages

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

import string

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

from sklearn import metrics

import seaborn as sns

import matplotlib.pyplot as plt
# Reading the dataset

df = pd.read_csv('../input/nlpdata/nlp_data.csv')

print(df.head(2))
# https://cmdlinetips.com/2018/11/how-to-split-a-text-column-in-pandas/

# https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/?ref=rp

sentiment_data = pd.read_csv("../input/nlpdata/nlp_data.csv", names = ["comments_text"])

print(sentiment_data.head(2))



#sentiment_data['comments'] = sentiment_data.comments_text.str.split('\t',expand=True,)

sentiment_data = sentiment_data["comments_text"].str.split(" ", n = 1, expand = True) 

sentiment_data.head(2)

# Getting the head of the dataset

sentiment_data.head(2)
# Shape of the dataset

sentiment_data.shape
sentiment_data.dtypes
sentiment_data.rename(columns={0:'label',1:'comments'}, inplace=True)
# Value counts for Target column

print('Label: 0-Negative Sentiment,1-Positive Sentiment\n')

print(sentiment_data['label'].value_counts())

# checking for null values

sentiment_data.isnull().sum()
import spacy

from spacy import displacy

nlp = spacy.load('en_core_web_sm')



doc = nlp(sentiment_data['comments'][0])

displacy.render(doc, style="dep" , jupyter=True)
# Create an nlp object

doc = nlp(sentiment_data['comments'][0])

 

# Iterate over the tokens

for token in doc:

    # Print the token and its part-of-speech tag

    print(token.text, "-->", token.dep_)
# Create an nlp object

doc = nlp(sentiment_data['comments'][0])



# Iterate over the tokens

for token in doc:

    # Print the token and its part-of-speech tag

    print(token.text, "-->", token.lemma_)
# Create an nlp object

doc = nlp(sentiment_data['comments'][2])

 

sentences = list(doc.sents)

len(sentences)
# Create an nlp object

doc = nlp(sentiment_data['comments'][2])



for ent in doc.ents:

    print(ent.text, ent.start_char, ent.end_char, ent.label_)
# Create an nlp object

doc = nlp(sentiment_data['comments'][0])

entities=[(i, i.label_, i.label) for i in doc.ents]

print('Entities')

print(entities)

print('\nDocument at Index 0')

print(doc)

print('\nEntities Captured')

displacy.render(doc, style = "ent",jupyter = True)
# Create an nlp object

doc = nlp(sentiment_data['comments'][0])



for token in doc:

    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
doc = nlp(sentiment_data['comments'][0])



for token1 in doc:

    for token2 in doc:

        print(token1.text, token2.text, token1.similarity(token2))
from IPython.display import YouTubeVideo

YouTubeVideo('ZIiho_JfJNw')
# Creating our list of punctuations

punc = string.punctuation



# Creating our list of stopwords

nlp = spacy.load('en_core_web_sm')

stopwords = spacy.lang.en.stop_words.STOP_WORDS



# Loading parser

parse = English()



# Creating the tokenizer function

def tokenize(token):

    # Creating the token object where all the tokenization functions are applied starting 

    # with parsing

    token_obj = parse(token)

    

    # Lemmatizing each token and converting it into lowercase

    token_obj = [i.lemma_.lower().strip() if i.lemma_!='-PRON-' else i.lower_ for i in token_obj]

    

    # Removing stop words and punctuations

    token_obj = [i for i in token_obj if i not in stopwords and i not in punc]

    

    # Returning the token

    return token_obj

# Creating custom transformer using spaCy

class transformer(TransformerMixin):

    def transform(self,X,**transform_params):

        

        # Cleaning the text

        return [clean(i) for i in X]

    # Fitting the transformer

    

    def fit(self,X,y=None,**fit_params):

        return self

    

    # Predicting the transformer

    def get_params(self,deep=True):

        return{}

    

# Basic clean function

def clean(i):

    

    # Removing the spaces and converting all the text into lowercase

    return i.strip().lower()
# Creating Count Vectorizer

count_vector = CountVectorizer(tokenizer=tokenize,ngram_range=(1,2))
# Creating TF-IDF Vectorizer

tfidf_vect= TfidfVectorizer( tokenizer=tokenize,use_idf=True, smooth_idf=True, sublinear_tf=False)
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english')

vect.fit(sentiment_data['comments'])

idf = vect._tfidf.idf_

wordDict=dict(zip(vect.get_feature_names(), idf))

print(wordDict)
from IPython.display import YouTubeVideo

YouTubeVideo('vEmm9fZJuuM')
word1={k: v for k, v in wordDict.items() if v < 5}

from wordcloud import WordCloud

wordcloud = WordCloud(width = 1200, height = 800, 

                background_color ='black', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(word1.keys()))

plt.title('Review Comments')

# plot the WordCloud image                        

plt.imshow(wordcloud); 

word1={k: v for k, v in wordDict.items() if v > 5 and v < 10}

from wordcloud import WordCloud

wordcloud = WordCloud(width = 1200, height = 800, 

                background_color ='black', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(word1.keys()))

plt.title('Review Comments')

# plot the WordCloud image                        

plt.imshow(wordcloud); 
# Importing train_test split

from sklearn.model_selection import train_test_split



# Assigning X and Y values



x = sentiment_data['comments']# the feature we want to analyze

y = sentiment_data['label'] # the labels



# Splitting the values into train and test

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2,test_size = 0.25)
# Importing Random Forest Classifier and fitting the model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced')



# Importing RandomizedSearchCV and assigning the parameters

from sklearn.model_selection import RandomizedSearchCV

params = {'criterion':['entropy','gini'],'max_depth':range(1,15,2)}



# Fitting the RandomizedSearchCV model

rfc = RandomizedSearchCV(rf,params)
# Creating the pipeline

pipe_rf = Pipeline([('clean',transformer()),

                 ('vectorizer',tfidf_vect),

                 ('model',rfc)])



# Model generation

pipe_rf.fit(x_train,y_train)
# Importing the metrics

from sklearn import metrics



# Predicting with test data

pred = pipe_rf.predict(x_test)
# Model accuracy,precision and recall

print('Accuracy',metrics.accuracy_score(y_test,pred))

print('Precision',metrics.precision_score(y_test,pred,average=None))

print('Recall',metrics.recall_score(y_test,pred,average=None))

print('F1-Score',metrics.f1_score(y_test,pred,average=None))
# Confusion matrix

conf_matrix_rf = metrics.confusion_matrix(y_test,pred)

print(conf_matrix_rf)
# Plotting the confusion matrix

cfm= conf_matrix_rf

lbl1=["Predicted Negative", "Predicted Positive"]

lbl2=["Actual Negative", "Actual Poistive"]

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(cfm, annot=True, cmap="Blues", fmt="d", xticklabels=lbl1, yticklabels=lbl2)

ax.set_ylim([0,2])

plt.show()
# Importing Support Vector Classifier algorithm

from sklearn.svm import SVC

svc = SVC(class_weight='balanced')



params1 = {'kernel':['linear','rbf','poly','sigmoid'],'C': [0.01, 0.1, 1,10],'gamma': [0.01,0.1,1,10]}



# Fitting the RandomizedSearchCV model

svcc = RandomizedSearchCV(svc,params1)
# Creating the pipeline

pipe = Pipeline([('clean',transformer()),

                 ('vectorizer',tfidf_vect),

                 ('model',svcc)])



# Model generation

pipe.fit(x_train,y_train)
# Predicting with test data

pred = pipe.predict(x_test)



# Model accuracy,precision and recall

print('Accuracy',metrics.accuracy_score(y_test,pred))

print('Precision',metrics.precision_score(y_test,pred,average=None))

print('Recall',metrics.recall_score(y_test,pred,average=None))

print('F1-Score',metrics.f1_score(y_test,pred,average=None))
# Confusion matrix

conf_matrix = metrics.confusion_matrix(y_test,pred)

print(conf_matrix)
# Plotting the confusion matrix

conf= conf_matrix

lbl1=["Predicted Negative", "Predicted Positive"]

lbl2=["Actual Negative", "Actual Poistive"]

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(conf, annot=True, cmap="Blues", fmt="d", xticklabels=lbl1, yticklabels=lbl2)

ax.set_ylim([0,2])

plt.show()