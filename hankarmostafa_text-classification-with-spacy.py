import datetime
import spacy 
import string
import seaborn as sb 
import matplotlib.pyplot as plt 
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
alexa=pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv',sep='\t')
# show infos about the df 
alexa.head()
#get the shape 
alexa.info()
alexa.shape
#check if there is null values 
alexa.isnull().sum()
#No empty values 
# how many feedbacks per class 
#1==pos
#0==neg
alexa.feedback.value_counts()
sb.countplot(data=alexa,x='feedback')
# class per variation 
alexa.rating.value_counts()
sb.countplot(data=alexa,x='rating',)
# define punctiations 
punctuations = string.punctuation
# load the spacy model
nlp = spacy.load('en_core_web_sm')
# defien stop words 
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(text):
    # Creating our token object, which is used to create documents with linguistic annotations.
    tokens = parser(text)

    # Lemmatizing each token and converting each token into lowercase
    lemmas = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens ]

    # Removing stop words
    clean_lemmas = [ word for word in lemmas if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return to_string(clean_lemmas)

# Define a function that turn list of tokens to a String object 
def to_string(lemmatized_text):
    return ' '.join([str(elem) for elem in lemmatized_text]) 
     
# Now Let's clean our reviews 
alexa['clean_text']= alexa['verified_reviews'].apply(spacy_tokenizer)
alexa.sample(5)
# create a bag of words by a Countvectorizer 
bow = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1)).fit_transform(alexa['clean_text'])


#print the spars matrix 
bow.shape
# Now let's pass the bow to TfIDf Transformer  
tfidf_transformer= TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(bow)

print(tfidf.shape)
from sklearn.model_selection import train_test_split

X = alexa['verified_reviews'] 
y = alexa['feedback'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=spacy_tokenizer)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', RandomForestClassifier(n_estimators=600))
])

#Fit the model 
pipeline.fit(X_train,y_train)
#make predictions 
preds=pipeline.predict(X_test)
print(preds[:10])
from sklearn import metrics
# Model Accuracy
print(" Classification report: \n ",metrics.classification_report(y_test, preds))
print('\n')
print("Confusion Matrix  :",metrics.confusion_matrix(y_test, preds))
print(" Accuracy Score :" ,metrics.accuracy_score(y_test, preds,))
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=spacy_tokenizer)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train,y_train)
predictions=pipeline.predict(X_test)
# Model Accuracy
print(" Classification report :",metrics.classification_report(y_test, predictions))
print("Confusion Matrix  :",metrics.confusion_matrix(y_test, predictions))
print(" Accuracy Score :" ,metrics.accuracy_score(y_test, predictions,))
#submit our predictions 
pred_test=pd.Series(preds)
submission = pd.DataFrame({'Id':pred_test.index, 'feedback ': pred_test.values})
submission.to_csv('submission.csv', index=False)
print(" Submission  successfully saved!")
submission.sample(10)
