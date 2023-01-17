#Import numpy,pandas and nltk
#Download necessary nltk Components 

import numpy as np
import pandas as pd
import nltk
train_data=pd.read_csv("../input/nlp-getting-started/train.csv")
test_data=pd.read_csv("../input/nlp-getting-started/test.csv")

#Check the number of records and number of features
train_data.shape,test_data.shape
#Checkout the train dataframe
train_data.head()
#Checkout the test dataframe
test_data.head()
#Check for NULL values in Train data
train_data.isnull().sum()
#Check for NULL values in test data
test_data.isnull().sum()
#Lets explore keyword column now
train_data.keyword.unique()
#we will be considering the text columns only 
train_text=train_data.text
test_text=test_data.text
#extract dependent variable
y=train_data.target
import re
#Write some basic function to clean our texts
def clean_text(text):
    text=text.lower()
    text=re.sub('#','',text)
    text=re.sub('[^a-zA-Z ]','',text)
    return text
#check train data before cleaning
train_text.head()
#apply that cleaning function to our data
train_text=train_text.apply(clean_text)
test_text=test_text.apply(clean_text)
#Check data after cleaning
train_text.head()
#import stopwords and lemmatizer for lemmatization and remove some unnecessary keywords 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer=WordNetLemmatizer()
#lemmatize the words in each text,remove some unnecessary keywords like this,the,an etc and append the final sentence in a list called train_sequence
train_sequence=[]
for i in range(len(train_text)):
    words=nltk.word_tokenize(train_text.iloc[i])
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sent=' '.join(words)
    train_sequence.append(sent)
#Check length of train_sequence list
len(train_sequence)
#do the same procedure for test data
test_sequence=[]
for i in range(len(test_text)):
    words=nltk.word_tokenize(test_text.iloc[i])
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sent=' '.join(words)
    test_sequence.append(sent)
len(test_sequence)
#Import tfidf vectorizer from sklearn library
from sklearn.feature_extraction.text import TfidfVectorizer
#create tfidf object which produces 10000 features
tfidf=TfidfVectorizer(min_df=2,ngram_range=(1,3),max_features=10000)
# fit and transform train data
vectorized_train=tfidf.fit_transform(train_sequence)
vectorized_train.shape
#transform test data
vectorized_test=tfidf.transform(test_sequence)
vectorized_test.shape
#convert vectorized sparse matrix into an array
vectorized_train=vectorized_train.toarray()
vectorized_test=vectorized_test.toarray()
vectorized_train[0]
#import LogisticRegression
from sklearn.linear_model import LogisticRegression
#split our train data into train and test set
from sklearn.model_selection import train_test_split
#creating 20% of test data from our dataset
X_train, X_test, y_train, y_test = train_test_split(vectorized_train,y,test_size=0.2,random_state=0)
classifier=LogisticRegression(C=3)
#fit training data 
classifier.fit(X_train,y_train)
#evaluating our model
classifier.score(X_test,y_test)
#prediction of out test data
y_pred=classifier.predict(vectorized_test)
id = test_data.id
output_df=pd.DataFrame({'id':id,'target':y_pred})
output_df
#create submissoin file
output_df.to_csv("submission2.csv",index=False)




