import re

import pandas as pd

import string

import multiprocessing

from nltk.corpus import stopwords

from flashtext.keyword import KeywordProcessor

from sklearn.model_selection import train_test_split

from sklearn import metrics

import nltk

# libraries for dataset preparation, feature engineering, model training 

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



#import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers



nltk.download('stopwords')
df=pd.read_csv('../input/train.csv',encoding='utf-8')

df.head()
df['author'].value_counts()
# Define a function for removing regex

def regex_filtering(text):

        if text:

            #removing all email metadata fix it for email terms only

            text=re.sub(r"^(sender|to|copy|from|sent|subject|date|cc|e|von|datum|an|importance|bcc):.*$"," ",text,flags=re.M)

            #removing all mail ids

            text=re.sub(r"\S*@\S*\s?"," ",text)

            #removing all links

            text=re.sub(r"(((https?|ftp|file):\/\/)|www\\.)\\S+", ' ', text, flags=re.MULTILINE)

            text=re.sub(r"\w*\.\w{1,4}", '', text, flags=re.MULTILINE)

            #removing all non word character

            text=re.sub(r"([^a-zA-Z0-9\\u00C0-\\u00FF@]|[Ã£Ã¢])+",' ',text)

            #removing words with numbers 

            text=re.sub(r'\w*\d\w*', ' ', text)

            #removing single characters

            text=re.sub(r'\b\S{1}\s+',' ',text)

            #removing words with repeating characters

            text=re.sub(r'\b(\w)\1{1,}\s+',' ',text)

            #removing punkt

            text = text.translate(str.maketrans('','',string.punctuation))

            #removing extra whitespace

            text=re.sub(r"\s\s+",' ',text)

            #removing repeating words

            text=re.sub(r"(\w+\s+)\1{1,}",' ',text)

            #removing whitespaces

            text=text.strip()

            return text
#Tokenize Terms and remove stopwords

def tokenize_term(x):

        predefined_stopwords='horror perfectly'

        english_stopwords=stopwords.words("english")

        german_stopwords=stopwords.words("german")

        stopwords_list=(list(predefined_stopwords.split(' '))+english_stopwords+german_stopwords)         

        keyword_processor_stopwords = KeywordProcessor()

        for each in stopwords_list:

            keyword_processor_stopwords.add_keyword(each,' ')   

        sentence=keyword_processor_stopwords.replace_keywords(x)

        return sentence.strip()
#Combined function

def preprocess(text):

    return regex_filtering(tokenize_term(text))
x=df['text'][0]

print('Before processing: '+x+'\n')

y=preprocess(x)

print('After processing: '+y)
print('Before removing null/empty records from data, size of dataframe is {}'.format(df.shape))

df=df.loc[~df['text'].isnull()]

df=df.loc[df['text']!='']

df=df.loc[~df['author'].isnull()]

df=df.loc[df['author']!='']

print('After removing null/empty records from data, size of dataframe is {}'.format(df.shape))
df['text_preprocessed']=df['text'].apply(preprocess)

df
print('Before removing null/empty preprocessed texts from data, size of dataframe is {}'.format(df.shape))

df=df.loc[~df['text_preprocessed'].isnull()]

df=df.loc[df['text_preprocessed']!='']

print('After removing null/empty preprocessed texts from data, size of dataframe is {}'.format(df.shape))
df.info()
#Lemmatize different forms of words

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize 



wordnet_lemmatizer = WordNetLemmatizer() 

lemmatized = [[wordnet_lemmatizer.lemmatize(word,pos='v') for word in word_tokenize(s)]

              for s in df['text_preprocessed']]

df['text_lemmatized']=[" ".join(i) for i in lemmatized]

df.head()
texts=df['text_lemmatized'].values

labels=df['author'].values

data = list(labels + '\t' + texts)

print('Total count of unique labels : {} \n Authors are : {}'.format(len(set(labels)),df['author'].value_counts()))
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=0)
# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(texts)

X_train_tfidf =  tfidf_vect.transform(X_train)

X_test_tfidf =  tfidf_vect.transform(X_test)



# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(texts)

X_train_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)

X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)



# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(texts)

X_train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_train) 

X_test_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_test) 
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)

    

    if is_neural_net:

        predictions = predictions.argmax(axis=-1)

    

    return metrics.accuracy_score(predictions, y_test)
# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), X_train_tfidf, y_train, X_test_tfidf)

print("Naive Bayes, WordLevel TF-IDF: {}".format(accuracy))



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), X_train_tfidf_ngram, y_train, X_test_tfidf_ngram)

print("Naive Bayes,  N-Gram Vectors: {}".format(accuracy))



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), X_train_tfidf_ngram_chars, y_train, X_test_tfidf_ngram_chars)

print("Naive Bayes,CharLevel Vectors: {}".format(accuracy))
# Linear Classifier on Word Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), X_train_tfidf, y_train, X_test_tfidf)

print ("Logistic Regression, WordLevel TF-IDF: "+ str(accuracy))



# Linear Classifier on Ngram Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), X_train_tfidf_ngram, y_train, X_test_tfidf_ngram)

print("Logistic Regression, N-Gram Vectors: "+ str(accuracy))



# Linear Classifier on Character Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), X_train_tfidf_ngram_chars, y_train, X_test_tfidf_ngram_chars)

print("Logistic Regression, CharLevel Vectors: "+ str(accuracy))
# SVM on word Level TF IDF Vectors

accuracy = train_model(svm.SVC(),X_train_tfidf, y_train, X_test_tfidf)

print("SVM, N-Gram Vectors: {}".format(accuracy))
# RF on Word Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(), X_train_tfidf, y_train, X_test_tfidf)

print("Random Forest, WordLevel TF-IDF: {}".format(accuracy))