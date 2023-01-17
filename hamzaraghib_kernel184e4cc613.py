# Import libararies



import re

import pandas as pd # CSV file I/O (pd.read_csv)

from nltk.corpus import stopwords

import numpy as np

import sklearn

import nltk

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score ,confusion_matrix



#import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import warnings

warnings.filterwarnings('ignore')
# To display entire text

pd.set_option('display.max_colwidth', -1)

#read dataset

news = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)

news = news[:20853]
news['text'] = news[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)

#delete Unnecessary data 

del news['authors']

del news['date'] 

del news['link'] 

del news['headline'] 

del news['short_description'] 
# Split the data into train and test.

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(news[['text']], news['category'], test_size=0.3)
# Convert pandas series into numpy array

X_train = np.array(X_train);

X_test = np.array(X_test);

Y_train = np.array(Y_train);

Y_test = np.array(Y_test);



cleanHeadlines_train = [] #To append processed headlines

cleanHeadlines_test = [] #To append processed headlines

number_reviews_train = len(X_train) #Calculating the number of reviews

number_reviews_test = len(X_test) #Calculating the number of reviews
from nltk.stem import PorterStemmer, WordNetLemmatizer

lemmetizer = WordNetLemmatizer()

stemmer = PorterStemmer()

def get_words(headlines_list):

    headlines = headlines_list[0]   

    headlines_only_letters = re.sub('[^a-zA-Z]', ' ', headlines)

    words = nltk.word_tokenize(headlines_only_letters.lower())

    stops = set(stopwords.words('english'))

    meaningful_words = [lemmetizer.lemmatize(w) for w in words if w not in stops]

    return ' '.join(meaningful_words)# + author_names)
for i in range(0,number_reviews_train):

    cleanHeadline = get_words(X_train[i]) #Processing the data and getting words with no special characters, numbers or html tags

    cleanHeadlines_train.append( cleanHeadline )
for i in range(0,number_reviews_test):

    cleanHeadline = get_words(X_test[i]) #Processing the data and getting words with no special characters, numbers or html tags

    cleanHeadlines_test.append( cleanHeadline )
vectorize = sklearn.feature_extraction.text.TfidfVectorizer(analyzer = "word", max_features=30000)

tfidwords_train = vectorize.fit_transform(cleanHeadlines_train)

X_train = tfidwords_train.toarray()



tfidwords_test = vectorize.transform(cleanHeadlines_test)

X_test = tfidwords_test.toarray()



tfidwords_claim = vectorize.transform(cleanHeadlines_claim)

X_claim = tfidwords_claim.toarray()
from sklearn.svm import LinearSVC



model = LinearSVC()

model.fit(X_train,Y_train)

Y_predict = model.predict(X_test)

accuracy = accuracy_score(Y_test,Y_predict)*100

print(format(accuracy, '.2f'))