import sys
import nltk
import sklearn
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
df = pd.read_csv('../input/homl2020imdb/train.csv')
df.shape
df = df.drop_duplicates()
df.shape
#checking class distrubution
df.sentiment.value_counts() #the data seems balanced
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df.sentiment = encoder.fit_transform(df.sentiment)

df.head()
#regex expressions
def prepro_review(review):
    # remove all html tags
    review = BeautifulSoup(review,'html.parser').get_text()
    
    # Replace email addresses with 'email'
    review = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$',' ',review)
    
    # Replace URLs with 'webaddress'
    review = re.sub(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',' ', review)
    
    # Replace money symbols with 'moneysymb'
    review = re.sub(r'£|\$', ' ', review)
    
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    review = re.sub(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' ', review)
    
    # remove number
    review = re.sub(r'\d+(\.\d+)?', ' ', review)
    
    # Remove punctuation
    review = re.sub(r'[^\w\d\s]', ' ', review)
    
    # Replace whitespace between terms with a single space
    review = re.sub(r'\s+', ' ', review)
    
    # Remove leading and trailing whitespace
    review = re.sub(r'^\s+|\s+?$', '', review)
    
    # change words to lower case
    review = review.lower()
    
    return review
df['processed_review'] = df['review'].apply(prepro_review)
df['processed_review'][0]
from nltk.corpus import stopwords

# remove stop words from text messages
stop_words = set(stopwords.words('english'))

df['processed_review'] = df['processed_review'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
df['processed_review'][0]
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
# Remove word stems using a Porter stemmer
#lemmetizing each word
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
df['processed_review'] =  df['processed_review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(term, get_wordnet_pos(term)) for term in x.split()))
df['processed_review'][0]
# use CountVectorizer scikit-learn object to create bag of words
from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
clean_train_reviews = []
for review in df['processed_review']:
    clean_train_reviews.append(review)
len(clean_train_reviews)
# fit-transform learns the vocabulary and transforms training data into feature vectors
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# transform list of strings to numpy array for more efficiency
train_data_features = train_data_features.toarray()
train_data_features.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_data_features,df["sentiment"],test_size=0.2,stratify=df["sentiment"],random_state=60616)
X_train.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(n_neighbors=2),
    DecisionTreeClassifier(random_state=60616),
    RandomForestClassifier(n_estimators = 100),
    LogisticRegression(random_state=60616),
    SGDClassifier(max_iter = 100, tol=1e-3),
    MultinomialNB()
]

for name, classifier in zip(names, classifiers):
    model = classifier
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    accuracy = accuracy_score(y_test,result)
    f1=f1_score(y_test,result)
    print("{} Accuracy: {}".format(name, accuracy*100))
    print("{} F1 Score: {}\n".format(name, f1*100))
mymodel = LogisticRegression(random_state=60616, max_iter=400)
mymodel.fit(X_train, y_train)
result = mymodel.predict(X_test)

accuracy = accuracy_score(y_test,result)
f1=f1_score(y_test,result)

print("Accuracy: {}".format(accuracy*100))
print("F1 Score: {}\n".format(f1*100))
test_df = pd.read_csv('../input/homl2020imdb/test.csv')
#preprocessing the test data

test_clean_review = []
for review in test_df['review']:
    review = prepro_review(review)
    
    split_review1 = review.split()
    meaningful_words = [w for w in split_review1 if w not in stop_words]
    sentencewithoutstopword = " ".join(meaningful_words)
    
    split_review2 = sentencewithoutstopword.split()
    lemmetize_words = [lemmatizer.lemmatize(term, get_wordnet_pos(term)) for term in split_review2]
    lemmetizedsentence = " ".join(lemmetize_words)

    test_clean_review.append(lemmetizedsentence)
# get a bag of words of test data
test_data_features = vectorizer.transform(test_clean_review)

# transform to numpy array for more efficiency
test_data_features = test_data_features.toarray()
testresult = mymodel.predict(test_data_features)
output = pd.DataFrame(data={"id":test_df["id"], "sentiment":testresult})
output.to_csv("test_result.csv", index=False)