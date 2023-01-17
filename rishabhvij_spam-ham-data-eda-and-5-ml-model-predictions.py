import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
messages = pd.read_csv("../input/spam.csv",encoding='latin-1')
messages.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1,inplace=True)
messages.rename(columns={"v1":"label", "v2":"message"},inplace=True)
messages.head()
messages.describe()
messages.groupby('label').describe()
messages['length'] = messages['message'].apply(len)
messages.head()
messages['length'].plot(bins=10, kind='hist') 
messages.length.describe()
messages[messages['length'] == 910]['message'].iloc[0]
messages.hist(column='length', by='label', bins=50,figsize=(12,4))
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
from sklearn.pipeline import Pipeline

pipeline1 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline1.fit(msg_train,label_train)
predictions1 = pipeline1.predict(msg_test)
print(classification_report(predictions1,label_test))
nbscore=accuracy_score(predictions1,label_test)
print(nbscore)
pipeline2 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ Logistic regression
])
pipeline2.fit(msg_train,label_train)
predictions2=pipeline2.predict(msg_test)
print(classification_report(predictions2,label_test))
lrscore=accuracy_score(predictions2,label_test)
print(lrscore)
pipeline3 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', KNeighborsClassifier()),  # train on TF-IDF vectors w/ K-nn
])
pipeline3.fit(msg_train,label_train)
predictions3=pipeline3.predict(msg_test)
print(classification_report(predictions3,label_test))
knnscore=accuracy_score(predictions3,label_test)
print(knnscore)
pipeline4 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ SVM
])
pipeline4.fit(msg_train,label_train)
predictions4=pipeline4.predict(msg_test)
print(classification_report(predictions4,label_test))
svmscore=accuracy_score(predictions4,label_test)
print(svmscore)
pipeline5 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', DecisionTreeClassifier()),  # train on TF-IDF vectors w/ DecisionTreeClassifier
])
pipeline5.fit(msg_train,label_train)
predictions5=pipeline5.predict(msg_test)
print(classification_report(predictions5,label_test))
dtscore=accuracy_score(predictions5,label_test)
print(dtscore)
results=[nbscore,lrscore,knnscore,svmscore,dtscore]
n=['Naive-B','Log. Reg.','KNN','SVM','Dtree']

ndf=pd.DataFrame(n)
rdf=pd.DataFrame(results)
rdf[1]=n
print('Accuracy')
rdf