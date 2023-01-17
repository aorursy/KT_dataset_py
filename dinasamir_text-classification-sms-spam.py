import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import nltk #NLTK provides  common NLP function

from nltk.corpus import stopwords

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import string

import re

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import fbeta_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
data = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding = 'latin-1')

data.head()
data=data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis = 1)

data = data.rename(columns={"v1":"label", "v2":"text"})
data.groupby("label").describe()
data.label.value_counts()
data["label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)

plt.ylabel("Spam vs Ham")

plt.legend(["Ham", "Spam"])

plt.show()
data.label.value_counts().plot.bar()
string.punctuation

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")



#def cleanText(message):

  #  message = message.translate(str.maketrans('', '', string.punctuation))

  #  print(message.split())

   # words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]

   # return " ".join(words)



#data["text"] = data["text"].apply(cleanText)
data.head()
def rx(text):

    # Applying Regular Expression

    #Replace email addresses with 'emailaddr'

    #Replace URLs with 'httpaddr'

    #Replace money symbols with 'moneysymb'

    #Replace phone numbers with 'phonenumbr'

    #Replace numbers with 'numbr'

    print(char for char in txt_no_stop_words)

    msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)

    msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)

    msg = re.sub('£|\$', 'moneysymb', text)

    msg = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', text)

    return msg
# create a dataframe from a word matrix

def wm2df(wm, feat_names):

    # create an index for each row

    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]

    df = pd.DataFrame(data=wm.toarray(), index=doc_names,columns=feat_names)

    return(df)
def cleanText(text):

   #remove punctuation

    txt_no_punctuation=[char for char in text if char not in string.punctuation]

    txt_no_punctuation = "".join(txt_no_punctuation).split()

    #remove stop_words

    txt_no_stop_words=[char.lower() for char in txt_no_punctuation if char.lower() not in stopwords.words("english")]

    # using stemming

    ps=nltk.PorterStemmer()

    clean_text=[ps.stem(word) for word in txt_no_stop_words]

    # using lemmatization

    ws=nltk.WordNetLemmatizer()

    clean_text=[ws.lemmatize(word) for word in txt_no_stop_words]

    return clean_text



data["text_clean"]=data["text"].apply(lambda x:cleanText(x))
#Bag of Words (CountVectorizer)

count_vect=CountVectorizer(analyzer=cleanText)

# convert the documents into a document-term matrix

wm=count_vect.fit_transform(data['text'])

# retrieve the terms found in the corpora

tokens = count_vect.get_feature_names()

# create a dataframe from the matrix

wm2df(wm, tokens)
print(X_counts.shape)

print(count_vect.get_feature_names())
#Bag of Words (CountVectorizer with n_gram)

#ngram_vect=CountVectorizer(ngram_range=(2,2),analyzer=cleanText)

#X_counts=ngram_vect.fit_transform(data['text'])

#print(X_counts.shape)

#print(ngram_vect.get_feature_names())



#Bag of Words (tf idf)

tfidf_vect=TfidfVectorizer(analyzer=cleanText)

X_counts=tfidf_vect.fit_transform(data['text'])

tfidf_vect._validate_vocabulary()

print(X_counts.shape)

print(tfidf_vect.get_feature_names())
data["text_len"]=data['text'].apply(len)

data.head()
bins=np.linspace(0,200,40)

plt.hist(data[data['label']=='spam']['text_len'],bins,alpha=0.5,normed=True,label='spam')

plt.hist(data[data['label']=='ham']['text_len'],bins,alpha=0.5,normed=True,label='ham')

plt.legend(loc='upper left')

plt.show()
X = X_counts.toarray()

y = data['label']

le = LabelEncoder()

y = le.fit_transform(y)

y = y.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
bayes_classifier = GaussianNB()

bayes_classifier.fit(X_train, y_train)

#Predicting

y_pred = bayes_classifier.predict(X_test)

# Evaluating

cm = confusion_matrix(y_test, y_pred)

cm
print ("Accuracy : %0.5f \n\n" % accuracy_score(ytest, bayes_classifier.predict(xtest)))

print (classification_report(ytest, bayes_classifier.predict(xtest)))
gaussianNb = MultinomialNB()

gaussianNb.fit(X_train, y_train)

y_pred = gaussianNb.predict(X_test)

print ("Accuracy : %0.5f \n\n" % accuracy_score(y_test, gaussianNb.predict(X_test)))

print(fbeta_score(y_test, y_pred, beta = 0.5))