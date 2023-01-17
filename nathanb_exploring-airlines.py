%matplotlib inline
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


df = pd.read_csv("../input/Tweets.csv")

print (df.shape)
#print (df.head())
print (df.info())

plt.figure(1)
pd.Series(df["airline_sentiment"]).value_counts().plot(kind = "barh" , title = "airline_sentiment")
plt.figure(2)
pd.Series(df["airline"]).value_counts().plot(kind = "barh" , title = "airline")
plt.figure(3)
pd.Series(df["tweet_location"]).value_counts().head(20).plot(kind = "barh" , title = "tweet_location")
plt.figure(4)
pd.Series(df["retweet_count"]).value_counts().plot(kind = "barh" , title = "retweet")
plt.figure(5)
pd.Series(df["name"]).value_counts().head(20).plot(kind = "barh" , title = "name")
plt.figure(6)
pd.Series(df["user_timezone"]).value_counts().head(20).plot(kind = "barh" , title = "user_timezone")
plt.figure(9)
pd.Series(df["negativereason"]).value_counts().plot(kind = "barh" , title = "negativereason")
#time conversion
df["tweet_created"] = df["tweet_created"].apply(lambda x: pd.to_datetime(x))
df["hour"] =  df["tweet_created"].apply(lambda x: x.hour)
df["dayofweek"] =  df["tweet_created"].apply(lambda x: x.dayofweek) #monday = 0 Sunday = 6
plt.figure(7)
tmp = pd.Series(df["hour"]).value_counts().sort_index().plot(kind = "barh" , title = "hour")
plt.figure(8)
pd.Series(df["dayofweek"]).value_counts().sort_index().plot(kind = "barh" , title = "dayofweek")


#exploring correlation between features
pd.crosstab(index = df["airline"] ,  columns = df["airline_sentiment"] ).plot(kind = "barh")

#the tweets themselves
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer("english")
import re
from bs4 import BeautifulSoup  

          
def cleanword(w):    
    return re.sub('[^a-zA-Z,]' , ' ' , w)  

def cleantext(review):
    review = BeautifulSoup(review ,"lxml").get_text()
    review_words = cleanword(review.lower()).split()    
    stop = stopwords.words('english')
    stemmed_words = [stemmer.stem(w) for w in review_words if w not in stop]
    return " ".join(stemmed_words)


print (cleantext("GSW2015!Winners rasta! so tomatoes"))
df["text"]  = df["text"].apply(cleantext)


################# ML ##############################################3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split 
 
X_clean = df["text"] 
#X_train, X_test, y_train, y_test = train_test_split(X_clean, Y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_clean, df["airline_sentiment"], test_size=0.33, random_state=42)

vectorizer = TfidfVectorizer(max_df=0.5, max_features=2000, min_df=2, stop_words='english')
vectorizer.fit(X_clean)                                 
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)



# select K best features
from sklearn.feature_selection import SelectKBest
word_list = vectorizer.get_feature_names()
term_doc_mat = vectorizer.fit_transform(X_clean)  
selector = SelectKBest(k=10).fit(term_doc_mat, df["airline_sentiment"])
informative_words_index = selector.get_support(indices=True)
labels = [word_list[i] for i in informative_words_index]
data = pd.DataFrame(term_doc_mat[:,informative_words_index].todense(), columns=labels)
data['airline_sentiment'] = df["airline_sentiment"]
print (data.corr())
#sns.heatmap(data.corr())


#using the metrics package
from sklearn.metrics import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
def printScores(y_test, y_pred, classif_name):    
    print ( "--------------  "  + classif_name + "  ------------------" ) 
    print ("recall : %0.2f" %  recall_score(y_test, y_pred) )
    print ("precision : %0.2f" %  precision_score(y_test, y_pred) )   
    print ("f1 : %0.2f" %  f1_score(y_test, y_pred)  )
    print ("accuracy : %0.2f" %  accuracy_score(y_test, y_pred)  )
    print ("---------------------------------------------------" ) 
    
#multinomial NB
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
printScores(y_test, y_pred, "MultinomialNB")

#logreg 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
printScores(y_test, y_pred, "LogisticRegression")

#random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
printScores(y_test, y_pred, "RandomForestClassifier")

#knn 
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
printScores(y_test, y_pred, "KNeighborsClassifier")

