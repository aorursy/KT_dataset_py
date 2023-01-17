# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk #for natural language processing
import string
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer

from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import emoji
from sklearn.base import BaseEstimator, TransformerMixin
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import collections
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")

#To check first five rows
train_df.head()
test_df.head()
#To check type of columns
train_df.info()
#To check size of data
print(train_df.shape)
print("training set has 3339 rows and 12 columns\n ")
#To check no. of null values
print(train_df.isnull().sum())
print("\nFollowing columns have null values more than 1:-\r\n1. negativereason\r\n2. negativereason_confidence\r\n3. tweet_created\r\n4. tweet_location\r\n5. usertimezone ")
#To count the no. of negative , positive and neutral tweets in training data
mood_count=train_df["airline_sentiment"].value_counts()
mood_count
# To plot the abouve stats
plt.bar(["Negative","Neutral",'Positive'],mood_count)
plt.xlabel("Mood")
plt.ylabel("Mood_Count")
plt.xticks(rotation=45)
plt.title("Count of Moods")
# To find the count of tweets for different airlines
train_df["airline"].value_counts()

# To plot the sentiment count airline wise
def plot_airline_wise_sentiments(Airline):
    df=train_df[train_df["airline"]==Airline]
    count=df["airline_sentiment"].value_counts()
    plt.bar(["Negative","Neutral","Positive"],count)
    plt.xlabel("Moods")
    plt.ylabel("Moood_Counts")
    plt.title("Mood counts for {}".format(Airline))
plt.figure(1,figsize=(20,12))
plt.subplot(231)
plot_airline_wise_sentiments("United")
plt.subplot(232)
plot_airline_wise_sentiments("Virgin America")


# To get the count of negative reasons . Here dict function is used to create dictionary.
NR_count=dict(train_df["negativereason"].value_counts())
print(NR_count)

# To get airline wise count of negative reasons
def NR_count(Airline):
    if(Airline=="All"):
        df=train_df
    else:
        df=train_df[train_df["airline"]==Airline]
    count=dict(df["negativereason"].value_counts())
    unique_reason=list(train_df["negativereason"].unique())
    unique_reason=[x for x in unique_reason if str(x)!='nan'] # To remove none values
    print(type(unique_reason))
    reason_frame=pd.DataFrame({'Reasons':unique_reason})
    reason_frame['Count']=reason_frame['Reasons'].apply(lambda x:count[x])
    return reason_frame
    
# To plot airline wise count of negative reason
def plot_reason(Airline):
    df=NR_count(Airline)
    index=df["Reasons"]
    plt.figure(figsize=(12,12))
    plt.bar(index,df["Count"])
    plt.xticks(rotation=45)
    plt.tick_params(top='off', bottom='on', left='off', right='off', labelleft='on', labelbottom='on')
    plt.xlabel("Negative Reasons")
    plt.ylabel("Count")
    plt.title("Negative reason count for "+Airline)

plot_reason("All")
plot_reason("United")
plot_reason("Virgin America")
# Functions to remove unnecesary words, symbols from text .
def remove_mentions(input_text): # To remove @....
    return re.sub(r'@\w+','',str(input_text))
def remove_urls(input_text): # To remove http.......
    return re.sub(r'http.?://[^\s]+[\s]?', '', str(input_text))
def emoji_oneword(input_text): # To remove emojis
    return input_text.replace('_','')
def remove_punctuation(input_text): # To remove punctuations(, . ! ') 
    punct=string.punctuation # punct now has all the punctuations used in english 
    trantab=str.maketrans(punct,len(punct)*' ') #every punctuation in punct will be mapped to ' ' and stored in trantab in a table
    return  input_text.translate(trantab) # Here punctuations in text will be replaced by ' ' as defined in trantab.
def remove_digits(input_text): # To remove digits 
     return re.sub(r'\d+','',str(input_text))
def to_lower(input_text): # To convert each word in lower case
     return input_text.lower()
def remove_stopWords(input_text): # To remove stop words like the, is , in ,not.
    stopwords_list = stopwords.words('english')
    whitelist=["n't","no","not"] # Some words which might indicate a certain sentiment are kept via a whitelist
    words=input_text.split() # By default it will split the words by ' '
    clean_words=[word for word in words if(word not in stopwords_list or word in whitelist) and len(word)>1]
    return " ".join(clean_words)
def stemming(input_text): # stemming means getting word to its original form eg: Difficulty -> Difficult
    porter=PorterStemmer()
    words=input_text.split()
    stemmed_words = [porter.stem(word) for word in words]
    return " ".join(stemmed_words)
 
pd.options.mode.chained_assignment = None  # default='warn' # To hide warnings
df=train_df[train_df["airline_sentiment"]=="negative"]
df["text"]=df["text"].fillna("No Value Found")
df["text"]=df["text"].apply(lambda x: emoji.demojize(x)) #To covert emoji in txt
df["text"]=df["text"].apply(lambda x: remove_mentions(x))
df["text"]=df["text"].apply(lambda x: remove_urls(x))
df["text"]=df["text"].apply(lambda x: emoji_oneword(x))
df["text"]=df["text"].apply(lambda x: remove_punctuation(x))
df["text"]=df["text"].apply(lambda x: remove_digits(x))
df["text"]=df["text"].apply(lambda x: to_lower(x))
df["text"]=df["text"].apply(lambda x: remove_stopWords(x))
# To create Word Cloud of most frequent negative words
words=' '.join( x for x in str(df.text.values).split())             
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
d = os.path.dirname("../input/")
plane_coloring = np.array(Image.open(os.path.join(d, "Airplane_Transparent_PNG_Clipart.png")))

wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=3000,
                      height=2500,
                      colormap="Blues",
                      max_words=15,
                      mask=plane_coloring
                     ).generate(words)
image_colors = ImageColorGenerator(plane_coloring)

plt.figure(1,figsize=(25,25))
plt.imshow(wordcloud,interpolation="bilinear")
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
#plt.imshow(plane_coloring, cmap=plt.cm.gray, interpolation="bilinear")
plt.axis('off')
plt.show()
# Function to get meaningful words from training file
def tweet_to_words(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",str(raw_tweet)) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))
# Function to get length of meaningful words from training file
def clean_tweet_length(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",str(raw_tweet))
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(len(meaningful_words))
# Changing the sentiment column values in numerical categorical form as we will only predict whether the sentiment is positive or negative, considering neutral sentiments as positive
train_df['sentiment']=train_df['airline_sentiment'].apply(lambda x: -1 if x=='negative' else(0 if x=='neutral' else 1))
#test_df['sentiment']=test_df['airline_sentiment'].apply(lambda x: -1 if x=='negative' else(0 if x=='neutral' else 1))

#train_df['clean_tweet']=train_df['text'].apply(lambda x: tweet_to_words(x))
train_df["text"]=train_df["text"].fillna("No Value Found")
train_df["text"]=train_df["text"].apply(lambda x: emoji.demojize(x)) #To covert emoji in txt
train_df["text"]=train_df["text"].apply(lambda x: remove_mentions(x))
train_df["text"]=train_df["text"].apply(lambda x: remove_urls(x))
train_df["text"]=train_df["text"].apply(lambda x: emoji_oneword(x))
train_df["text"]=train_df["text"].apply(lambda x: remove_punctuation(x))
train_df["text"]=train_df["text"].apply(lambda x: remove_digits(x))
train_df["text"]=train_df["text"].apply(lambda x: to_lower(x))
train_df["text"]=train_df["text"].apply(lambda x: remove_stopWords(x))
train_df['clean_tweet']=train_df['text']
train_df['Tweet_length']=train_df['text'].apply(lambda x: clean_tweet_length(x))
train,test = train_test_split(train_df,test_size=0.2,random_state=42)
# for original test data to be used later
#test_df['clean_tweet']=test_df['text'].apply(lambda x: tweet_to_words(x))
test_df["text"]=test_df["text"].fillna("No Value Found")
test_df["text"]=test_df["text"].apply(lambda x: emoji.demojize(x)) #To covert emoji in txt
test_df["text"]=test_df["text"].apply(lambda x: remove_mentions(x))
test_df["text"]=test_df["text"].apply(lambda x: remove_urls(x))
test_df["text"]=test_df["text"].apply(lambda x: emoji_oneword(x))
test_df["text"]=test_df["text"].apply(lambda x: remove_punctuation(x))
test_df["text"]=test_df["text"].apply(lambda x: remove_digits(x))
test_df["text"]=test_df["text"].apply(lambda x: to_lower(x))
test_df["text"]=test_df["text"].apply(lambda x: remove_stopWords(x))
test_df['clean_tweet']=test_df['text']
test_df['Tweet_length']=test_df['text'].apply(lambda x: clean_tweet_length(x))
# Creating train and test clean tweet words data
train_clean_tweet=[]
for tweet in train['clean_tweet']:
    train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in test['clean_tweet']:
    test_clean_tweet.append(tweet)
    
# for original train and test data to be used later
train_original_clean_tweet=[]
for tweet in train_df['clean_tweet']:
    train_original_clean_tweet.append(tweet)
test_original_clean_tweet=[]
for tweet in test_df['clean_tweet']:
    test_original_clean_tweet.append(tweet)

from sklearn.feature_extraction.text import CountVectorizer
#v = CountVectorizer(analyzer = "word")
v = TfidfVectorizer(analyzer="word")
train_features= v.fit_transform(train_clean_tweet)
#print(train_features)
word_freq = dict(zip(v.get_feature_names(), np.asarray(train_features.sum(axis=0)).ravel())) #zip function is used for mapping values in different lists
#print(train_features.sum(axis=0))
#print(word_freq)
word_counter = collections.Counter(word_freq)
#print(word_counter)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.xticks(rotation=45)
plt.show();
test_features=v.transform(test_clean_tweet)

# for original train and test data to be used later
train_original_features= v.fit_transform(train_original_clean_tweet)
test_original_features=v.transform(test_original_clean_tweet)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True,gamma='auto'),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]
dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['sentiment'])
        pred = fit.predict(dense_test)
    
    accuracy = accuracy_score(pred,test['sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
Index = [1,2,3,4,5,6,7]
plt.bar(Index,Accuracy)
plt.xticks(Index, Model,rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracies of Models')
dense_original_features=train_original_features.toarray()
dense_original_test= test_original_features.toarray()
index=Accuracy.index(max(Accuracy))
classifier=Classifiers[index]
try:
    fit = classifier.fit(train_original_features,train_df['sentiment'])
    pred = fit.predict(test_original_features)
except Exception:
    fit = classifier.fit(dense_original_features,train_df['sentiment'])
    pred = fit.predict(dense_original_test)
pred=pred.astype(object)
pred[pred==0]='neutral'
pred[pred==-1]='negative'
pred[pred==1]='positive'
d={"tweet_id":test_df.tweet_id,"airline_sentiment":pred}
submission=pd.DataFrame(d)
submission.to_csv('submission.csv', index=False)
