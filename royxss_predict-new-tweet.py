# Import File and Packages

import sklearn

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

%matplotlib inline

import re

import warnings

warnings.filterwarnings("ignore")



# Read csv file into data frame

tweet=pd.read_csv("../input/Tweets.csv")
# Preprocess the data {'negative': 0 , 'positive': 1 , 'neutral': 2}



df=tweet.iloc[:,(10,1)]

df.columns = ['data', 'target']

df['target']=df['target'].str.strip().str.lower()

df['target']=df['target'].map({'negative': 0 , 'positive': 1 , 'neutral': 2})



# Copy df to a temporary dataframe for pre-processing

# Below assignment is causing problems

dft=df
%%time

# Remove @tweets, numbers, hyperlinks that do not start with letters

dft['data']=dft['data'].str.replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])"," ")



# tokenize into words

import nltk

dft['data']=dft['data'].apply(nltk.word_tokenize)
%%time

# stem the tokens

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

dft['data']=dft['data'].apply(lambda x: [stemmer.stem(y) for y in x])
%%time

# Lemmatizing

lemmatizer = nltk.WordNetLemmatizer()

dft['data']=dft['data'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
%%time

# Remove stopwords

stopwords = nltk.corpus.stopwords.words('english')



# stem the stopwords

stemmed_stops = [stemmer.stem(t) for t in stopwords]



# remove stopwords from stemmed/lemmatized tokens

dft['data']=dft['data'].apply(lambda x: [stemmer.stem(y) for y in x if y not in stemmed_stops])



# remove words whose length is <3

dft['data']=dft['data'].apply(lambda x: [e for e in x if len(e) >= 3])
%%time

# Detokenize cleaned dataframe for vectorizing

dft['data']=dft['data'].str.join(" ")
#Print attributes of tweet, X and y

print('Shape of original file : ', tweet.shape)

print('All columns of the original file : ', tweet.columns.tolist() , '\n')

print('Columns dft dataframe : ',dft.columns.tolist(), '\n') 

print('Shape data and target : ', dft['data'].shape, dft['target'].shape, '\n')

print('Mood Count target :\n',tweet['airline_sentiment'].value_counts())
%%time

from sklearn.cross_validation import train_test_split

from sklearn.dummy import DummyClassifier

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer



X=dft['data']

y=dft['target']



arr_Accu=[]



#Using train_test_split

#Selecting the best random state and comapring the accuracy using Dummy Classifier

for i in range(1,20):

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=i)



    vect = CountVectorizer(stop_words='english',analyzer="word",min_df = 2, max_df = 0.8)

    X_train_dtm = vect.fit_transform(X_train)

    X_test_dtm = vect.transform(X_test)

    feat_dtm = vect.get_feature_names()

    #feat_dtm



    clf = DummyClassifier()

    clf.fit(X_train_dtm, y_train)

    y_pred = clf.predict(X_test_dtm)



    accuracy = metrics.accuracy_score(y_test, y_pred)

    #print(accuracy)

    arr_Accu.append(accuracy)



for j in range(1,20):

    print("Random State : ", j, "   Accuracy : ", arr_Accu[j-1])
%%time

#Using K-fold validation

#Selecting the best fold and comparing the accuracy using Naive Bayes

from sklearn.cross_validation import cross_val_score

arr_Accu=[]



#Selecting the best random state and comparing the accuracy using dummy classifier

for i in range(3,15):



    vect = CountVectorizer(stop_words='english',analyzer="word",min_df = 2, max_df = 0.8)

    X_dtm = vect.fit_transform(X)



    clf = DummyClassifier()

    accuracy = cross_val_score(clf, X_dtm, y, cv=i, scoring='accuracy')

    

    arr_Accu.append(np.mean(accuracy))



#print(arr_Accu)

for j in range(3,15):

    print("K-Fold : ", j, "   Accuracy : ", arr_Accu[j-3])
def print_top_words():    

    # Print top words

    vect = CountVectorizer(stop_words='english',analyzer="word",min_df = 2, max_df = 0.8)

    data_dtm = vect.fit_transform(dft['data'])

    feat_dtm = vect.get_feature_names()



    # Count words

    freq_tbl=pd.DataFrame({'Word':feat_dtm,'Occurence':np.asarray(data_dtm.sum(axis=0)).ravel().tolist()})

    freq_tbl['Word']=freq_tbl['Word'].str.strip()



    # Print top words

    topt = freq_tbl.sort(['Occurence'], ascending=[False]).head(10)

    y = topt['Occurence']

    plt.grid()

    X = range(1, 11)

    plt.bar(X,y,color='g')

    plt.xlabel('Top words')

    plt.ylabel('Occurence')

    plt.title('Frequency of top 10 words')

    plt.xticks(X,topt['Word'],rotation=90)

    

def print_top_neg_words():    

    # Print top negative words

    vect = CountVectorizer(stop_words='english',analyzer="word",min_df = 2, max_df = 0.8)

    filt = dft[dft['target'] == 0]

    data_dtm = vect.fit_transform(filt['data'])

    feat_dtm = vect.get_feature_names()



    # Count words

    freq_tbl=pd.DataFrame({'Word':feat_dtm,'Occurence':np.asarray(data_dtm.sum(axis=0)).ravel().tolist()})

    freq_tbl['Word']=freq_tbl['Word'].str.strip()



    # Print top negative words

    topt = freq_tbl.sort(['Occurence'], ascending=[False]).head(10)

    y = topt['Occurence']

    plt.grid()

    X = range(1, 11)

    plt.bar(X,y,color='g')

    plt.xlabel('Top negative words')

    plt.ylabel('Occurence')

    plt.title('Frequency of top 10 negative words')

    plt.xticks(X,topt['Word'],rotation=90)

    

def print_top_pos_words():    

    # Print top positive words

    vect = CountVectorizer(stop_words='english',analyzer="word",min_df = 2, max_df = 0.8)

    filt = dft[dft['target'] == 1]

    data_dtm = vect.fit_transform(filt['data'])

    feat_dtm = vect.get_feature_names()



    # Count words

    freq_tbl=pd.DataFrame({'Word':feat_dtm,'Occurence':np.asarray(data_dtm.sum(axis=0)).ravel().tolist()})

    freq_tbl['Word']=freq_tbl['Word'].str.strip()



    # Print top positive words

    topt = freq_tbl.sort(['Occurence'], ascending=[False]).head(10)

    y = topt['Occurence']

    plt.grid()

    X = range(1, 11)

    plt.bar(X,y,color='g')

    plt.xlabel('Top positive words')

    plt.ylabel('Occurence')

    plt.title('Frequency of top 10 positive words')

    plt.xticks(X,topt['Word'],rotation=90)





plt.figure(1,figsize=(16, 16))

plt.subplot(251)

print_top_words()  

plt.subplot(253)

print_top_pos_words()

plt.subplot(255)

print_top_neg_words()
%%time

import time

#Train Test split data with random state=11

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=11)



#Vectorize

vect = CountVectorizer(stop_words='english',analyzer="word",min_df = 2, max_df = 0.8)

X_train_dtm = vect.fit_transform(X_train)

X_test_dtm = vect.transform(X_test)

feat_dtm = vect.get_feature_names()



#Initialize classifier stats

clf_stats=pd.DataFrame()
# Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()



start_time=time.time()

clf.fit(X_train_dtm, y_train)

runtime = time.time()-start_time

y_pred = clf.predict(X_test_dtm)

accuracy = metrics.accuracy_score(y_test, y_pred)



print('Accuracy : ',accuracy)



#Store stats for classifier

clf_stats=clf_stats.append({'Classifier': 'Multinomial Naive Bayes', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = MultinomialNB()'}, ignore_index=True)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()



start_time=time.time()

clf.fit(X_train_dtm, y_train)

runtime = time.time()-start_time

y_pred = clf.predict(X_test_dtm)

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy : ',accuracy)



#Store stats for classifier

clf_stats=clf_stats.append({'Classifier': 'Logistic Regression', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = LogisticRegression()'}, ignore_index=True)
# K Nearest Neighbour

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=2)



start_time=time.time()

clf.fit(X_train_dtm, y_train)

runtime = time.time()-start_time

y_pred = clf.predict(X_test_dtm)

accuracy = metrics.accuracy_score(y_test, y_pred)



print('Accuracy : ',accuracy)



#Store stats for classifier

clf_stats=clf_stats.append({'Classifier': 'KNN', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = KNeighborsClassifier(n_neighbors=2)'}, ignore_index=True)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy')



start_time=time.time()

clf.fit(X_train_dtm, y_train)

runtime = time.time()-start_time

y_pred = clf.predict(X_test_dtm)

accuracy = metrics.accuracy_score(y_test, y_pred)



print('Accuracy : ',accuracy)



#Store stats for classifier

clf_stats=clf_stats.append({'Classifier': 'Decision Trees', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = DecisionTreeClassifier(criterion=''entropy'')'}, ignore_index=True)
# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(criterion='entropy')



start_time=time.time()

clf.fit(X_train_dtm, y_train)

runtime = time.time()-start_time

y_pred = clf.predict(X_test_dtm)

accuracy = metrics.accuracy_score(y_test, y_pred)



print('Accuracy : ',accuracy)



#Store stats for classifier

clf_stats=clf_stats.append({'Classifier': 'Random Forest', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = RandomForestClassifier(criterion=''entropy'')'}, ignore_index=True)
# AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()



start_time=time.time()

clf.fit(X_train_dtm, y_train)

runtime = time.time()-start_time

y_pred = clf.predict(X_test_dtm)

accuracy = metrics.accuracy_score(y_test, y_pred)



print('Accuracy : ',accuracy)



#Store stats for classifier

clf_stats=clf_stats.append({'Classifier': 'ADA Boost', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = AdaBoostClassifier()'}, ignore_index=True)
# SVM SVC

from sklearn.svm import LinearSVC

clf = LinearSVC()



start_time=time.time()

clf.fit(X_train_dtm, y_train)

runtime = time.time()-start_time

y_pred = clf.predict(X_test_dtm)

accuracy = metrics.accuracy_score(y_test, y_pred)



print('Accuracy : ',accuracy)



#Store stats for classifier

clf_stats=clf_stats.append({'Classifier': 'SVM SVC', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = LinearSVC()'}, ignore_index=True)
# GaussianNB

# Not good with words

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()



start_time=time.time()

clf.fit(X_train_dtm.toarray(), y_train)

runtime = time.time()-start_time

y_pred = clf.predict(X_test_dtm.toarray())

accuracy = metrics.accuracy_score(y_test, y_pred)



print('Accuracy : ',accuracy)



#Store stats for classifier

clf_stats=clf_stats.append({'Classifier': 'Gaussian NB', 'Accuracy': accuracy, 'Runtime': runtime, 'Callable': 'clf = GaussianNB()'}, ignore_index=True)
import matplotlib.pyplot as plt

%matplotlib inline



print (clf_stats[['Classifier','Accuracy','Runtime']].sort(['Accuracy'], ascending=[False]))

#Plot performance measures of classifiers



x=range(1,(len(clf_stats.Classifier)+1))

plt.figure(1,figsize=(14, 5))



plt.subplot(131)

plt.xlabel('Classifier')

plt.ylabel('Classifier Accuracy')

plt.plot(x, clf_stats['Accuracy'],color='g')

plt.xticks(x,clf_stats.Classifier,rotation=90)



plt.subplot(133)

plt.xlabel('Classifier')

plt.ylabel('Classifier Runtime')

plt.plot(x, clf_stats['Runtime'],color='g')

plt.xticks(x,clf_stats.Classifier,rotation=90)    
# Clean input tweet



def fmt_input_tweet(txt):

    

    # Remove @tweets, numbers, hyperlinks that do not start with letters

    txt = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])"," ",txt)

    #print(txt)

    

    # tokenize into words

    tokens = [word for word in nltk.word_tokenize(txt)]

    #print(tokens)



    # only keep tokens that start with a letter (using regular expressions)

    clean_tokens = [token for token in tokens if re.search(r'^[a-zA-Z]+', token)]

    #print('clean_tokens:\n',clean_tokens)



    # stem the tokens

    stemmer = SnowballStemmer('english')

    stemmed_tokens = [stemmer.stem(t) for t in clean_tokens]

    #print('stemmed_tokens:\n',stemmed_tokens)



    #Lemmatizing

    lemmatizer = nltk.WordNetLemmatizer()

    lem_tokens = [lemmatizer.lemmatize(t) for t in stemmed_tokens]

    #print('lemmatizer : \n',lem_tokens)

    

    #Remove stopwords

    stopwords = nltk.corpus.stopwords.words('english')



    # stem the stopwords

    stemmed_stops = [stemmer.stem(t) for t in stopwords]



    # remove stopwords from stemmed/lemmatized tokens

    lem_tokens_no_stop = [stemmer.stem(t) for t in lem_tokens if t not in stemmed_stops]



    # remove words whose length is <3

    clean_lem_tok = [e for e in lem_tokens_no_stop if len(e) >= 3]

    #print('clean_lem_tok: ',clean_lem_tok)

    

    # Detokenize new tweet for vector processing

    new_formatted_tweet=" ".join(clean_lem_tok)

    #print('new_formatted_tweet: ',new_formatted_tweet)

    

    return new_formatted_tweet

    
# Logistic Regression performs better. So it will automatically used as an appropriate classifier



# Vectorize, fit, transform. Select model randomly

vect = CountVectorizer(stop_words='english', analyzer="word", min_df = 2, max_df = 0.8)

X_dtm = vect.fit_transform(X)

feat_dtm = vect.get_feature_names()



# Select the best performing classifier

Call_clf = str(clf_stats[['Callable','Accuracy']].sort(['Accuracy'], ascending=[False]).head(1).iloc[:,(0)])

temp = Call_clf.__repr__()

Call_clf = temp[temp.index('c'):(temp.index(')'))+1]

print('Model :',temp[(temp.index('=') + 1) : temp.index('(')])

exec(Call_clf)

clf.fit(X_dtm.toarray(), y) 



def classify_new_tweet(new_twt):  



    fmt_twt = fmt_input_tweet(new_twt)

    fmt_twt_dtm = vect.transform([fmt_twt])[0]

    #print('Formatted Tweet :',fmt_twt)

    pred = clf.predict(fmt_twt_dtm.toarray())



    def mood(x):

        return {

            0: 'negative',

            1: 'positive',

            2: 'neutral'

        }[x]



    print('Mood of the incoming tweet is:',mood(pred[0]))
twt='@united I am sick!! https://www.abc.com'

classify_new_tweet(twt)
# Predict performance of random prediction. Testing for positive classes.

y_random_pred = np.ones(y.shape[0])

accuracy = metrics.accuracy_score(y, y_random_pred)

print('Accuracy of random (positive) prediction : ',accuracy)
# Predict performance of majority class prediction. Testing for negative classes.

y_random_pred = np.zeros(y.shape[0])

accuracy = metrics.accuracy_score(y, y_random_pred)

print('Accuracy of majority class (negative) prediction : ',accuracy)