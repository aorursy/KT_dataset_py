# First my program will install all the required libraries and will read the csv file and will convert it to dataframe df.
# It will give column names to the datafram df and then it will drop some columns except text and sentiment.
# It will count the tweets group by sentiment.
# It will add new column pre_clean_len to dataframe which is length of each tweet.
# plot pre_clean_len column.
# check for any tweets greater than 140 characters.
# for each text i am calling tweet_cleaner function which will remove convert words to lower case, remove URL, remove hashtag, remove @mentions, HTML decoding, UTF-8 BOM decoding and converting words like isn't to is not.
# And all this it will store in list called clean_tweet_texts.
# Again it will tokenize the tweets in clean_tweet_texts and will do lemmatizing for every word in  list and after lemmatization it will join all the wirds again and will store it in new list called clean_df1.
# This clean_df1 is then converted to dataframe and a sentiment column is added to it which is from old dataframe df.
# Again it will add new column pre_clean_len to dataframe which is length of each tweet.
# Again check for any tweets greater than 140 characters.
# All the tweets is given to new variable x.
# All the tweets sentiments is given to new variable y and plot the see shaoe of both x and y variable.
# Now split the dataset in ratio 80:20 whereas 80% is for training and 20% is for testing.
# Split both the x and y variables.
# make a new instance vect of Tf-idf vectorizer and pass parameter as analyzer = "word" and ngrams_range = (1,1).
# this ngrams_range is for feature selection is given (1,1) it will only select unigrams, (2,2) only bigrams, (3,3) only trigrams, (1,2) unigrams and bigrams, (1,3) unigrams, bigrams and trigrams.
# we can also remove stop words over here by simply add new parameter stop_words = 'english'.
# fit or traing data tweets to vect.
# transform our training data tweets.
# transform our testing data tweets.
# import naive bayes and make object of it. Fit our traing tweets data and training tweets sentiment to the model.
# do 10- fold cross validation on the training data and  calculate the mean accuracy of it.
# predict the sentiments of testing tweets data.
# calculate the accuracy of predicted sentiments with the original tweets sentiment of testing data.
# plot the confusion matrix between original testing sentiment data and predicted sentiment.
# import logistic regression and make object of it. Fit our traing tweets data and training tweets sentiment to the model.
# do 10- fold cross validation on the training data and  calculate the mean accuracy of it.
# predict the sentiments of testing tweets data.
# calculate the accuracy of predicted sentiments with the original tweets sentiment of testing data.
# plot the confusion matrix between original testing sentiment data and predicted sentiment.
# import SVM and make object of it. Fit our traing tweets data and training tweets sentiment to the model.
# do 10- fold cross validation on the training data and  calculate the mean accuracy of it.
# predict the sentiments of testing tweets data.
# calculate the accuracy of predicted sentiments with the original tweets sentiment of testing data.
# plot the confusion matrix between original testing sentiment data and predicted sentiment.
import pandas as pd #import pandas
import numpy as numpy #import numpy
from sklearn.utils import shuffle # to shuffle the data 
import random # import random
import sklearn # import sklearn
import nltk # import nltk
from nltk.corpus import stopwords #import stop words
import re # import regular expression
from nltk.tokenize import word_tokenize # import word_tokenize
import matplotlib
import matplotlib.pyplot as plt #import matplotlib.pyplot 
df = pd.read_csv("../input/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None) #read csv file without header as dataframe
from sklearn.feature_extraction.text import TfidfVectorizer #  import TF-idf vectorizer
df = shuffle(df) # shuffle csv file
#tweets1 = df.iloc[0:9999,]
#tweets1.to_csv('tweets1.csv', sep=',')
%matplotlib inline
#data
print(sklearn.__version__)
print(matplotlib.__version__)
print(numpy.__version__)
print(pd.__version__)
print(nltk.__version__)
df.columns = ["sentiment", "id", "date", "query", "user", "text"] # give column names
#data
df = df.drop(["id", "date", "query", "user"], axis = 1) #drop some column from the dataframe 
#data
df.head() # get the first 5 rows from the dataframe
df.sentiment.value_counts() # count the number of sentiments with respect to their tweet(4 stands for positive tweet and 0 stands for negative tweet)
df['pre_clean_len'] = [len(t) for t in df.text] # add new column pre_clean_len to dataframe which is length of each tweet
plt.boxplot(df.pre_clean_len) # plot pre_clean_len column
plt.show()
df[df.pre_clean_len > 140].head(10)  # check for any tweets greater than 140 characters
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'        # remove @ mentions fron tweets
pat2 = r'https?://[^ ]+'        # remove URL's from tweets
combined_pat = r'|'.join((pat1, pat2)) #addition of pat1 and pat2
www_pat = r'www.[^ ]+'         # remove URL's from tweets
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",   # converting words like isn't to is not
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):  # define tweet_cleaner function to clean the tweets
    soup = BeautifulSoup(text, 'lxml')    # call beautiful object
    souped = soup.get_text()   # get only text from the tweets 
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")    # remove utf-8-sig codeing
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed) # calling combined_pat
    stripped = re.sub(www_pat, '', stripped) #remove URL's
    lower_case = stripped.lower()      # converting all into lower case
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case) # converting word's like isn't to is not
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)       # will replace # by space
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1] # Word Punct Tokenize and only consider words whose length is greater than 1
    return (" ".join(words)).strip() # join the words
nums = [0,400000,800000,1200000,1600000] # used for batch processing tweets
#nums = [0, 9999]
clean_tweet_texts = [] # initialize list
for i in range(nums[0],nums[4]): # batch process 1.6 million tweets                                                               
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))  # call tweet_cleaner function and pass parameter as all the tweets to clean the tweets and append cleaned tweets into clean_tweet_texts list
#clean_tweet_texts
word_tokens = [] # initialize list for tokens
for word in clean_tweet_texts:  # for each word in clean_tweet_texts
    word_tokens.append(word_tokenize(word)) #tokenize word in clean_tweet_texts and append it to word_tokens list
# word_tokens
# stop = set(stopwords.words('english'))
# clean_df =[]
# for m in word_tokens:
#     a = [w for w in m if not w in stop]
#     clean_df.append(a)
# Lemmatizing
df1 = [] # initialize list df1 to store words after lemmatization
from nltk.stem import WordNetLemmatizer # import WordNetLemmatizer from nltk.stem
lemmatizer = WordNetLemmatizer() # create an object of WordNetLemmatizer
for l in word_tokens: # for loop for every tokens in word_token
    b = [lemmatizer.lemmatize(q) for q in l] #for every tokens in word_token lemmatize word and giev it to b
    df1.append(b) #append b to list df1
# Stemming
# df1 = [] 
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# for l in word_tokens:
#     b = [ps.stem(q) for q in l]
#     df1.append(b)
#df
clean_df1 =[] # initialize list clean_df1 to join word tokens after lemmatization
for c in df1:  # for loop for each list in df1
    a = " ".join(c) # join words in list with space in between and giev it to a
    clean_df1.append(a) # append a to clean_df1
#clean_df1
clean_df = pd.DataFrame(clean_df1,columns=['text']) # convert clean_tweet_texts into dataframe and name it as clean_df
clean_df['target'] = df.sentiment # from earlier dataframe get the sentiments of each tweet and make a new column in clean_df as target and give it all the sentiment score
#clean_df
clean_df['clean_len'] = [len(t) for t in clean_df.text] # Again make a new coloumn in the dataframe and name it as clean_len which will store thw number of words in the tweet
clean_df[clean_df.clean_len > 140].head(10) # agin check id any tweet is more than 140 characters
X = clean_df.text # get all the text in x variable
y = clean_df.target # get all the sentiments into y variable
print(X.shape) #print shape of x
print(y.shape) # print shape of y
from sklearn.cross_validation import train_test_split #from sklearn.cross_validation import train_test_split to split the data into training and tesing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 1) # split the data into traing and testing set where ratio is 80:20


# X_train is the tweets of training data, X_test is the testing tweets which we have to predict, y_train is the sentiments of tweets in the traing data and y_test is the sentiments of the tweets  which we will use to measure the accuracy of the model
vect = TfidfVectorizer(analyzer = "word", ngram_range=(1,3)) # Get Tf-idf object and save it as vect. We can select features from here we just have simply change 
                                                                                     #the ngram range to change the features also we can remove stop words over here with the help of stop parameter
vect.fit(X_train) # fit or traing data tweets to vect
X_train_dtm = vect.transform(X_train) # transform our training data tweets
X_test_dtm = vect.transform(X_test)# transform our testing data tweets
from sklearn.naive_bayes import MultinomialNB # import Multinomial Naive Bayes model from sklearn.naive_bayes
nb = MultinomialNB(alpha = 10) # get object of Multinomial naive bayes model with alpha parameter = 10
nb.fit(X_train_dtm, y_train)# fit our both traing data tweets as well as its sentiments to the multinomial naive bayes model
from sklearn.model_selection import cross_val_score  # import cross_val_score from sklear.model_selection
accuracies = cross_val_score(estimator = nb, X = X_train_dtm, y = y_train, cv = 10) # do K- fold cross validation on our traing data and its sentimenst with 10 fold cross validation
accuracies.mean() # measure the mean accuray of 10 fold cross validation
y_pred_nb = nb.predict(X_test_dtm) # predict the sentiments of testing data tweets
from sklearn import metrics # import metrics from sklearn
metrics.accuracy_score(y_test, y_pred_nb) # measure the accuracy of our model on the testing data
from sklearn.metrics import confusion_matrix # import confusion matrix from the sklearn.metrics
confusion_matrix(y_test, y_pred_nb) # plot the confusion matrix between our predicted sentiments and the original testing data sentiments
from sklearn.linear_model import LogisticRegression # import Logistic Regression model from sklearn.linear_model
logisticRegr = LogisticRegression(C = 1.1) # get object of logistic regression model with cost parameter = 1.1
logisticRegr.fit(X_train_dtm, y_train)# fit our both traing data tweets as well as its sentiments to the logistic regression model
from sklearn.model_selection import cross_val_score # import cross_val_score from sklear.model_selection
accuracies = cross_val_score(estimator = logisticRegr, X = X_train_dtm, y = y_train, cv = 10) # do K- fold cross validation on our traing data and its sentimenst with 10 fold cross validation
accuracies.mean() # measure the mean accuray of 10 fold cross validation
y_pred_lg = logisticRegr.predict(X_test_dtm)  # predict the sentiments of testing data tweets
from sklearn import metrics # import metrics from sklearn
metrics.accuracy_score(y_test, y_pred_lg) # measure the accuracy of our model on the testing data
from sklearn.metrics import confusion_matrix # import confusion matrix from the sklearn.metrics
confusion_matrix(y_test, y_pred_lg) # plot the confusion matrix between our predicted sentiments and the original testing data sentiments
from sklearn.svm import LinearSVC # import SVC model from sklearn.svm
svm_clf = LinearSVC(random_state=0) # get object of SVC model with random_state parameter = 0
svm_clf.fit(X_train_dtm, y_train)# fit our both traing data tweets as well as its sentiments to the SVC model
from sklearn.model_selection import cross_val_score  # import cross_val_score from sklear.model_selection
accuracies = cross_val_score(estimator = svm_clf, X = X_train_dtm, y = y_train, cv = 10)# do K- fold cross validation on our traing data and its sentimenst with 10 fold cross validation
accuracies.mean() # measure the mean accuray of 10 fold cross validation
y_pred_svm = svm_clf.predict(X_test_dtm)  # predict the sentiments of testing data tweets
from sklearn import metrics  # import metrics from sklearn
metrics.accuracy_score(y_test, y_pred_svm)  # measure the accuracy of our model on the testing data
from sklearn.metrics import confusion_matrix # import confusion matrix from the sklearn.metrics
confusion_matrix(y_test, y_pred_svm)# plot the confusion matrix between our predicted sentiments and the original testing data sentiments
