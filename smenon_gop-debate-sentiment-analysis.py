import pandas as pd 
import numpy as np
import string, re
import nltk
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer, CountVectorizer
from sklearn import naive_bayes,metrics, linear_model,svm, grid_search
import time,random
import operator
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
punctuation = list(string.punctuation)
stop_list = stop_list + punctuation +["rt", 'url']

data = pd.read_csv("../input/Sentiment.csv")
classifier =[]
def preprocess(tweet):
    if type(tweet)!=type(2.0):
        tweet = tweet.lower()
        tweet = " ".join(tweet.split('#'))
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))','URL',tweet)
        tweet = re.sub("http\S+", "URL", tweet)
        tweet = re.sub("https\S+", "URL", tweet)
        tweet = re.sub('@[^\s]+','AT_USER',tweet)
        tweet = tweet.replace("AT_USER","")
        tweet = tweet.replace("URL","")
        tweet = tweet.replace(".","")
        tweet = tweet.replace('\"',"")
        tweet = tweet.replace('&amp',"")
        tweet  = " ".join([word for word in tweet.split(" ") if word not in stop_list])
        tweet  = " ".join([word for word in tweet.split(" ") if re.search('^[a-z]+$', word)])
        tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split(" ")])
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = tweet.strip('\'"')
    else:
        tweet=''
    return tweet

data['processed_text'] = data.text.apply(preprocess)
categories = data.sentiment.unique()
categories  = categories.tolist()

x = data.processed_text.values
y = data.sentiment.values
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33 )

def benchmark(clf,xtrain,ytrain,xtest,ytest,categories,vec_name):
    print('_' * 80)
    print("Training on "+ vec_name +" : ")
    print(clf)
    clf.fit(xtrain, ytrain)
    pred = clf.predict(xtest)
    score = metrics.accuracy_score(ytest, pred)
    print("Accuracy:   %0.3f" % score)
    print("Confusion Matrix:\n",confusion_matrix(pred, y_test),"\n")
    print("Classification Report:\n",metrics.classification_report(y_test, pred, target_names=categories))
    print('_' * 80)
vec_name = ['Count Vectorizer','Bigram Count Vectorizer','Hashing Vectorizer','Tfidf Vectorizer']
# loop = True 
# while loop :
#     print("""
#     0 - > Count Vectorizer
#     1 - > Bigram Count Vectorizer
#     2 - > Hashing Vectorizer
#     3 - > Tfidf Vectorizer
#     """)
#     ans = int(input("Choose Vectorizer: "))
#     if ans not in range(4):
#         print("Wrong Input, Try again!")
#         loop =True
#     else:
#         vec_index = ans
#         loop = False

# Vectorizer Definitions
vectorizer=[]
vectorizer.append(CountVectorizer(min_df = 0.01,max_df = 0.5, stop_words = 'english', analyzer='word'))
vectorizer.append(CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1))
vectorizer.append(HashingVectorizer(stop_words='english', non_negative=True))
vectorizer.append(TfidfVectorizer(min_df = 0.01, max_df = 0.5, sublinear_tf = True,stop_words = 'english'))

vec_index = 1
x_train_vec = vectorizer[vec_index].fit_transform(x_train)
x_test_vec = vectorizer[vec_index].transform(x_test)

alpha=[ float(1)/float((10**exponent)) for exponent in range(-2, 5)]
compare_alphas = []

if (vec_index !=2):
    feature_names = vectorizer[vec_index].get_feature_names()

x_train_vec = vectorizer[vec_index].fit_transform(x_train)
x_test_vec = vectorizer[vec_index].transform(x_test)
    
print('_' * 80)
for i in alpha: 
    mnb = naive_bayes.MultinomialNB(alpha = i)
    mnb.fit(x_train_vec, y_train)
    y_pred = mnb.predict(x_test_vec)
    compare_alphas.append((i,mnb.score(x_test_vec,y_test)))
    print('Multinomial Naive Bayes on '+ vec_name[vec_index] +' for alpha = '+ str(i) +' : ',
         round(metrics.accuracy_score(y_test, y_pred),5))

compare_alphas = sorted(compare_alphas, key=lambda x: x[1],reverse=True)
alpha = compare_alphas[0][0]
mnb = naive_bayes.MultinomialNB(alpha = alpha)
classifier.append(mnb)
alpha=[ float(1)/float((10**exponent)) for exponent in range(-2, 5)]
compare_alphas = []
print('_' * 80)
for i in alpha: 
    bnb = BernoulliNB(alpha = i)
    bnb.fit(x_train_vec, y_train)
    y_pred = bnb.predict(x_test_vec)
    compare_alphas.append((i,bnb.score(x_test_vec,y_test)))
    print('Bernoulli Naive Bayes on '+vec_name[vec_index] +' for alpha = '+ str(i) +' : ',
         round(metrics.accuracy_score(y_test, y_pred),5))

compare_alphas = sorted(compare_alphas, key=lambda x: x[1],reverse=True)
alpha = compare_alphas[0][0]
bnb = BernoulliNB(alpha= alpha)
classifier.append(bnb)
C =[ float(1)/float((10**exponent)) for exponent in range(-4, 5)]
compare_C = []
print('_' * 80)
for i in C:
    logit = linear_model.LogisticRegression(multi_class='multinomial'\
                                                  ,solver='newton-cg',C=i)
    logit.fit(x_train_vec, y_train)
    y_pred = logit.predict(x_test_vec)
    compare_C.append((i,logit.score(x_test_vec,y_test)))
    print('Logistic Regression on '+vec_name[vec_index] +' C = '+str(i)+' : ', 
          round(metrics.accuracy_score(y_test, y_pred),5))


compare_C = sorted(compare_C, key=lambda x: x[1],reverse=True)
C = compare_C[0][0]    
logit= linear_model.LogisticRegression(multi_class='multinomial'\
                                                  ,solver='newton-cg',C=C)
classifier.append(logit)


rfc = RandomForestClassifier(n_estimators=100)
classifier.append(rfc)
ridgeClf = RidgeClassifier(tol=1e-2, solver="sag")
classifier.append(ridgeClf)
perceptron = Perceptron(n_iter=50,alpha=100)
classifier.append(perceptron)
passive_aggressive = PassiveAggressiveClassifier(n_iter=50)
classifier.append(passive_aggressive)
kNN = KNeighborsClassifier(n_neighbors=10)
classifier.append(kNN)

for clf in classifier:
    benchmark(clf,x_train_vec,y_train,x_test_vec,y_test,categories,  vec_name[vec_index])