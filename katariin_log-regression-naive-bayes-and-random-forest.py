# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load the data
trainjson = pd.read_json('../input/train.json', orient='columns')
#testjson = pd.read_json('../input/test.json', orient='columns')
#test_data = np.array(testjson)
train_data = np.array(trainjson)
#Delete empty rows
#train_data = [i for i in train_data if i[6] !=""] 
#train_data = np.array(train_data)
#print(len(train_data))
#Train data
train_data_X = train_data[:3000,6] # request_text 
train_data_Y = train_data[:3000,22] #requester_received_pizza
        
#print(len(train_data_X)) #3000
#print(len(train_data_Y)) #3000


#Test data
test_data_X = train_data[3000:,6] # request_text 
test_data_Y = train_data[3000:,22] #requester_received_pizza

#print(len(test_data_X)) #3000
#print(len(test_data_Y)) #3000

#for i in test_data_Y:
#    test_data_Y = np.where(i is False,0,1)

i=train_data_Y
train_data_Y=np.where(i==True, 1, 0)

a=test_data_Y
test_data_Y=np.where(a==True, 1, 0)

import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer

#Preparation for futher work

#FOR TRAIN DATA
#make new array, for all processed trained data
array_for_train = []
new_train_data_X = ""

for s in train_data_X:
    #first, we need to remove punctuation from sentences
    exclude = set(string.punctuation)
    new_train_data_X = ''.join(ch for ch in s if ch not in exclude)
    #add to array in lower case, to make words same, for example, Pizza == pizza
    array_for_train.append(new_train_data_X.lower())


#FOR TEST DATA
array_for_test = []
new_test_data_X = ""

for ss in test_data_X:
    #first, we need to remove punctuation from sentences
    exclude = set(string.punctuation)
    new_test_data_X = ''.join(ch for ch in ss if ch not in exclude)
    #add to array in lower case, to make words same, for example, Pizza == pizza
    array_for_test.append(new_test_data_X.lower())
   
#Test
print(array_for_test[2])

#https://www.tutorialspoint.com/python/python_stemming_and_lemmatization.htm
#stemming, lemmatization 
from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#porter=PorterStemmer()

# Snowball streamer - this algorithm was created by Martin Portrer.
# This algorithm consistently applies a series of rules of the English language -  cuts off the endings and suffixes
# It means, that plays, played and playing are the same words from word play
porter = SnowballStemmer("english") #language that we are looking for is engish

def stemSentence(sentence):
    token_words=word_tokenize(sentence) # splitting sentecne for pieces
    stem_sentence=[]
    stopWords = set(stopwords.words('english')) #establish stopwords algorithm. Stop words -  words that are not necessary in sentence or don´t give any meaning
    #stop words can be articles (a, the), prepositions (in, on..) and etc
    for word in token_words:
        if word not in stopWords: #throwing away these stop words
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")
    return "".join(stem_sentence)


#For training data
new_array_for_train = []
for w in array_for_train:
    new_array_for_train.append(stemSentence(w))

#For testing data
new_array_for_test = []
for ww in array_for_test:
    new_array_for_test.append(stemSentence(ww))
    
#Test
print(array_for_train[1])
print("  After Stemming    ")    
print(new_array_for_train[1])

#Most useful word
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
#wordcloud = WordCloud(background_color="lightblue").generate(" ".join([i for i in new_array_for_train]))
#plt.imshow(wordcloud)
#plt.axis("off")
#plt.title("Most used words")
#visualize most used words from when request resulted in free pizza
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
m=[]
for k in train_data:
    if k[22]:
        m.append(k[7])
wordcloud = WordCloud(background_color="white", max_words=50).generate(" ".join([i for i in m]))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Most used words that granted pizza")
#visualize most used words from when request did not result in free pizza
n=[]
for l in train_data:
    if not l[22]:
        n.append(l[7])
wordcloud = WordCloud(background_color="white", max_words=50).generate(" ".join([i for i in n]))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Most used words that did not grant pizza")
# Visualize how many requests resulted in free pizza
import plotly.graph_objs as go
import seaborn as sns
a=[]
for i in train_data:
    a.append(int(i[22]))
(sns.countplot(x = a)).set_title("# requests that resulted in free pizza");
print(len(new_array_for_train))
print(len(new_array_for_test))
#print(type(new_array_for_train))
new_array_for_train = np.array(new_array_for_train)
new_array_for_test = np.array(new_array_for_test)
#print(new_array_for_test[3])
from sklearn.feature_extraction.text import CountVectorizer
# Input
#new_array_for_train
#new_array_for_test
# Output - true/false
# train_data_Y
# test_data_Y

#print(np.array(new_array_for_train)[1])
#print(type(new_array_for_train))

#Initialize a CountVectorizer object
vect = CountVectorizer()
#Transform the training data
c_train = vect.fit_transform(new_array_for_train).toarray()

#Transform the test data
c_test = vect.transform(new_array_for_test).toarray()


# Printing first 20 features
# print(vect.get_feature_names()[:20])

print(type(c_train))
#c_test.shape
#c_train.shape
#NOT NECESSARY
#TfidfVectorizer - convert text vectors to objects that can be used as input to the assessment
#from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object
#vectorizer = TfidfVectorizer()
# Transform the training data: tfidf_train 
#t_train = vectorizer.fit_transform(new_array_for_train)
# Transform the training data: tfidf_train 
#t_test = vectorizer.fit_transform(new_array_for_test)

# Prints the first 20 features
#print(vectorizer.get_feature_names()[:20])
# Import the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Instantiate a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(c_train, train_data_Y)

# Create the prediction
pred = nb_classifier.predict(c_test)

# Calculate the accuracy score
score = metrics.accuracy_score(test_data_Y, pred)
print(score * 100 )
# Calculate the confusion matrix
cm = metrics.confusion_matrix(test_data_Y, pred)
pd.DataFrame(data = cm, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])
#visualize results accuracy
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

def model_report(title, test_data_Y, predictions):

    cm = metrics.confusion_matrix(test_data_Y, predictions)
    plt.figure(figsize=(3,3))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy: {0}'.format(round(metrics.accuracy_score(test_data_Y, predictions),2))
    plt.title(all_sample_title, size = 14)
    plt.show()
    
    fpr, tpr, threshold = metrics.roc_curve(test_data_Y, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
nb = BernoulliNB()

nb.fit(c_train , train_data_Y)
predictions = nb.predict(c_test )

model_report("Naive Bayes",test_data_Y, predictions)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(c_train, train_data_Y)
logpred = log_reg.predict(c_test)

# Calculate the accuracy score
score1 = metrics.accuracy_score(test_data_Y, logpred)
print(score1 * 100 )
# Calculate the confusion matrix
cm1 = metrics.confusion_matrix(test_data_Y, logpred)
pd.DataFrame(data = cm1, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])
##Random forest for bag-of-words? No.
#Random forest is a very good, robust and versatile method, however it’s no mystery that for high-dimensional sparse data 
#it’s not a best choice. But we already have logistic regression which is similar to SVM
# the main difference between logistic regression and SVM is that SVMs are designed to generate more complex decision boundaries
from sklearn.ensemble import RandomForestClassifier

randforclass = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state = 0)
randforclass.fit(c_train,train_data_Y)
randforclass_predict = randforclass.predict_proba(c_test)[:20]
#The first column shows persentage of false (0) probability and the second column show persentage of true(1) probability
print("  0 ", "         1")
print(randforclass_predict)

#Our output, what is need to be
print(test_data_Y[:20])
randforpred = randforclass.predict(c_test)

score2 = metrics.accuracy_score(test_data_Y, randforpred)
print(score * 100 )

# Calculate the confusion of random forestmatrix
cm2 = metrics.confusion_matrix(test_data_Y, randforpred)
#plot_confusion_matrix(cm,class_names)
pd.DataFrame(data = cm2, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])
#cross validation helps improve predicting accuracy
#cross validation, we will be comparing 2 machine learning algorithms
#They will be Logistic Regression and Random Forest
from sklearn.model_selection import cross_val_score
print("\n\nLog:")
#The function used for Kfolds is called cross_val_score()
#scoring=’accuracy’ means measure the algorithm for accuracy
#cv = 10, means that run the Logistic Regresion with our input/outputs ten times and measure the accuracy each time
kk = cross_val_score(log_reg, c_train, train_data_Y, scoring='accuracy', cv = 10)


#1 means, that the accuracy is 100%
print(kk)
#We use the mean() function to find the average accuracy
accuracy = kk.mean() * 100
print("Accuracy of Logistic Regression is: " , accuracy)

#Conclusion: Linear regression model was working food without cross validation.
print("Random Forest: ")
kkk = cross_val_score(randforclass, c_train, train_data_Y, scoring='accuracy', cv = 10)
print(kkk)
accuracy = kkk.mean() * 100
print("Accuracy of Random Forests is: " , accuracy)

#Conclusion: RF didn't overfit the data and prediction results were not bad, therefore, there was no need to use cross-validation (proved)
# But anyway, using cross-validation improves predictuon accuracy approx. by 0.44%
#t-test -> thi will give you a p-value
# you can also do e.g. box plots between the two result groups

from scipy.stats import ttest_ind

#ttest_ind - it's used for two indemendent samples
ttest_pair1 = ttest_ind(kk, kkk)
ttest_pair1