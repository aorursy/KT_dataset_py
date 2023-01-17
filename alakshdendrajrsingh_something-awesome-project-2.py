#importing libraries
from sklearn.model_selection import train_test_split
import json
import numpy as np
import random
###CREATING CLASSES TO MAKE CODE CLEANER###
##########################
#sentiment class just to assign the values
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

############################
#Review class assigns value to the review list => text, score and sentiment 
class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
    
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        if self.score == 3:
            return Sentiment.NEUTRAL
        if self.score >=4:
            return Sentiment.POSITIVE
        
#############################
#this class balances the positive and negative reviews by shortning the positive reviews 
class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
        
   
    def evenly_distribute(self):
        negative = list(filter(lambda x : x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x : x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[ :len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)
        
    def get_text(self): #assigns text value to the train_x
        return [x.text for x in self.reviews]
        
    def tk_sentiment(self): #assign sentiment value to the train_y
        return [x.sentiment for x in self.reviews]
        
        
#importing file and loading the neccesary data in the reviews list
import json
file_name = '../input/reviewsdb/Books_small_10000.json'

reviews =[]
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall'])) 
        
reviews[5].sentiment
#the train_test_split function
training, test = train_test_split(reviews, test_size= 0.33, random_state= 42)#separating dataset for traning and testing

#making the objects of the ReviewContainer class for training and testing
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)
#running the function for training dataset
#distributing the reviews evenly so that our classifier is not biased towards on type of sentiment
train_container.evenly_distribute()
#separating review text and sentiment
train_x = train_container.get_text()
train_y = train_container.tk_sentiment()

#running the fuctions for testing dataset
test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.tk_sentiment()
####################################
#train_y
train_y.count(Sentiment.NEGATIVE)
#converting the text of the reviews list to numbers throught CountVectorizer class 
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)
#numbers corresponding to the text
print(train_x[19])
print(train_x_vectors[19])
from sklearn import svm

clf_svm = svm.SVC(kernel='linear') #using linear kernel
clf_svm.fit(train_x_vectors, train_y) #traning our svm model
#reading a review
test_x[6]
#sentiment prediction of the above review by our classifier 
clf_svm.predict(test_x_vectors[6])
from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()
#training our model
clf_dec.fit(train_x_vectors, train_y)

#storing the predicted sentiments
pred_dec = clf_dec.predict(test_x_vectors)
test_x[17]
#sentiment prediction of the above review by our classifier 
clf_dec.predict(test_x_vectors[17])
from sklearn.linear_model import LogisticRegression

#making an object of logistic Regeression function
clf_log = LogisticRegression()
#traning our model
clf_log.fit(train_x_vectors, train_y)

#prediction
pred_dec = clf_log.predict(test_x_vectors[0])
test_x[31]
#sentiment prediction of the above review by our classifier 
clf_dec.predict(test_x_vectors[31])
print("Accuracy of Support Vector Machine: " + str(clf_svm.score(test_x_vectors, test_y)))
print("Accuracy of Decision Tree: " + str(clf_dec.score(test_x_vectors, test_y)))
print("Accuracy of Logistic Regression: " + str(clf_log.score(test_x_vectors, test_y)))

#print the number of positive and negative reviews
#to check our model is not baised
print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))
opt = 'y'
while opt=='y':
    input_review = input("Enter your review: ")
    input_review = [input_review]
    vector_r = vectorizer.transform(input_review)
    predict_r_svm = clf_svm.predict(vector_r)
    predict_r_dec = clf_dec.predict(vector_r)
    predict_r_log = clf_log.predict(vector_r)
    print("===========================================================")
    print("Sentiment of review by algorithms : ")
    print("Support Vector Machine : "+ predict_r_svm[0])
    print("Decision Tree : "+ predict_r_dec[0])
    print("Logistic Regression : "+ predict_r_log[0])
    ch = input("Want to predict more (y/n) :")
    print("===========================================================")
    opt = ch

