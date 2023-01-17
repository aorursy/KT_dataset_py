#Well lets do this.
#At first import what we need.

import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

#panda does the reading  and saves as DataFrame
train = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train.shape)
train.head()
#Raw Unclean Review Text

print(train.review[0])

#Using BeautifulSoup to clean data initially & remove html tags and comments 


parsedRev = BeautifulSoup(train.review[0],"html.parser")


#Print the result to compare with Unclean data 
print(parsedRev.get_text())
#We can see the result has Numbers and Symbols in it. Not good for "Bag of Words". Lets start removing.



cleanRev = re.sub("[^a-zA-Z]"," ",parsedRev.get_text())
print(cleanRev)
# changing all the words to lowercase to create a "bag of words"
lcCleanRev = cleanRev.lower()

# Split to create an array from which  "stop words" will be removed
words = lcCleanRev.split()
# Stopwords from nltk are used in this phase
#some stopwords in english language are
## print(stopwords.words("english"))
#removing most common words from split array
bow = [w for w in words if w not in stopwords.words("english")]

#Bag of Words
print(bow)
# Function to do the make collection of cleaned text using all the reviews

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review,"html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]"," ",review_text)
    words = letters_only.lower().split()
    
    #create a set of stopwords so that we don't have to access corpus to search for a stopword
    stop = set(stopwords.words("english"))
    
    #removing stopwords from the raw_review
    meaningful_words = [w for w in words if w not in stop]
    
    return(" ".join(meaningful_words))
# Just Checking
check_review = review_to_words(train.review[0])
print(check_review)
#number of reviews
num_reviews = train.review.size
print("number of reviews :",num_reviews)
#storing all reviews in a list
clean_train_reviews = []
for i in range(num_reviews):
    clean_train_reviews.append(review_to_words(train.review[i]))
    if(i%5000==0):
        print("Breathe In... Breathe Out")
print("Cleaning Completed")
print("Creating a Bag of Words: ")

# We use CountVectorizer imported from sklearn.feature_extraction.text to create token counts of document


# Setting Parameters as None
vectorizer = CountVectorizer(analyzer="word",
                            tokenizer=None,
                            preprocessor=None,
                            stop_words=None,
                            max_features=5000)

# We train the classifer using fit_transform() method
train_data_features = vectorizer.fit_transform(clean_train_reviews)

#change the classifier into array
train_data_features = train_data_features.toarray()
print(train_data_features.shape)
#see all the features names
vocab = vectorizer.get_feature_names()
print(" , ".join(vocab[0:10])," . . . . "," , ".join(vocab[-10:]))

#frequency of each word is found using np.sum()
dist = np.sum(train_data_features,axis=0)
ct = 0
for tag,count in zip(vocab,dist):
    print(tag,":",count,end="\n ")
#Check if words starting with any alphabet is missing or not?
startswith = []
for val in vocab:
    if(val[0] not in startswith):
        startswith.append(val[0])
print(startswith)

#counting the total numbers of different words starting with each alphabet
counts = np.zeros((len(startswith)),dtype=np.int)
for val in vocab:
    index = startswith.index(val[0])
    counts[index] += 1
print(counts)
    
# Lets do some Plotting to visually show above information
plt.figure(1,figsize=(15,5))
plt.plot(counts)
nums = [i for i in range(26)]
plt.xticks(nums,startswith)
plt.grid()
plt.ylabel("frequency")
print(plt.show())


# Using Random Forrest Classifier for classification
forest = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
print("Fitting RandomForest")
forest = forest.fit(train_data_features,train["sentiment"])
print("RandomForest Done.")
# Using Naive-Bayes

naive = MultinomialNB()
print("Fitting NaiveBayes . . . ")
naive.fit(train_data_features,train["sentiment"])
print("NaiveBayes Done.")
adaboost = AdaBoostClassifier(n_estimators = 100)
print("Fitting AdaBoost . . . ")
adaboost.fit(train_data_features,train["sentiment"])
print("Fitting completed.")
#Now lets check against Test Cases
test = pd.read_csv("../input/testData.tsv",header=0,delimiter="\t",quoting=3)
print("Shape :",test.shape)
num_reviews = len(test["review"])
clean_test_reviews = []
print("Cleaning and parsing . . . . ")
for i in range(0,num_reviews):
    if((i+1)%5000 == 0):
        print(i+1," reviews processed . . .")
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)
print("Processing complete.")
test_data_features = vectorizer.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
print("Prediction using RandomForest")
result1 = forest.predict(test_data_features)
print("Prediction using Naive Bayes")
result2 = naive.predict(test_data_features)
print("Prediction using AdaBoost")
result3 = adaboost.predict(test_data_features)
print("Completed")
result = result1+result2+result3
for i in range(25000):
    if(result[i]==1):
        result[i]=0
    elif(result[i]==2):
        result[i]=1
    elif(result[i]==3):
        result[i]=1
output = pd.DataFrame(data = {"id":test["id"],"sentiment":result})
output.to_csv("submission.csv", index=False, quoting=3)



