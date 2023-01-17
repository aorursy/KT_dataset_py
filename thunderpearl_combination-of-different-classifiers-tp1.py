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

# coding: utf-8

# In[1]:



# coding: utf-8

# In[1]:


# We are using pickle classifier
# pickle is used when we want to use our trained algorithm again
# This is the code for naive bayes and this is sometimes also called as stupid bayes algorithm
import nltk
import random

# corpus is mostly related to speech or reviews
from nltk.corpus import movie_reviews

# Importing pickle 
import pickle

# Our jo is to combine all the above classifiers so that we can select the category which is voted by most classifiers....
from nltk.classify import ClassifierI

# For getting the mode function
from statistics import mode


# In[2]:


# creating a class for starting a voting system
# ClassifierI is the parent class to the My_vote_classifier class
class My_vote_classifier(ClassifierI):
    
    # inside the constructor we are gonna pass a list of classifiers
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
        
    # defining our classify function
    
    # These methods are the abstract methods which are needed to be implemented
      # So, we can not change their name
    def classify(self,features):
        votes = []
        for x in self._classifiers:
            v = x.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):
        votes = []
        for y in self._classifiers:
            v = y.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        
        conf = choice_votes / len(votes)
        return conf


        
        

my_documents1 = [(list(movie_reviews.words(fileid)),category) 
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]


# In[3]:


# Shuffled the my_documents1
random.shuffle(my_documents1)

print(type(my_documents1))

print(len(my_documents1))

# Accessing the first index tuple which have a list of words and category
print(my_documents1[1])

print(my_documents1[2])

print(my_documents1[3])


# In[4]:


my_all_words = []


# In[5]:


# Storing all the words of into a list from movie_reviews
for word in movie_reviews.words():

    # converting everything into the lower case
    my_all_words.append(word.lower())


# In[6]:


# It gets the frequency of all the words....
my_all_words_freq = nltk.FreqDist(my_all_words)


# In[8]:


type(my_all_words_freq)


# In[9]:


len(my_all_words_freq)


# In[10]:


# This will print the ddictionary for all the words with their respective frequency from higher to lower ....

my_all_words_freq


# In[11]:


print(my_all_words_freq.keys())


# In[12]:


# Printing starting 3000 keys 

print(list(my_all_words_freq.keys())[:3000])


# In[13]:


# Storing all the keys or say words from 0 to 3000 in a variable called my_word_features
# We took 3000 because there are lots of non word characters in the data hence,we wanted to have healthy 
 # datasets for the words....
my_word_features = list(my_all_words_freq.keys())[:3000]


# In[14]:


# Defining a function to get the features
def get_features(documents):
    words = set(documents)
    features = {}
    for x in my_word_features:
        # filling our above empty dictionary with boolean values for the keys
        features[x] = (x in words)
    return features


# In[15]:


print((get_features(movie_reviews.words('neg/cv000_29416.txt'))))


# In[16]:


# my_documents1 is a list of tuples and in each tuple we have lots of word and then category....
# So, every time in each for loop iteration we are passing some words with category and the function get_features()
 # will return key word and its boolean value

my_feature_sets = [(get_features(rev),category) for (rev, category) in my_documents1]


# In[17]:


len(my_feature_sets)


# In[18]:


type(my_feature_sets)


# In[19]:


my_feature_sets[1]


# In[20]:


my_feature_sets[2]


# In[21]:


my_training_set = my_feature_sets[:1900]


# In[22]:


my_testing_set = my_feature_sets[1900:]


# In[23]:


# posterior = prior_occurence * likelihood / evidence

my_classsifier = nltk.NaiveBayesClassifier.train(my_training_set)


# In[24]:


print("The accuracy of my classifier in percentage is: ", (nltk.classify.accuracy(my_classsifier, my_testing_set))*100)


# In[25]:


my_classsifier.show_most_informative_features(10)


# In[26]:


# b along with w refers for bytes

save_my_classifier_1 = open("naivebayes.pickle","wb") 


# In[27]:


# We are dumping the classifier "my_classifier" in the object "save_my_classifier_1"

pickle.dump(my_classsifier, save_my_classifier_1)


# In[28]:


# closing the object

save_my_classifier_1.close()


# In[32]:


# Now, we will like to open our saved classifier again in an object

my_classifer_saved_1 = open("naivebayes.pickle","rb")


# In[33]:




my_classifier_2 = pickle.load(my_classifer_saved_1)


# In[34]:


print("My Original naive bayes accuracy of my 2nd which is loaded classfier is: ",(nltk.classify.accuracy(my_classifier_2,my_testing_set))*100)


# In[5]:


# scikit-learn have it's own ml algos
# scikit-leanr have various classification, clustering and regression algorithms
from nltk.classify.scikitlearn import SklearnClassifier


# In[6]:


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB


# In[7]:


# Using the classifier of scikit-learn

my_mnb_classifier_1 = SklearnClassifier(MultinomialNB())


# In[8]:


# Now, we will train classifier with the training data

my_mnb_classifier_1.train(my_training_set)


# In[9]:


# Printing the accuracy of scikit-learn classifier

print("Accuracy of 1st Multinomial classifier from the scikit-learn in percentage is:- ",(nltk.classify.accuracy(my_mnb_classifier_1,my_testing_set))*100)


# In[13]:


# Trying the Bernouli classifier from scikit-learn

my_bernouli_classifier_1 = SklearnClassifier(BernoulliNB())


# In[14]:


my_bernouli_classifier_1.train(my_training_set)


# In[15]:



print("Accuracy of 1st Bernouli classifier from the scikit-learn in percentage is:- ",
      (nltk.classify.accuracy(my_bernouli_classifier_1,my_testing_set))*100)


# In[16]:


# Now, trying the logistic regression from sklearn

from sklearn.linear_model import LogisticRegression, SGDClassifier


# In[17]:


from sklearn.svm import SVC, LinearSVC, NuSVC


# In[18]:


my_logistic_regression_classifier_1 = SklearnClassifier(LogisticRegression())   


# In[19]:


my_logistic_regression_classifier_1.train(my_training_set)


# In[20]:


print("Accuracy of 1st Logistic Regression classifier from the scikit-learn in percentage is:- ",
      (nltk.classify.accuracy(my_logistic_regression_classifier_1,my_testing_set))*100)


# In[21]:


my_sgdc_classifier_1 = SklearnClassifier(SGDClassifier())


# In[22]:


my_sgdc_classifier_1.train(my_training_set)


# In[23]:


print("Accuracy of 1st SGD classifier from the scikit-learn in percentage is:- ",
      (nltk.classify.accuracy(my_sgdc_classifier_1,my_testing_set))*100)


# In[24]:


my_svc_classifier_1 = SklearnClassifier(SVC())


# In[25]:


my_svc_classifier_1.train(my_training_set)


# In[26]:


print("Accuracy of 1st SVC classifier from the scikit-learn in percentage is:- ",
      (nltk.classify.accuracy(my_svc_classifier_1,my_testing_set))*100)


# In[27]:


# Just try other two classifiers LinearSVC and NuSVC as well 
my_linear_svc_classifier_1 = SklearnClassifier(LinearSVC())


# In[28]:


my_linear_svc_classifier_1.train(my_training_set)


# In[29]:


print("Accuracy of 1st linear svc classifier from the scikit-learn in percentage is:- ",
      (nltk.classify.accuracy(my_linear_svc_classifier_1,my_testing_set))*100)


# In[30]:


my_nu_svc_classifier_1 = SklearnClassifier(NuSVC())


# In[31]:


my_nu_svc_classifier_1.train(my_training_set)


# In[32]:


print("Accuracy of 1st Nu SVC classifier from the scikit-learn in percentage is:- ",
      (nltk.classify.accuracy(my_nu_svc_classifier_1,my_testing_set))*100)


# Creating the object of the class defined above
# And, the constructor we will be passing all of our classifiers
voted_classifier_1 = My_vote_classifier(my_mnb_classifier_1,my_bernouli_classifier_1,
                                     my_logistic_regression_classifier_1,my_sgdc_classifier_1,my_svc_classifier_1,
                                    my_linear_svc_classifier_1,my_nu_svc_classifier_1)
print("Accuracy of voted classifier from the scikit-learn in percentage is:- ",
      (nltk.classify.accuracy(voted_classifier_1,my_testing_set))*100)

print(type(my_testing_set))
print(my_testing_set[0][0])
print("Classification: ",voted_classifier_1.classify(my_testing_set[0][0]),
      " Confidence:- ",voted_classifier_1.confidence(my_testing_set[0][0]))
