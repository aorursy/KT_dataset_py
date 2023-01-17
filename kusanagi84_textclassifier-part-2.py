# Import the needed packages

import numpy as np  # numerical processing

import pandas as pd # dataframes

import random       # random number generation   



from sklearn.feature_extraction import text 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer #Added this for calculating word importance

import spacy #Package for Tokenization, Lemmatization (normalizing word representation)
data = [('I am very happy', 'pos'),

        ('I am satisfied with this.', 'pos'),

        ('This is the best', 'pos'),

        ('This is extremely Good!', 'pos'),

        ('This is not bad', 'pos'),    #Problem 4: Added negation

        ('I am not that sad', 'pos'),  #Problem 4: Added negation

        ('I am quite positive about it', 'pos'), #Problem 1: Added 'positive' keyword

        ("I am really satisfied", 'pos'),



        ('I am extremely sad', 'neg'),

        ('This is very bad', 'neg'),

        ('It is BAD!', 'neg'),

        ('This is extremely bad', 'neg'),

        ('This is not good', 'neg'),  #Problem 4: Added negation

        ('I am not satisfied with this', 'neg'), #Problem 4: Added negation

        ('I am really very negative about this', 'neg') #Problem 1: Added 'negative' keyword 

       ]



#Make it into a DataFrame for easier processing later on

df = pd.DataFrame(data, columns=['text','label']) #make dataframe from dictionary

display(df.head(5)) #display first 10 to see it everything is ok

df.to_csv("train_data.csv", index=False) #save to csv
import nltk

from nltk.stem.porter import *

from nltk import word_tokenize



#Create own tokenization (method of splitting text into individual word tokens)

#Problem 3 - Fix typos by replacing words, e.g represent 'the best' and 'good' the same way

class LemmaTokenizer:

    def __init__(self):

        self.sp = spacy.load('en_core_web_sm') #load english language data

    def __call__(self, doc):

        replacements = {'goood': 'good'} #Problem 3: replace specific tokens, e.g., common typos

        tokens = self.sp(doc) #tokenize the document (split into words) - doc is one sentence

        return [replacements.get(t.lemma_,t.lemma_) for t in tokens] #replace some tokens

    

#More on Porter tokenization: https://stackabuse.com/python-for-nlp-tokenization-stemming-and-lemmatization-with-spacy-library/ 

class PorterTokenizer: 

    def __init__(self):

        self.stemmer = PorterStemmer() #Create porter tokenizer

    def __call__(self, doc):

        replacements = {'goood': 'good'} #Problem 3: replace specific tokens, e.g., common typos

        return [replacements.get(self.stemmer.stem(t), self.stemmer.stem(t)) for t in word_tokenize(doc)]  

    

#Sentences

sentences = ["These are the best", "This is good", "I'm good", "I am best"] #some example test sentences



pTok = PorterTokenizer() # Create an object of Porter Tokenizer

lTok = LemmaTokenizer() # Create an object of Lemma Tokenaize



#loop through the sentences

for sent in sentences:

    print("--Sentence: \""+sent+"\"--")

    pTokens = pTok.__call__(sent) #extract tokens from sentence using PorterTokenizer (essentially split into a list of words)

    lTokens = lTok.__call__(sent)

    print("\tPorter:", end=' ')

    for token in pTokens:

        print(token, end=' | ') #print the unchanged word -> lemmatized version

    print("\n\tLemma:", end=' ')

    for token in lTokens:

        print(token, end=' | ') #print the unchanged word -> lemmatized version

    print("\n")

    
#Converting words to number using Scikit-learn CountVectorizer



#print(text.ENGLISH_STOP_WORDS)

     

#Problem 3: tokenizer=PorterTokenizer(), LemmaTokenizer() - own tokenization that can fix typos and normalize words (lemmatization)

#Problem 2: stop_words=text.ENGLISH_STOP_WORDS.difference({'not'}) - removing stop words, such as 'is', 'this', 'that', 'it'

#          Porter specific - stop_words=['!','.','am','about','veri','is', 'i','it','quit','that','the','with','thi','extrem']

#          Lemma specific - stop_words=['this','that','the','i','.','!','with','-PRON-','about', 'be', 'very', 'quite', 'extremely', 'really']

#Problem 4: ngram_range=(1,2) - we will not rely on just inidividual words, but on tuples as well



count_vect = CountVectorizer(

    tokenizer = LemmaTokenizer(),

    stop_words = ['about', 'this', 'quite', 'that', 'the', '!', '.', 'i', '-PRON-', 'with'],

    ngram_range=(1,2)

    

) #create the CountVectorizer object for spliting sentence into words and replaceing them with numbers

                            

#Learn the vocabulary and transform our dataset from words to ids

word_counts = count_vect.fit_transform(df['text'].values) #Transform the text into numbers (bag of words features)



#get our vocabulary, index represents id

vocabulary = count_vect.get_feature_names()

#Loop through the vocabulary and print word id and words:

print("Vocabulary:", len(vocabulary), "phrases")

for word_id, word in enumerate(vocabulary):

    print(str(word_id)+" -> "+str(vocabulary[word_id])) #Show ID and word

    

print("\nSentences:", df['text'].values) #just show the list of our sentences for reference

    

print("\nSize (sentenes x words):", word_counts.shape) #Display the size of our array (list of lists)

print("Representation of our sentences as an array of word counts:") 

    

#Check how text sentences from our data were replaced by numbers

print(word_counts.toarray()) #represent the 
#Transform our dataset to prioritize unique words

tfidf_transformer = TfidfTransformer(use_idf=True)

word_importance = tfidf_transformer.fit_transform(word_counts) #calculates word importance based on data



#Print a list of our sentences

print("Sentences:", df['text'].values) #just show the list of our sentences for reference



#Display the size of our array (list of lists)

print("\nSize (sentenes x words):", word_importance.shape) 

print("Representation of our sentences as an array of importance scores:") 

#Display an array (list of lists) of tf-idf scored phrases

print(np.round(word_importance.toarray(),2))



#This looks closer at one sentence from our dataset

doc = 13 #Which document from our dataframe to show

feature_index = word_importance[doc,:].nonzero()[1] #get only non-zero (present) phrases in that document

tfidf_scores = zip(feature_index, [word_importance[doc, x] for x in feature_index]) #get the tf-idf score for each phrase

for w, s in [(vocabulary[i], s) for (i, s) in tfidf_scores]:

    print(w, s)
#Import different classifiers from Scikit-learn: 

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB 

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC



#Create the machine learning classifier object

clf = LinearSVC() 



X = word_importance #features of sentences (word importance numbers)

y = df['label'].values #correct labels for each document

#print("X:", X.toarray())

#print("y:", y)



#Train the classifer on our data

#First argument is an array (list of lists) representing words present in each sentence

#Second argument are the names of our classes ('pos' and 'neg' in this case)

clf.fit(X, y) #Change: word_counts changed to word_importance
#Import accuracy_score function that calculates classification accuracy

#check: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

from sklearn.metrics import accuracy_score



#predict labels for all the sentences

pred_label = clf.predict(X) #Change: word_counts change to word_importance

#Calculate accuracy on our training data, parameters: correct labels, predicted labels

print("Mean accuracy on training data:", accuracy_score(df['label'].values, pred_label))



#Let's put true classes and predictions next to one another

display(pd.DataFrame({"sentence":df['text'].values, 

                      "true label":df['label'].values, 

                      "predicted": pred_label}))
#Let's transform the sentence into numbers

sentences = ['I feeling not bad', "It is extremely bad", "This is not that good at all", "I am really satisfied"]

for s in sentences:

    print("'%s': %s" % (s, clf.predict( tfidf_transformer.transform( count_vect.transform([s])))) )



#Let's loop through individual extracted words and their importance

wc = count_vect.transform(["I am really very satisfied"]) #get phrase count

wi = tfidf_transformer.transform(wc) #get phrase importance weight

for word_id, importance in enumerate(wi.toarray()[0]):

    print("["+str(word_id)+"]"+str(vocabulary[word_id])+" -> "+str(importance))
# Problem 1: Previously unseen words

sentences = ["This is exceptionally bad!", "This is exceptionally good!", 

             "This is very positive!", "This is very negative!"]

print("--- Problem 1: Previously unseen words ---")

for s in sentences:

    wc = count_vect.transform([s]) #get phrase count

    wi = tfidf_transformer.transform(wc) #get phrase importance weight

    print("'%s': %s" % (s, clf.predict(wi)) )



#display(df)

    

# Problem 2: Unimportant words influencing the prediction, e.g;"it", "is", "extremely"

print("\n--- Problem 2: Unimportant words influencing the prediction ---")

sentences = ['This is extremely good', 'It is extremely good']

for s in sentences:

    wc = count_vect.transform([s]) #get phrase count

    wi = tfidf_transformer.transform(wc) #get phrase importance weight

    print("'%s': %s" % (s, clf.predict( wi)) )



# Problem 3: Typos, different pronounciation

print("\n--- Problem 3: Typos, different pronounciation ---")

sentences = ['This is extremely good', 'This is extremely goood']

for s in sentences:

    wc = count_vect.transform([s]) #get phrase count

    wi = tfidf_transformer.transform(wc) #get phrase importance weight

    print("'%s': %s" % (s, clf.predict(wi)) )

    #bow = count_vect.transform([s])

    #print(bow.toarray())



# Problem 4: Negations (importance of word ordering)

sentences = ["This is good!", "This is not good!", "This is bad!", "This is not that bad!"]

print("\n--- Problem 4: Word sequences (negations) --- ")

for s in sentences:

    wc = count_vect.transform([s]) #get phrase count

    wi = tfidf_transformer.transform(wc) #get phrase importance weight

    print("'%s': %s" % (s, clf.predict(wi)) ) 
from sklearn.model_selection import train_test_split #import a function that does splitting of the data

from sklearn import metrics # Import a number of metrics from scikit-learn



#This code splits our dataset into training subset (X_train, y_train) and testing subset (X_test, y_test)

X_train, X_test, y_train, y_test = train_test_split(df['text'].values, # The features of our text documents

                                                    y,                 # The correct labels for examples

                                                    test_size=0.20,    # How much of the data do we put aside for testing, 0.2 -> 20%

                                                    random_state=0)    #  A random 'seed' number used to impact the randomization (same 'seed' will give the same split)



print("--Train data---")

print(X_train)

print(y_train)



print("\n---Test data---")

print(X_test)

print(y_test)



#Let's train our classifer only on the training subset of our data

wc = count_vect.transform(X_train) #get phrase count

wi = tfidf_transformer.transform(wc) #get phrase importance weight

#Train the classifier on the test subset of the data

clf.fit(wi, y_train)



#Let's evaluate its performance only on unseen data (testing subset)

wc = count_vect.transform(X_test) #get phrase count

wi = tfidf_transformer.transform(wc) #get phrase importance weight

#Predict text labels for the test subset of the data

pred_label = clf.predict(wi)

print("Predictions:", pred_label)



print("\n---Accuracy on test data---")

print("Mean accuracy on test data:", accuracy_score(y_test, pred_label))
#import packages needed for cross-validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold



# Transform text data into phrase counts (wc) and then phrase importance scores (wi)

wc = count_vect.transform(df['text'].values) #get phrase count

wi = tfidf_transformer.transform(wc) #get phrase importance weight



# Assigning featues to X and correct labels to y

X = wi #input, features of our sentences

y = df['label'].values #output, correct labels



#Random split of the data int k subsets

cv = StratifiedKFold(3,              # The number of splits (subsets we divide our data into)           

                     shuffle=True,   # Should it be random?

                     random_state=0) # Randomization 'seed', if the same number is used the random split will be the same



#Perform the cross validation, train on n-1 splits and test on the remaining n-th split

scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')



print("Scores for each fold:", scores) # A list of individual scores for each split

print("Mean accuracy score: %0.2f" % np.mean(scores)) #Average score