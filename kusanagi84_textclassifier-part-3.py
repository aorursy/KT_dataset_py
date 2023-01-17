import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.feature_extraction import text #text processing

from sklearn.feature_extraction.text import CountVectorizer #spliting and numeric representation (Bag-of-words/n-grams)

from sklearn.feature_extraction.text import TfidfTransformer #calculating word importance score (TF/IDF)
#Import a Scikit-learn dataset - 20newsgroups dataset

from sklearn.datasets import fetch_20newsgroups



#There are 20 different groups there, we will select only 4 for the tutorial

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']



#Download the newsgroup data - training subset (part of the data assigned for training the classifer) - check part 2 if this is confusing

#Make sure your Internet is on in Kaggle!!!

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)



#Download the newsgroup data - testing subset (part of the data assigned for testing the performance of the classifer) - check part 2 if this is confusing

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
#Display the length of the data

print("Number of documents in train:", len(twenty_train.data))

print("Number of documents in test:", len(twenty_test.data))



#Display names of newsgroup categories (our classification labels)

print("Category names:", twenty_train.target_names)



#Each document has its category represented as a number

print("Document label ids:", twenty_train.target[:10])



#We need to look-up the category name for the category number for each documents to get the actual newsgroup name as text

print("Category names for the first few documents:")

for t in twenty_train.target[:10]:

    print(twenty_train.target_names[t])



#Let's look at first few lines of some documents from the data

print("\n--- Example document ---")

print("\n".join(twenty_train.data[2000].split("\n")[:15]))
#The object type for the dataset is: 'sklearn.utils.Bunch'

print("Type of the data object used:", type(twenty_train)) 



#Turn the training subset of the data into a dataframe

train_df = pd.DataFrame(data=np.c_[twenty_train.data, [twenty_train.target_names[t] for t in twenty_train.target]], 

                             columns=['text','label'])

display(train_df)
#Turn the testing subset of the data into a dataframe

test_df = pd.DataFrame(data=np.c_[twenty_test.data, [twenty_test.target_names[t] for t in twenty_test.target]], 

                             columns=['text','label'])

display(test_df)
import nltk # NLP processing toolkit

from nltk.stem.porter import * # Importing Porter Stemming from NLTK

from nltk import word_tokenize # Importing Tokenizer from NLTK

import spacy # An NLP library for text processing



#Create own tokenization (method of splitting text into individual word tokens)

class SpacyTokenizer:

    def __init__(self):

        self.sp = spacy.load('en_core_web_sm') #Load english data

    def __call__(self, doc):

        tokens = self.sp(doc) #Tokenize the sentence (doc)  - essentially split into words

        return [t.lemma_ for t in tokens] #Look throgh the tokens and get their lemmatized representation (essentially normalized)

    

#Porter Stemming (chops off word endings to normalize representation, check the link above)

class PorterTokenizer:

    def __init__(self):

        self.stemmer = PorterStemmer() #Create Porter Stemmer object

    def __call__(self, doc):

        tokens = word_tokenize(doc) #Tokenize the sentence (doc) - essentially split into words

        return [self.stemmer.stem(t) for t in tokens]  #Loop through the tokens and apply Porter Stemming
from sklearn.pipeline import Pipeline #load the scikit-learn Pipeline module



#Load some classifiers

from sklearn.naive_bayes import MultinomialNB 

from sklearn.svm import LinearSVC



#Create the data processing pipeline - this does not process the data yet, just defines the processing steps we will execute later

clf_pipe = Pipeline([

    ('vect',  #Step 1: Split sentences into phrases and replace the phrases with numeric ids - check parts 1,2 

         CountVectorizer(

             ngram_range=(1,2), #how long sequences of words are we considering (1-only individual words)

             stop_words=text.ENGLISH_STOP_WORDS #what common words do we remove in processing, check: text.ENGLISH_STOP_WORDS

             #tokenize=PorterTokenizer() #uncomment to use own tokenizer (cell above)

         )

    ),

    ('tfidf', TfidfTransformer()), #Step 2: Calculate word importance using TF/IDF, emphasize unique words - check part 2

    ('clf', LinearSVC()), #Step 3: An ML classifer to use

])



#print(text.ENGLISH_STOP_WORDS) #show english stop words
#Pipeline based processing and training a classifer combined

#Inputs are: 

#         sentencs as text - train_df['text'].values

#         correct labels for traininf - train_df['label'].values

clf_pipe.fit(train_df['text'].values, train_df['label'].values) 
cv = clf_pipe.named_steps['vect'] # get the object for pipeline step under 'vect' - CountVectorizer

tfidf = clf_pipe.named_steps['tfidf'] # get the object for pipeline step under 'tfidf' - TfidfTransformer



vocab = cv.get_feature_names() # get the vocabulary - list of phrases found in the data

print("Number of phrases in vocabulary:", len(vocab)) # show how many there are

print("Id assigned to a phrase:",cv.vocabulary_.get(u'algorithm')) # get the id assigned to a particular phrase

print("Show first few phrases in the vocabulary:",vocab[20000:20050])



sentence = 'You need to love others and be benevolent'

bow = cv.transform([sentence]) # Transform an exacmple sentence to list of ids

print("\nSentence text:", sentence)

print("Sentence vector size:",bow.shape) # display the size of the numeric representation of this sentence

tf = tfidf.transform(bow) # calculate word importance (TF/IDF) based representation of the sentence



#display the non-zero (extracted) phrases for the above sentence

for w_idx in bow.nonzero()[1]:

    print(vocab[w_idx],"->",bow.toarray()[0,w_idx],"->",tf.toarray()[0,w_idx])
#Let's try to classifier some examples

docs_new = ['God is love', 'OpenGL on the GPU is fast', "These test results are worrysome", 'graphic cards are expensive', 

           "let's check the diagnosis", 'how did the surgery go'] # list of sentences to classify

predicted = clf_pipe.predict(docs_new) # predicting the labels (category) for our exemple sentences



#Printing sentences and the predicted labels

for doc, category in zip(docs_new, predicted):

    print('%r => %s' % (doc, category))
from sklearn import metrics  #classification evaluation metrics: accuracy score, confusion matrix, etc.



#Predicting labels for all test subset documents (we loaded this at the beginning)

predicted = clf_pipe.predict(test_df['text'].values)



#Calculating the accuracy on the test subset

print("Mean accuracy on test data:", metrics.accuracy_score(test_df['label'].values, predicted)) 



#Calculateing some more advances metrics for evaluation

print("More advanced metrics:")

print(metrics.classification_report(test_df['label'].values, predicted))
#Calculating the confusion-matrix - allows us to see which categories are hard to distinguish

metrics.confusion_matrix(test_df['label'].values, predicted)
#We can also plot it

metrics.plot_confusion_matrix(clf_pipe, #our classification pipline from above

                    test_df['text'].values, #text sentences from test set

                    test_df['label'].values, #correct labels from test set

                    cmap=plt.cm.Blues, #color scheme

                    normalize='pred', #whether to normalize everything (make it add up to 1.00), None if no normalization

                    values_format = '.2f') #how to display the values (float/integer), if not normalized counts use: 'd'

plt.show()