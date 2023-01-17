# Import the needed packages

import numpy as np  # numerical processing

import pandas as pd # dataframes

import random       # random number generation   



from sklearn.feature_extraction import text       #Text processing functions from Sciki-learn

from sklearn.feature_extraction.text import CountVectorizer    #For numeric representation of words
data = [('I am very happy', 'pos'),

        ('I am satisfied with this.', 'pos'),

        ('This is the best', 'pos'),

        ('This is extremely Good!', 'pos'),

        

        ('I am extremely sad', 'neg'),

        ('This is very bad', 'neg'),

        ('It is BAD!', 'neg'),

        ('This is extremely bad', 'neg')]



#Make it into a DataFrame for easier processing later on

df = pd.DataFrame(data, columns=['text','label']) #make dataframe from dictionary

display(df.head(10)) #display first 10 rows to see it everything is ok

df.to_csv("train_data.csv", index=False) #save to csv
#Converting words to word ids using Scikit-learn CountVectorizer

count_vect = CountVectorizer() #create the CountVectorizer object

print("Sentences:", df['text'].values)

count_vect.fit(df['text'].values) #fit into our dataset



#Get a list of unique words found in the document (vocabulary)

word_list = count_vect.get_feature_names()

print("\nUnique words:", word_list)



#Let's look up the id of a particular word: 'happy'

print("\nId assigned to 'happy':", count_vect.vocabulary_.get('happy'))

#Let's look up what word has been assigned id of 3:

print("Word under id=3:", word_list[0])
print("List all the words and their ids")

#Check all the words that were extracted and their ids:

for word_id, word in enumerate(word_list):

    print(str(word_id)+" -> "+str(word_list[word_id])) #Show ID and word

    

#Transform our dataset from words to ids

print("\nSentence:", df['text'].values) #just show the list of our sentences for reference



word_counts = count_vect.transform(df['text'].values) #Transform the text into numbers (bag of words features)

print("\nSize (sentenes x words):", word_counts.shape) #Display the size of our array (list of lists)

print("Representation of our sentences as an array of ids:")

#Check how text sentences from our data were replaced by numbers

print(word_counts.toarray()) #represent the 
## Let's try this on a sentence that is completely and not present in our training data

new_sentence = "Such a very good good good product!"

bow = count_vect.transform([new_sentence]) #transform the text to features (bag of words)

print("New, unseen sentence: "+str(new_sentence)+"\nas numbers:"+str(bow.toarray()))



#Let's loop through individual extracted words and their counts

for word_id, count in enumerate(bow.toarray()[0]):

    print("["+str(word_id)+"] "+str(word_list[word_id])+" -> "+str(count))
#Import classifiers from Scikit-learn: 

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



#Create the machine learning classifier object

clf = LogisticRegression() #SVC() #DecisionTreeClassifier() #LogisticRegression()



X = word_counts #data features (word counts)

y = df['label'].values #correct labels

print('X:', X.toarray())

print('y:', y)



#Train the classifer on our data

#First argument is an array (list of lists) representing words (ids) present in each sentence

#Second argument are the names of our classes ('pos' and 'neg' in this case)

clf.fit(X, y) #training on X and y
#Import accuracy_score function that calculates classification accuracy

from sklearn.metrics import accuracy_score



#predict labels for all the sentences

pred_label = clf.predict(X) #predicting based on X

print("Predicted labels:", pred_label)



#Calculate accuracy on our training data, parameters: correct labels, predicted labels

print("Mean accuracy on training data:", accuracy_score(y, pred_label))



#Let's put true classes and predictions next to one another in a dataframe

display(pd.DataFrame({"sentence":df['text'].values, 

                      "true label":df['label'].values, 

                      "predicted": pred_label}))
#Let's transform the sentence into numbers

sentences = ['I feel very much excepionally satisfied right now', 'I feel very bad!!!']

for s in sentences:

    bow = count_vect.transform([s]) #turn text into word id representation - bag-of-words

    #print(bow.toarray())

    print("'%s': %s" % (s, clf.predict(bow)) )



#Let's loop through individual extracted words and their counts

sentence = "I feel very sad!!!"

bow = count_vect.transform([sentence]) #turn text into word id representation - bag-of-words

print(f"\nSentence \"{sentence}\" representation:")

for word_id, count in enumerate(bow.toarray()[0]):

    print("["+str(word_id)+"] "+str(word_list[word_id])+" -> "+str(count))
sentences = ["This is exceptionally bad!", "This is exceptionally good!", 

             "This is very positive!", "I don't know"]

print("--- Problem 1: Previously unseen words ---")

for s in sentences:

    print("'%s': %s" % (s, clf.predict( count_vect.transform([s]))) )

    #bow = count_vect.transform([s])

    #print(bow.toarray())
display(df)
sentences = ['This is extremely good', 'It is extremely good']

for s in sentences:

    print("'%s': %s" % (s, clf.predict( count_vect.transform([s]))) )
sentences = ['This is exmkljtremely good', 'This is extremely goood']

for s in sentences:

    print("'%s': %s" % (s, clf.predict( count_vect.transform([s]))) )

sentences = ["This is good!", "This is not good!", 'I am happy', 'I am not happy', "This is bad!", "This is not that bad!"]

for s in sentences:

    print("'%s': %s" % (s, clf.predict( count_vect.transform([s]))) ) 

    #bow = count_vect.transform([s])

    #print(bow.toarray())