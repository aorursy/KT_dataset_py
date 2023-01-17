import pandas as pd

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score
data_directory = "../input/"

data = pd.read_csv(data_directory + '/spam.csv', encoding='latin-1') 

data.head()
text_train, text_test, tag_train, tag_test = train_test_split(data.v2,data.v1, test_size=0.25,random_state=0) 

print(text_train.shape, text_test.shape,tag_train.shape,tag_test.shape)

print()

print("After Spliting:")

print(text_train.head())

print()

print(tag_train.head())
vectorizer = CountVectorizer()



text_train_count = vectorizer.fit_transform(text_train)

text_test_count  = vectorizer.transform(text_test)

# print ('Summary:')

# print (text_train_count.shape)

# print (text_test_count.shape)

# print (vectorizer.vocabulary_)

# Occurrences take data_train_count to become a matrix like 2d array, then use sum with axis = 0 to add value from each columns vertically to get an array that each contains the 

# total sume of each letter

wordFreqeuncy = pd.DataFrame({'Word': vectorizer.get_feature_names(), 'occurrences':text_train_count.toarray().sum(axis=0)})

# print(data_train_count.toarray().sum(axis=0))

wordFreqeuncy['frequency'] = wordFreqeuncy['occurrences']/np.sum(wordFreqeuncy['occurrences'])

plt.plot(wordFreqeuncy.occurrences) # plot occurance with occurance id

plt.show()

wordFreqeuncySort = wordFreqeuncy.sort_values(by=['occurrences'], ascending=False)

wordFreqeuncySort.head()
clf = MultinomialNB()

clf.fit(text_train_count, tag_train)

predictions = clf.predict(text_test_count)

print(predictions)
# Compare tag_test with our prediction: 

print ("accuracy_score: ",accuracy_score(tag_test, predictions))

print()

print("classification_report")

print (classification_report(tag_test, predictions))

print()

print("confusion_matrix")

print (confusion_matrix(tag_test, predictions))

print()

print("cross_val_score")

cross_val = cross_val_score(clf, text_train_count, tag_train, cv=20, scoring='accuracy')

print (cross_val)

print()

print("Mean of cross_val_score: ",np.mean(cross_val))