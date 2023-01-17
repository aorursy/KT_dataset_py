# Opening and Reading the files into a list 

with open("../input/imdb_labelled.txt","r") as text_file:

    lines = text_file.read().split('\n')
# split the line by new-line character such that each line has one element of the list

lines[0:10]
# Read the lines from both the files and append in same list

with open("../input/yelp_labelled.txt","r") as text_file:

    lines += text_file.read().split('\n')

with open("../input/amazon_cells_labelled.txt","r") as text_file:

    lines += text_file.read().split('\n')
# split by tab and remove corrupted data if any or lines which are not tab seperated

lines = [line.split("\t") for line in lines if len(line.split("\t"))==2 and line.split("\t")[1]!='']
# print the lines one is string and another is integer 0 or 1

lines[0:10]
# Seperate the sentences

train_documents = [line[0] for line in lines ]

train_documents[0:10]
# Seperate the labels

train_labels = [int(line[1]) for line in lines]

train_labels[0:10]
from sklearn.feature_extraction.text import CountVectorizer
# Instatiate the Countvectorizer

count_vectorizer = CountVectorizer(binary='true')

# Train the documents

train_documents = count_vectorizer.fit_transform(train_documents)
train_documents
# print first document

print(train_documents[0])
# Training Phase

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB().fit(train_documents,train_labels)
# Test Phase

classifier.predict(count_vectorizer.transform(["this is the best movie"]))
classifier.predict(count_vectorizer.transform(["this is the worst movie"]))