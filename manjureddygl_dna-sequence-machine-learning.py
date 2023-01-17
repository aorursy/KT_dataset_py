import os

print(os.listdir("../input"))
# Loading library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 
human_data = pd.read_table('../input/human_data.txt')

human_data.head()
chimp_data = pd.read_table('../input/chimp_data.txt')

chimp_data.head()

dog_data= pd.read_table('../input/dog_data.txt')

dog_data.head()
# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)

def getKmers(sequence, size=6):

    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)

human_data = human_data.drop('sequence', axis=1)

chimp_data['words'] = chimp_data.apply(lambda x: getKmers(x['sequence']), axis=1)

chimp_data = chimp_data.drop('sequence', axis=1)

dog_data['words'] = dog_data.apply(lambda x: getKmers(x['sequence']), axis=1)

dog_data = dog_data.drop('sequence', axis=1)
human_texts = list(human_data['words'])

for item in range(len(human_texts)):

    human_texts[item] = ' '.join(human_texts[item])

y_human_data = human_data.iloc[:, 0].values                         
chimp_texts = list(chimp_data['words'])

for item in range(len(chimp_texts)):

    chimp_texts[item] = ' '.join(chimp_texts[item])

y_chimp_data = chimp_data.iloc[:, 0].values   
dog_texts = list(dog_data['words'])

for item in range(len(dog_texts)):

    dog_texts[item] = ' '.join(dog_texts[item])

y_dog_data = dog_data.iloc[:, 0].values 
print(human_texts[2])
print(chimp_texts[2])
print(dog_texts[2])
y_human_data
y_chimp_data
y_dog_data
# The n-gram size of 4 was previously determined by testing

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(4,4))

#X = cv.fit_transform(human_texts)

X_human=cv.fit_transform(human_texts)

X_chimp = cv.fit_transform(chimp_texts)

X_dog = cv.fit_transform(dog_texts)
print(X_human.shape)

print(X_chimp.shape)

print(X_dog.shape)
human_data['class'].value_counts().sort_index().plot.bar();
chimp_data['class'].value_counts().sort_index().plot.bar();
dog_data['class'].value_counts().sort_index().plot.bar();
# Splitting the human dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_human_train, X_human_test, y_human_train, y_human_test = train_test_split(X_human, 

                                                    y_human_data, 

                                                    test_size = 0.20, 

                                                    random_state=42)
# Splitting the chimp dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_chimp_train, X_chimp_test, y_chimp_train, y_chimp_test = train_test_split(X_chimp, 

                                                    y_chimp_data, 

                                                    test_size = 0.20, 

                                                    random_state=42)
# Splitting the dog dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_dog_train, X_dog_test, y_dog_train, y_dog_test = train_test_split(X_dog, 

                                                    y_dog_data, 

                                                    test_size = 0.20, 

                                                    random_state=42)
print(X_human_train.shape)

print(X_human_test.shape)
print(X_chimp_train.shape)

print(X_chimp_test.shape)
print(X_dog_train.shape)

print(X_dog_test.shape)
### Multinomial Naive Bayes Classifier ###

# The alpha parameter was determined by grid search previously

from sklearn.naive_bayes import MultinomialNB

classifier1 = MultinomialNB(alpha=0.1)

classifier1.fit(X_human_train, y_human_train)

classifier2 = MultinomialNB(alpha=0.1)

classifier2.fit(X_chimp_train, y_chimp_train)

classifier3 = MultinomialNB(alpha=0.1)

classifier3.fit(X_dog_train, y_dog_train)

y_human_pred = classifier1.predict(X_human_test)

y_chimp_pred = classifier2.predict(X_chimp_test)

y_dog_pred = classifier3.predict(X_dog_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_metrics(y_test, y_predicted):

    accuracy = accuracy_score(y_test, y_predicted)

    precision = precision_score(y_test, y_predicted, average='weighted')

    recall = recall_score(y_test, y_predicted, average='weighted')

    f1 = f1_score(y_test, y_predicted, average='weighted')

    return accuracy, precision, recall, f1
print("Confusion matrix\n")

print(pd.crosstab(pd.Series(y_human_test, name='Actual'), pd.Series(y_human_pred, name='Predicted')))

accuracy, precision, recall, f1 = get_metrics(y_human_test, y_human_pred)

print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
print("Confusion matrix\n")

print(pd.crosstab(pd.Series(y_chimp_test, name='Actual'), pd.Series(y_chimp_pred, name='Predicted')))

accuracy, precision, recall, f1 = get_metrics(y_chimp_test, y_chimp_pred)

print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
print("Confusion matrix\n")

print(pd.crosstab(pd.Series(y_dog_test, name='Actual'), pd.Series(y_dog_pred, name='Predicted')))

accuracy, precision, recall, f1 = get_metrics(y_dog_test, y_dog_pred)

print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))