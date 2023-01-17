# Importing all necessary libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_table('../input/human-dna/human_dna.txt')
df.head()
df.info()
df.describe()
df.shape
# Creating a function to convert sequence strings into k-mer words, default size = 6 (hexamer words).

def getKmers(sequence, size=6):

    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
df['words'] = df.apply(lambda x: getKmers(x['sequence']), axis=1)

df = df.drop('sequence', axis=1)
df.head()
human_texts = list(df['words'])

for item in range(len(human_texts)):

    human_texts[item] = ' '.join(human_texts[item])

y_data = df.iloc[:, 0].values
print(human_texts[2])
y_data
# Creating the Bag of Words model using CountVectorizer().

# This is equivalent to k-mer counting.

# The n-gram size of 4 was previously determined by testing.

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(4,4))

X = cv.fit_transform(human_texts)
print(X.shape)
df['class'].value_counts().sort_index().plot.bar()
# Splitting the human dataset into the training testing sets.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.20, random_state=42)
print(X_train.shape)

print(X_test.shape)
# Implementing a Multinomial Naive Bayes Classifier. 

# The alpha parameter was determined by grid search previously.

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.1)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Getting the accuracy of the model.

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("Confusion matrix\n")

print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

def get_metrics(y_test, y_predicted):

    accuracy = accuracy_score(y_test, y_predicted)

    precision = precision_score(y_test, y_predicted, average='weighted')

    recall = recall_score(y_test, y_predicted, average='weighted')

    f1 = f1_score(y_test, y_predicted, average='weighted')

    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)

print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))