import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
human = pd.read_table('../input/humandata/human_data.txt')

human.head()
chimp = pd.read_table('../input/chimp-data/chimp_data.txt')

dog = pd.read_table('../input/dog-data/dog_data.txt')

chimp.head()

dog.head()
def getKmers(sequence, size=6):

    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)

human = human.drop('sequence', axis=1)

chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)

chimp = chimp.drop('sequence', axis=1)

dog['words'] = dog.apply(lambda x: getKmers(x['sequence']), axis=1)

dog = dog.drop('sequence', axis=1)
human.head()
human_texts = list(human['words'])

for item in range(len(human_texts)):

    human_texts[item] = ' '.join(human_texts[item])

y_h = human.iloc[:, 0].values
human_texts[0]
y_h
chimp_texts = list(chimp['words'])

for item in range(len(chimp_texts)):

    chimp_texts[item] = ' '.join(chimp_texts[item])

y_c = chimp.iloc[:, 0].values                       



dog_texts = list(dog['words'])

for item in range(len(dog_texts)):

    dog_texts[item] = ' '.join(dog_texts[item])

y_d = dog.iloc[:, 0].values                         
# This is equivalent to k-mer counting

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(4,4))

X = cv.fit_transform(human_texts)

X_chimp = cv.transform(chimp_texts)

X_dog = cv.transform(dog_texts)
print(X.shape)

print(X_chimp.shape)

print(X_dog.shape)

human['class'].value_counts().sort_index().plot.bar()
chimp['class'].value_counts().sort_index().plot.bar()
dog['class'].value_counts().sort_index().plot.bar()
# Splitting the human dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y_h, 

                                                    test_size = 0.20, 

                                                    random_state=42)
print(X_train.shape)

print(X_test.shape)
### Multinomial Naive Bayes Classifier ###

# The alpha parameter was determined by grid search previously

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.1)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
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
# Predicting the chimp, dog and worm sequences

y_pred_chimp = classifier.predict(X_chimp)

y_pred_dog = classifier.predict(X_dog)
# performance on chimp genes

print("Confusion matrix\n")

print(pd.crosstab(pd.Series(y_c, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))

accuracy, precision, recall, f1 = get_metrics(y_c, y_pred_chimp)

print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
# performance on dog genes

print("Confusion matrix\n")

print(pd.crosstab(pd.Series(y_d, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))

accuracy, precision, recall, f1 = get_metrics(y_d, y_pred_dog)

print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))