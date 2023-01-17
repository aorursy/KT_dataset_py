import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '../input/'
df = pd.read_csv(data_dir + 'spam.csv', encoding='latin-1')
df.head()
df.dtypes
df.describe()
df.info()
df.shape
from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(df.v2,
                                                                 df.v1,
                                                                 test_size=0.2,
                                                                 random_state=0)
print('Content after spliting:')
print(data_train.head())
print(label_train.head())
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
data_train_cnt = vectorizer.fit_transform(data_train)
data_test_cnt = vectorizer.transform(data_test)
# print(data_train_cnt)
# print(len(vectorizer.vocabulary_))
# print(data_test_cnt)
wordFrequency = pd.DataFrame({'Term': vectorizer.get_feature_names(), 'Count': data_train_cnt.toarray().sum(axis=0)})
wordFrequency['Frequency'] = wordFrequency['Count'] / wordFrequency['Count'].sum()
plt.plot(wordFrequency['Count'])
plt.xlabel('Vocabulary_Index of Term')
plt.ylabel('Count')
plt.plot
plt.plot(wordFrequency['Frequency'])
plt.xlabel('Vocabulary_Index of Term')
plt.ylabel('Frequency')
plt.plot
wordFrequency_sort = wordFrequency.sort_values(by='Count', ascending=False)
wordFrequency_sort.head()
print(data_train_cnt.shape, label_train.shape, data_test_cnt.shape)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(data_train_cnt, label_train)
predictions = clf.predict(data_test_cnt)
print(predictions)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print('Accuracy: \n', accuracy_score(label_test, predictions))
print('Confusion Matrix: \n', confusion_matrix(label_test, predictions))
print('Classification Report: \n', classification_report(label_test, predictions))
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(clf, data_train_cnt, label_train, cv=20, scoring='accuracy')
print(accuracy)
print(accuracy.mean())