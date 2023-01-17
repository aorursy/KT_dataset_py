import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.model_selection import train_test_split



pd.reset_option('^display.', silent=True)



# Load half the data and separate target from predictors

X = pd.read_csv('../input/hatred-on-twitter-during-metoo-movement/MeTooHate.csv', nrows=300000)

X.dropna(axis=0, subset=['text', 'category'], inplace=True)

y = X.category

X.drop(['category'], axis=1, inplace=True)



# Drop columns not used for modelling

cols_to_drop = ['status_id', 'created_at', 'location']

X.drop(cols_to_drop, axis=1, inplace=True)



# Split the data while maintaining the proportion of hate/non-hate (stratify) 

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)



# Reset the index

X_train = X_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)

X_test_stats = X_test.copy()



print("Total training samples:", len(X_train))

print("Total test samples:", len(X_test))



X_train.head(10)
# Show descriptive statistics of training set

X_train.describe()
# Show how many values are non-null for each feature

X_train.info()
# Print a random tweet as a sample

sample_index = 25

print(X_train.iloc[sample_index])
# Plot the target label and notice that it is imbalanced



y_train.value_counts().plot(kind='bar')
# Convert the text feature into a vectors of tokens

from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',

                             lowercase=True, stop_words='english')

X_train_cv = cv.fit_transform(X_train.text)

X_test_cv = cv.transform(X_test.text)



# Scale numerical features (followers, retweets etc.)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols = ['favorite_count', 'retweet_count', 'followers_count', 'friends_count', 'statuses_count']

X_train_sc = scaler.fit_transform(X_train[cols])

X_test_sc = scaler.transform(X_test[cols])



# Merge the numerical features with our count vectors

import scipy.sparse as sp

train_count = sp.csr_matrix(X_train_cv)

train_num = sp.csr_matrix(X_train_sc)

X_train = sp.hstack([train_count, train_num])



test_count = sp.csr_matrix(X_test_cv)

test_num = sp.csr_matrix(X_test_sc)

X_test = sp.hstack([test_count, test_num])



# Save top words for training set

word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
# Top 20 words occuring in tweets

pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False).head(20)
# Train a Naive-Bayes classifier to classify hate/non-hate tweets



from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
# Plot scores and make a confusion matrix for non-hate/hate predictions



from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.metrics import confusion_matrix

n_classes = 2

cm = confusion_matrix(y_test, predictions, labels=range(n_classes))



print(f'Number of samples to classify: {len(X_test.toarray())}\n')

print(f'Accuracy score: {accuracy_score(y_test, predictions)}')

print(f'Precision score: {precision_score(y_test, predictions)}')

print(f'Recall score: {recall_score(y_test, predictions)}\n')

print(f'Confusion matrix: \n{cm}')
# Normalize the confusion matrix and plot it



plt.figure(figsize=(6,6))

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, square=True, annot=True, cbar=False,

            xticklabels=['non-hate', 'hate'], yticklabels=['non-hate', 'hate'])

plt.xlabel('Predicted label')

plt.ylabel('True label')
# Plot the ROC curve for the MNB classifier

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, predictions)

plt.figure(figsize=(8,8))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='MNB')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.show()
# Show how the first 50 test tweets were classified and their true label

testing_predictions = []

for i in range(len(X_test.toarray())):

    if predictions[i] == 1:

        testing_predictions.append('Hate')

    else:

        testing_predictions.append('Non-hate')

check_df = pd.DataFrame({'actual_label': list(y_test), 'prediction': testing_predictions, 'text':list(X_test_stats.text)})

check_df.replace(to_replace=0, value='Non-hate', inplace=True)

check_df.replace(to_replace=1, value='Hate', inplace=True)

check_df.iloc[:50]