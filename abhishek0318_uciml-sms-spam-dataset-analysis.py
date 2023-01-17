import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/spam.csv', encoding='latin1', usecols=[0, 1], names=['is spam', 'text'])[1:]

df['is spam'] = df['is spam'].apply(lambda x: 0 if x == 'ham' else 1)

df.head()
df.groupby('is spam').describe()
def percentage_digits(text):

    counter = 0

    for i in text:

        if i.isdigit():

            counter += 1

    return (counter / len(text)) * 100



df['percentage digits'] = df['text'].apply(percentage_digits)

df.head()
plt.hist(df[df['is spam'] == 1]['percentage digits'], bins=10, range=(0, 20), rwidth=0.8)

plt.xlabel('Percentage digits')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage digits of spam messages.')

plt.show()
plt.hist(df[df['is spam'] == 0]['percentage digits'], bins=10, range=(0, 20), rwidth=0.8)

plt.xlabel('Percentage digits')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage digits of non-spam messages.')

plt.show()
df['percentage digits'] = df['percentage digits'].apply(lambda x: 0 if x < 5 else 1)

df.head(2)
df.groupby(['is spam']).mean()
def percentage_question_mark(text):

    counter = 0

    for i in text:

        if i == '?':

            counter += 1

    return (counter / len(text)) * 100



df['percentage question mark'] = df['text'].apply(percentage_question_mark)

df.head()
plt.hist(df[df['is spam'] == 1]['percentage question mark'], bins=10, range=(0, 4), rwidth=0.8)

plt.xlabel('Percentage question mark')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage question mark of spam messages.')

plt.show()
plt.hist(df[df['is spam'] == 0]['percentage question mark'], bins=10, range=(0, 4), rwidth=0.8)

plt.xlabel('Percentage question mark')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage question mark of non-spam messages.')

plt.show()
df['percentage question mark'] = df['percentage question mark'].apply(lambda x: 0 if x < 1 else 1)

df.head(2)
df.groupby(['is spam']).mean()
def percentage_exclamation_mark(text):

    counter = 0

    for i in text:

        if i == '!':

            counter += 1

    return (counter / len(text)) * 100



df['percentage exclamation mark'] = df['text'].apply(percentage_exclamation_mark)

df.head()
plt.hist(df[df['is spam'] == 1]['percentage exclamation mark'], bins=10, range=(0, 3), rwidth=0.8)

plt.xlabel('Percentage exclamation mark')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage exclamation mark of spam messages.')

plt.show()
plt.hist(df[df['is spam'] == 0]['percentage exclamation mark'], bins=10, range=(0, 3), rwidth=0.8)

plt.xlabel('Percentage exclamation mark')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage exclamation mark of non-spam messages.')

plt.show()
df['percentage exclamation mark'] = df['percentage exclamation mark'].apply(lambda x: 0 if x < 0.5 else 1)

df.head(2)
df.groupby(['is spam']).mean()
def percentage_period(text):

    counter = 0

    for i in text:

        if i == '.':

            counter += 1

    return (counter / len(text)) * 100



df['percentage period'] = df['text'].apply(percentage_period)

df.head()
plt.hist(df[df['is spam'] == 1]['percentage period'], bins=10, range=(0, 20), rwidth=0.8)

plt.xlabel('Percentage period')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage period in spam messages.')

plt.show()
plt.hist(df[df['is spam'] == 0]['percentage period'], bins=10, range=(0, 20), rwidth=0.8)

plt.xlabel('Percentage period')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage period in non-spam messages.')

plt.show()
df['percentage period'] = df['percentage period'].apply(lambda x: 0 if x < 5 else 1)

df.head(2)
df.groupby(['is spam']).mean()
def percentage_caps(text):

    counter = 0

    for i in text:

        if i.isupper():

            counter += 1

    return (counter / len(text)) * 100



df['percentage caps'] = df['text'].apply(percentage_caps)

df.head()
plt.hist(df[df['is spam'] == 1]['percentage caps'], bins=10, range=(0, 40), rwidth=0.8)

plt.xlabel('Percentage caps')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage caps in spam messages.')

plt.show()
plt.hist(df[df['is spam'] == 0]['percentage caps'], bins=10, range=(0, 40), rwidth=0.8)

plt.xlabel('Percentage caps')

plt.ylabel('Number of messages')

plt.title('Distribution of percentage caps in non-spam messages.')

plt.show()
df['percentage caps'] = df['percentage caps'].apply(lambda x: 0 if x < 5 else 1)

df.head(2)
df.groupby(['is spam']).mean()
def contains_emoji(text):

    return int(':)' in text or ':(' in text or ':-)' in text or ':=D' in text or ':D' in text or ':P' in text)



df['contains emoji'] = df['text'].apply(contains_emoji)

df.head(2)
df.groupby(['is spam']).mean()
df['length'] = df['text'].apply(len)

df.head()
plt.hist(df[df['is spam'] == 1]['length'], bins=10, range=(0, 200), rwidth=0.8)

plt.xlabel('Length of message')

plt.ylabel('Number of messages')

plt.title('Distribution of length of spam messages.')

plt.show()
plt.hist(df[df['is spam'] == 0]['length'], bins=10, range=(0, 200), rwidth=0.8)

plt.xlabel('Length of message')

plt.ylabel('Number of messages')

plt.title('Distribution of length of non-spam messages.')

plt.show()
df['length'] = pd.cut(df['length'], bins=[0, 100, 150, 200, 1000], labels=['<100','100-150', '150-200', '>200'])

df.head()
df = pd.concat([df, pd.get_dummies(df['length'])], axis=1)

df = df.drop(['length'], axis=1)

df.head(2)
from nltk.corpus import stopwords



def isalpha(word):

    wrod = word.replace('.', '')

    return word.isalpha()



def clean_sms(text):

    text = text.lower()

    return (' '.join(filter(lambda x: isalpha(x) and x not in stopwords.words("english"), text.split()))).replace('.', '').split()



from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(strip_accents='ascii', min_df=5, analyzer=clean_sms)

df = pd.concat([df, pd.DataFrame(cv.fit_transform(df['text']).todense(), columns=cv.get_feature_names(), index=np.arange(1, cv.fit_transform(df['text']).todense().shape[0] + 1)).drop(['text'], axis=1)], axis=1)

df.head(2)
X = df.drop(['is spam', 'text'], axis = 1)

y = df['is spam']



print("X.shape:", X.shape)

print("y.shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.6)

X_cv, X_test, y_cv, y_test = train_test_split(X_other, y_other, test_size=0.5)



print("Train dataset size: ", X_train.shape[0])

print("CV size: ", X_cv.shape[0])

print("Test size: ", X_test.shape[0])
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf = clf.fit(X_train, y_train)
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

print(classification_report(y_train, clf.predict(X_train), target_names=['ham', 'spam']))

print(classification_report(y_cv, clf.predict(X_cv), target_names=['ham', 'spam']))

print('F1 score (Train data):', f1_score(y_train, clf.predict(X_train)))

print('F1 score (CV data):', f1_score(y_cv, clf.predict(X_cv)))
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf = clf.fit(X_train, y_train)
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

print(classification_report(y_train, clf.predict(X_train), target_names=['ham', 'spam']))

print(classification_report(y_cv, clf.predict(X_cv), target_names=['ham', 'spam']))

print('F1 score (Train data):', f1_score(y_train, clf.predict(X_train)))

print('F1 score (CV data):', f1_score(y_cv, clf.predict(X_cv)))
print(classification_report(y_test, clf.predict(X_test), target_names=['ham', 'spam']))

print('F1 score (Test data):', f1_score(y_test, clf.predict(X_test)))
y_actual = y_test

y_predicted = clf.predict(X_test)
true_positives = X_test[(y_actual == 1) & (y_predicted == 1)]

true_negatives = X_test[(y_actual == 0) & (y_predicted == 0)]

false_positives = X_test[(y_actual == 0) & (y_predicted == 1)]

false_negatives = X_test[(y_actual == 1) & (y_predicted == 0)]
print('False positives')

df.loc[list(false_positives.index)]['text']
print('False negatives')

df.loc[list(false_negatives.index)]['text']