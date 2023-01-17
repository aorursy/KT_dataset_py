import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
data = pd.read_csv('../input/spam.csv', encoding='latin-1')
data.head()
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.head()
data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
data.head()
data.count()  # check if the two columns have the same number of values
data['label'].apply(len).min()  # check for empty labels
data['message'].apply(len).min()  # check for empty messages
set(data['label'])  # check if there are labels other than 'ham' and 'spam'
data['label'] = data['label'].apply(lambda label: 0 if label == 'ham' else 1)
data.head()
data.describe()
# Empirical distribution of the labels
print('Percentage of spams: {0}%'.format(round(100 * data['label'].sum() / len(data['label']), 2)))
plt.hist(data['label'], bins=3, weights=np.ones(len(data['label'])) / len(data['label']))
plt.xlabel('label')
plt.ylabel('Empirical PDF')
plt.title('Empirical distribution of the labels')
# extract spams and hams
spams = data['message'].iloc[(data['label'] == 1).values]
hams = data['message'].iloc[(data['label'] == 0).values]
print(spams[:10])
print(hams[:10])
# message length
data['message_length'] = data['message'].apply(lambda message: len(message))
data.head()
plt.hist(data['message_length'], bins=50, weights=np.ones(len(data))/len(data))
plt.xlabel('Message length')
plt.ylabel('Empirical PDF')
plt.title('Messages lengths')
plt.hist(spams.apply(lambda x: len(x)),
         bins=50,
         weights=np.ones(len(spams)) / len(spams),
         facecolor='r',
         label='Spams')
plt.hist(hams.apply(lambda x: len(x)),
         bins=50,
         weights=np.ones(len(hams)) / len(hams),
         facecolor='g',
         alpha=0.8,
         label='Hams')
plt.xlabel('Message lenght')
plt.ylabel('Empirical PDF')
plt.title('Spam/ham messages lengths')
plt.legend()
# most common words in spam and ham
spam_tokens = []
for spam in spams:
    spam_tokens += nltk.tokenize.word_tokenize(spam)
ham_tokens = []
for ham in hams:
    ham_tokens += nltk.tokenize.word_tokenize(ham)
print(spam_tokens[:10])
print(ham_tokens[:10])
# remove stop words and puncuation from tokens
stop_words = ['.', 'to', '!', ',', 'a', '&', 
              'or', 'the', '?', ':', 'is', 'for',
              'and', 'from', 'on', '...', 'in', ';',
              'that', 'of']
for tokens in [spam_tokens, ham_tokens]:
    for stop_word in stop_words:
        try:
            while True:
                tokens.remove(stop_word)
        except ValueError:  # all occurrences of the stop word have been removed
            pass
most_common_tokens_in_spams = Counter(spam_tokens).most_common(20)
most_common_tokens_in_hams = Counter(ham_tokens).most_common(20)
print(most_common_tokens_in_spams)
print(most_common_tokens_in_hams)
data, test_data = train_test_split(data, test_size=0.3)
print('Train-valid data length: {0}'.format(len(data)))
print('Test data length: {0}'.format(len(test_data)))
binary_vectorizer = CountVectorizer(binary=True)
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
def feature_extraction(df, test=False):
    if not test:
        tfidf_vectorizer.fit(df['message'])
    
    X = np.array(tfidf_vectorizer.transform(df['message']).todense())
    return X
train_df, valid_df = train_test_split(data, test_size=0.3)

X_train = feature_extraction(train_df)
y_train = train_df['label'].values

X_valid = feature_extraction(valid_df, test=True)
y_valid = valid_df['label'].values
clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    'svm2': SVC(kernel='rbf'),
    'svm3': SVC(kernel='sigmoid'),
    'mlp1': MLPClassifier(),
    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}
f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1_scores[clf_name] = f1_score(y_pred, y_valid)
f1_scores
clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm': SVC(kernel='linear'),
    'mlp': MLPClassifier(),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}
def feature_extraction(df, test=False):
    if not test:
        count_vectorizer.fit(df['message'])
    
    X = np.array(count_vectorizer.transform(df['message']).todense())
    return X
X_train = feature_extraction(train_df)
X_valid = feature_extraction(valid_df, test=True)
f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1_scores[clf_name] = f1_score(y_pred, y_valid)
f1_scores
def feature_extraction(df, test=False):
    if not test:
        binary_vectorizer.fit(df['message'])
    
    X = np.array(binary_vectorizer.transform(df['message']).todense())
    return X
X_train = feature_extraction(train_df)
X_valid = feature_extraction(valid_df, test=True)
f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1_scores[clf_name] = f1_score(y_pred, y_valid)
f1_scores
clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)
f1_score(y_pred, y_valid)
def feature_extraction(df, test=False):
    if not test:
        count_vectorizer.fit(df['message'])
    
    X = np.array(count_vectorizer.transform(df['message']).todense())
    X = np.concatenate((X, df['message_length'].values.reshape(-1, 1)), axis=1)
    return X
X_train = feature_extraction(train_df)
X_valid = feature_extraction(valid_df, test=True)
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)
f1_score(y_pred, y_valid)
def feature_extraction(df, test=False):
    if not test:
        tfidf_vectorizer.fit(df['message'])
    
    X = np.array(tfidf_vectorizer.transform(df['message']).todense())
    return X
X_train = feature_extraction(train_df)
X_valid = feature_extraction(valid_df, test=True)
alpha_values = [i * 0.1 for i in range(11)]
max_f1_score = float('-inf')
best_alpha = None
for alpha in alpha_values:
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    current_f1_score = f1_score(y_pred, y_valid)
    if current_f1_score > max_f1_score:
        max_f1_score = current_f1_score
        best_alpha = alpha
print('Best alpha: {0}'.format(best_alpha))
print('Best f1-score: {0}'.format(max_f1_score))
clf = MultinomialNB(alpha=0.1)
X_train = feature_extraction(data)
y_train = data['label'].values

X_test = feature_extraction(test_data, test=True)
y_test = test_data['label'].values
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)