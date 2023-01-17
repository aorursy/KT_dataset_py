import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
true = pd.read_csv('./data/true.csv')
true.head()
true.isnull().sum()
true['title_word_count'] = true['title'].map(lambda x: len(x.split(' ')))
plt.hist(true['title_word_count'], bins = 15, color = 'g')
plt.title('Distribution of Title Word Counts for Real News')
plt.xlabel('Word Count')
plt.ylabel('Number of Titles');
fake = pd.read_csv('./data/fake.csv')
fake.head()
fake.isnull().sum()
fake['title_word_count'] = fake['title'].map(lambda x: len(x.split(' ')))
plt.hist(fake['title_word_count'], bins = 15, color = 'salmon')
plt.title('Distribution of Title Word Counts for Fake News')
plt.xlabel('Word Count')
plt.ylabel('Number of Titles');
def most_freq(df):
    cvec = CountVectorizer(stop_words = 'english')
    cvec.fit(df['title'])
    X_train = cvec.transform(df['title'])
    X_train_df = pd.DataFrame(X_train.toarray(),
                              columns=cvec.get_feature_names())
    top_words = {}
    for i in X_train_df.columns:
        top_words[i] =  X_train_df[i].sum()
    return pd.DataFrame(sorted(top_words.items(), key = lambda x: x[1], reverse = True)).head(10)
common_true = most_freq(true)
common_true
common_fake = most_freq(fake)
# code inspired by 4.05 classification metrics

plt.figure(figsize = (10, 7))

plt.bar(x = common_true[0],
        height = common_true[1],
        color = 'g',
        alpha = 0.6,
        label = 'Real news')
plt.bar(x = common_fake[0],
        height = common_fake[1],
        color = 'salmon',
        alpha = 0.6,
        label = 'Fake news')

plt.xticks(rotation=45)
plt.ylabel('Word Count')
plt.xlabel('Words')
plt.title('Common Words Used in Real and Fake News', fontsize=18)

plt.legend(fontsize=14);
true['category'] = 1
fake['category'] = 0
df = pd.concat([true, fake])
df.shape
df = df.loc[df['date']!= 'https://100percentfedup.com/served-roy-moore-vietnamletter-veteran-sets-record-straight-honorable-decent-respectable-patriotic-commander-soldier/',]
df = df.loc[df['date']!= 'https://100percentfedup.com/video-hillary-asked-about-trump-i-just-want-to-eat-some-pie/']
df = df.loc[df['date']!= 'https://100percentfedup.com/12-yr-old-black-conservative-whose-video-to-obama-went-viral-do-you-really-love-america-receives-death-threats-from-left/']
df = df.loc[df['date']!= 'https://fedup.wpengine.com/wp-content/uploads/2015/04/hillarystreetart.jpg']
df = df.loc[df['date']!= 'https://fedup.wpengine.com/wp-content/uploads/2015/04/entitled.jpg']
# Dropped a row with a 'date' url
df.drop([18933], inplace=True)
# Converted 'date' to a datetime pandas format
df['date'] = pd.to_datetime(df['date'])
# Created another column for weekday
df['weekday'] = df['date'].dt.weekday
X = df['title']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 42)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
y_test.value_counts(normalize=True)
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def porter(text):
    return(stemmer.stem(w) for w in analyzer(text))
pipe = Pipeline([
    ('cvec', CountVectorizer(analyzer=porter, stop_words='english')),
    ('logreg', LogisticRegression(max_iter=1000, solver='liblinear'))
])
pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)
pipe.score(X_test, y_test)
pipe = Pipeline([
    ('cvec', CountVectorizer(stop_words='english')),
    ('logreg', LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train);
pipe.score(X_train, y_train)
pipe.score(X_test, y_test)
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression(solver = 'liblinear', random_state=42))
])
pipe_params = {
    'tfidf__ngram_range': [(1,2)],
    'tfidf__stop_words': ['english'],    
    'logreg__penalty': [ 'l2'],
    'logreg__C': [ 10],
    'logreg__max_iter' : [ 1000]
    
}

gs = GridSearchCV(pipe,
                  param_grid = pipe_params,
                  cv=5,
                  scoring = 'accuracy',
                  verbose = 1)

gs.fit(X_train, y_train)

print(f'Best cross validation score: {gs.best_score_}')
print(f'Best parameters to use: {gs.best_params_}')
print(f'Testing score: {gs.score(X_test, y_test)}')
stemmer = PorterStemmer()
analyzer = TfidfVectorizer().build_analyzer()
def porter(text):
    return(stemmer.stem(w) for w in analyzer(text))
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer=porter)),
    ('logreg', LogisticRegression(solver = 'liblinear'))
])
pipe_params = {
    'tfidf__stop_words': ['english', None],
    'tfidf__max_features': [12_000],
    'tfidf__ngram_range': [(1, 2)],
    'logreg__penalty': ['l2'],
    'logreg__C': [15],
    'logreg__max_iter' : [1000]
    
}

gs = GridSearchCV(pipe,
                  param_grid = pipe_params,
                  cv=5,
                  scoring = 'accuracy',
                  verbose = 1)

gs.fit(X_train, y_train)

print(f'Best cross validation score: {gs.best_score_}')
print(f'Best parameters to use: {gs.best_params_}')
print(f'Testing score: {gs.score(X_test, y_test)}')
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer=porter)),
    ('rf', RandomForestClassifier(random_state = 42))
])
params = {
    'rf__n_estimators': [100],
    'rf__max_depth': [None, 1, 2],
    'rf__max_features': ['auto', 'log2']
}

gs = GridSearchCV(pipe,
                  param_grid=params,
                  cv=2,
                  scoring='accuracy',
                  verbose=1)

gs.fit(X_train, y_train)

print(f'Best cross validation score: {gs.best_score_}')
print(f'Best parameters to use: {gs.best_params_}')
print(f'Testing score: {gs.score(X_test, y_test)}')
