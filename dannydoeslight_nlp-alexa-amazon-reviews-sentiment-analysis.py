import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter='\t')

df.head()
df.info()
np.bincount(df['feedback'])

# looks like a pretty imbalanced classes
df.describe()

# looks like the positive feedback: 1 is the predominant class, mostly around a rating of 5
import matplotlib.pyplot as plt

import seaborn as sns



sns.countplot("rating", hue="feedback", data=df)

# it is quite expected that a rating under 3 to be a negative one.
plt.figure(figsize=(40,8))

sns.countplot("variation", hue="feedback", data=df)

# looks like Charcoal Fabric and Black Dot are the most bought / rated
plt.figure(figsize=(60,8))

sns.boxplot('date', 'rating', hue="feedback", data=df)

# During May and June mostly positive feedback, while starting with July predominantly negative.
df.isna().sum()

# now we can proceed with NLP.
# will show some love to sklearn, a library not that much used for NLP

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer() # will try stop words and apply other improvements later

vect.fit(df.verified_reviews)
print(f"Vocabulary size: {len(vect.vocabulary_)}") 

print(f"Vocabulary content:\n {vect.vocabulary_}")
X = vect.transform(df.verified_reviews) 

y = df.feedback
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                test_size=0.33, random_state=42, stratify=y) # stratify: class in-balance in the data would be preserved.
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(solver='lbfgs'), X_train, y_train, cv=5) 

print(f"Mean cross-validation accuracy: {np.mean(scores):.2f}")
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(LogisticRegression(solver='lbfgs'), param_grid, cv=5) 

grid.fit(X_train, y_train)

print(f"Best cross-validation score: {grid.best_score_:.2f}") 

print(f"Best parameters: {grid.best_params_}")
print(f"Test score is: {grid.score(X_test, y_test):.2f}")
y_pred = grid.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

cr = classification_report(y_test, y_pred)

print(cr)
LR = LogisticRegression(C=1, random_state=9, solver='lbfgs')

LR.fit(X_train, y_train)



features = vect.get_feature_names()

feature_importance = abs(LR.coef_[0])

#feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)



plt.figure(figsize=(12,600))

#sns.set(rc={'figure.figsize':(6,600)})

plt.title('Feature Importances')

plt.barh(range(len(indices)), feature_importance[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



# positive words as: love, great and 

# negative words as: poor, awful have top importance
# vect.get_params().keys()
# just curious if removing the stopwords and 

# cutting on uninformative features and

# adding pair of tokens(which increases the features no) improves the score:



vect = CountVectorizer(min_df=5, ngram_range=(2, 2), stop_words="english").fit(df.verified_reviews) 



X = vect.transform(df.verified_reviews)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                test_size=0.33, random_state=42, stratify=y)

grid = GridSearchCV(LogisticRegression(solver='lbfgs'), param_grid, cv=5) 

grid.fit(X_train, y_train)

print(f"Best cross-validation score: {grid.best_score_:.2f}") 

print(f"Best parameters: {grid.best_params_}")
# rescaling with tf-idf (instead of dropping features):



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

tfidf = TfidfVectorizer(min_df=5, norm=None)

LR = LogisticRegression(solver='lbfgs')

pipe = Pipeline([('tfidf', tfidf),('LR', LR)])

param_grid = {'LR__C': [0.001, 0.01, 0.1, 1, 10],

             'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)]}

grid = GridSearchCV(pipe, param_grid, cv=5)
X = df.verified_reviews

y = df.feedback
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                test_size=0.33, random_state=42, stratify=y)

grid.fit(X_train, y_train)

print(f"Best cross-validation score: {grid.best_score_:.2f}")

print(f"Best parameters: {grid.best_params_}")
print(grid.score(X_test, y_test))

y_pred = grid.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

cr = classification_report(y_test, y_pred)

print(cr)
import spacy

T = X[2]

nlp = spacy.load('en_core_web_sm')

doc = nlp(T)



for token in doc:

    token_text = token.text

    token_pos = token.pos_

    token_dep = token.dep_

    print(f'{token_text:<12} {token_pos:<10} {token_dep:<10}')

for ent in doc.ents:

    print("\n", ent.text, ent.label_)
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

pattern = [ {'LEMMA': 'love', 'POS': 'VERB'},

            { 'POS': 'DET', 'OP': '?'},

            {'POS': 'NOUN'} ]

matcher.add('Positive Reviews incl Love', None, pattern)

for doc in nlp.pipe(X):

    #doc = nlp(X[i]) # unsing nlp.pipe for more efficient text processing

    matches = matcher(doc)

    for match_id, start, end in matches:

        matched_span = doc[start:end]

        print(matched_span.text)