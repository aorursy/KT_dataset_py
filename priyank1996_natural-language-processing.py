import pandas as pd

import regex as re

import nltk
df = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

df.head()
from nltk.corpus import stopwords #remove words that are non significant for ML for e.g: This

from nltk.stem.porter import PorterStemmer #bring everything to one tense
review = re.sub('[^a-zA-Z]',' ', df.Review[0])

review = review.lower()

review
review = review.split()

review
ps = PorterStemmer()

review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

review
review = ' '.join(review)

review
corpus = []

for i in range(len(df)):

    review = re.sub('[^a-zA-Z]',' ', df.Review[i])

    review = review.lower()

    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
df['Filter_Review'] = corpus

df.head()
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(df.Filter_Review).toarray()
y = df.Liked.values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(confusion_matrix(y_test,y_pred), annot= True)

plt.xlabel('y_test')

plt.ylabel('y_predicted')

plt.title('Confusion Matrix using Naive Bayes')
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

cv = []

model_rf= RandomForestClassifier(n_estimators=20)

model_dt = DecisionTreeClassifier()

models = [model, model_rf, model_dt]

for i in models:

    cv.append(cross_val_score(i,X,y, cv = 5).mean())

print('''Naive Bayes: {},

Random Forest: {},

Decision Tree: {}'''.format(cv[0],cv[1],cv[2]))