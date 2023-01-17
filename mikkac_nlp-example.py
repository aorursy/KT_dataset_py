import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/reviews/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

df.head()
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer() # find root of the word- i.e loved -> love



corpus = []

for i in range(0, df.shape[0]):

  review = re.sub('[^a-zA-Z]', ' ', df['Review'][i]).lower().split()

  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  review = ' '.join(review)

  corpus.append(review)



corpus[:5]
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1000)

X = cv.fit_transform(corpus).toarray()

y = df.loc[:, 'Liked'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def score_model(model, X_train, X_test, y_train, y_test):

  model = model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  ac = accuracy_score(y_test, y_pred)

  pc = precision_score(y_test, y_pred)

  rs = recall_score(y_test, y_pred)

  fs = f1_score(y_test, y_pred)

  return [ac, pc, rs, fs]
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression



models = []



models.append(RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=42))

models.append(KNeighborsClassifier(n_neighbors=7, leaf_size=20))

models.append(GaussianNB())

models.append(SVC(C=5.0, kernel='rbf', random_state=42))

models.append(LogisticRegression(random_state=42))



models_names = []

cols = ['Accuracy', 'Precision', 'Recall', 'F1']

scores = pd.DataFrame(columns=cols)

scores

for model in models:

  models_names.append(model.__class__.__name__)

  scores_row =  score_model(model, X_train, X_test, y_train, y_test)

  scores = pd.concat([scores, pd.DataFrame([scores_row], columns=cols)], axis=0)



models_names = pd.DataFrame(models_names, columns = ['Model'])

final_scores = models_names.join(scores.reset_index(), sort=False).reset_index().drop(['index', 'level_0'], axis=1).set_index('Model')

final_scores