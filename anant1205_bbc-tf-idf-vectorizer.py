# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data = pd.read_csv("../input/ai-academy-intermediate-class-competition-1/BBC News Train.csv")
data = data[["Text", "Category"]]
data
# Any results you write to the current directory are saved as output.
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(data["Text"])
print(len(vectorizer.get_feature_names()))
print(X.shape)
data["category_id"]=data["Category"].factorize()[0]
data = data[["Text", "category_id", "Category"]]
data
category_id_data = data[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_data.values)
id_to_category = dict(category_id_data[['category_id', 'Category']].values)
category_id_data
id_to_category
data.sample(5, random_state=0)
data.groupby('Category').category_id.count()
data.groupby('Category').category_id.count().plot.bar(ylim=0)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(data.Text).toarray()
print(features)
labels = data.category_id
print(labels)
features.shape
from sklearn.feature_selection import chi2

N = 5
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
from sklearn.manifold import TSNE
import matplotlib as plt
SAMPLE_SIZE = int(len(features) * 0.3)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
for category, category_id in sorted(category_to_id.items()):
    points = projected_features[(labels[indices] == category_id).values]
    plt.pyplot.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
plt.pyplot.title("tf-idf feature vector for each article, projected on 2 dimensions.",
          fontdict=dict(fontsize=15))
plt.pyplot.legend()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=500, max_depth=4),
    MultinomialNB(),
    LogisticRegression(random_state=4),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
print(cv_df)
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  print(accuracies)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
print(entries)
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.20, random_state=0)
for model in models:
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
model = models[2]
model.fit(features, labels)
model.coef_
test_data = pd.read_csv("../input/bbc-test-3/BBC News Test.csv")
test_data
test_data.Text.tolist()
test_features = tfidf.transform(test_data.Text.tolist())
Y_pred = model.predict(test_features)
Y_pred
submission = []
for pred in Y_pred:
    submission.append(id_to_category[pred])
submission
submission = pd.DataFrame({
    "ArticleId": test_data["ArticleId"],
    "Category": submission
})
submission
submission.to_csv('submission.csv', index=False)
ls
