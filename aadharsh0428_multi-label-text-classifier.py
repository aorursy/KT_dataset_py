import pandas as pd

data = pd.read_csv("../input/Blooms Dataset - Sheet1.csv")
data.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features =  tfidf.fit_transform(data.Question).toarray()

labels = data.Class

features.shape
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Class'], random_state = 0)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

models = [

    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),

    LinearSVC(),

    MultinomialNB(),

    LogisticRegression(random_state=0),

]

CV = 5

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

for model in models:

  model_name = model.__class__.__name__

  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

  for fold_idx, accuracy in enumerate(accuracies):

    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)

sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)

cv_df.groupby('model_name').accuracy.mean()