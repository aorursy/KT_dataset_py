import numpy as np

import pandas as pd

import warnings

import os

from matplotlib import pyplot as plt

from timeit import default_timer as timer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from sklearn.svm import LinearSVC

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.neighbors import KNeighborsClassifier
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
np.random.seed(42)
warnings.simplefilter(action='ignore', category=FutureWarning)
df_train = pd.read_csv('/kaggle/input/bigdata2020classification/train.csv/train.csv')

df_test = pd.read_csv('/kaggle/input/bigdata2020classification/test_without_labels.csv//test_without_labels.csv')
df_train.head()
df_train = df_train.drop(['Id'], axis=1)
len_start = len(df_train)

df_train.info()
df_train.drop_duplicates(subset='Content', keep='last', inplace=True)
len_end = len(df_train)
# Display number of duplicates in the dataset



print('{} duplicate articles were removed from the dataset'

      .format(len_start - len_end))
# Relative frequency of labels



rel_freq = df_train.groupby('Label').Title.count() / len(df_train)

print(rel_freq)
fig, ax = plt.subplots(figsize=(6,5))

rel_freq.plot.bar(ax=ax)

ax.set_title('Relative Frequencies of the Class Labels')

ax.set_xlabel('')

fig.tight_layout()

plt.show()
# Concatenate the titles and the contents



df_train['Text'] = df_train['Title'] + ' ' + df_train['Content']

X = df_train['Text'].to_dense().values

y = df_train['Label'].to_dense().values
df_train['Text'][259]
def get_metrics(y_true, y_pred, metrics):

    metrics[0] += accuracy_score(y_true, y_pred)

    metrics[1] += precision_score(y_true, y_pred, average='macro')

    metrics[2] += recall_score(y_true, y_pred, average='macro')

    metrics[3] += f1_score(y_true, y_pred, average='macro')

    return metrics
def evaluate_classifier(clf, kfold, X, y, vectorizer):

    metrics = np.zeros(4)

    start = timer()

    for train, cv in kfold.split(X, y):

        X_train, X_cv = X[train], X[cv]

        y_train, y_cv = y[train], y[cv]

        X_train_gen = [x for x in X_train]

        vectorizer.fit(X_train_gen)

        X_train_vec = vectorizer.transform(X_train_gen)

        clf.fit(X_train_vec, y_train)

        X_cv_gen = [x for x in X_cv]

        X_cv_vec = vectorizer.transform(X_cv_gen)

        y_pred = clf.predict(X_cv_vec)

        metrics = get_metrics(y_cv, y_pred, metrics)

    dt = timer() - start

    metrics = metrics * 100 / 5

    print('Evaluation of classifier finished in {:.2f} s \n'

          'Average accuracy: {:.2f} % \n'

          'Average precision: {:.2f} % \n'

          'Average recall: {:.2f} % \n'

          'Average F-Measure: {:.2f} % \n'

          .format(dt, metrics[0], metrics[1],

                  metrics[2], metrics[3]))
# 5-Fold Cross-Validation



kf = KFold(n_splits=5, shuffle=True, random_state=56)



# Stop Words and TF-IDF Vectorizer



stop_words = ENGLISH_STOP_WORDS

tfidf = TfidfVectorizer(stop_words=stop_words, min_df=3,

                        max_df=0.5, ngram_range=(1, 2))



# Classifiers 



svm = LinearSVC(tol=1e-05, max_iter=1500)

ridge = RidgeClassifier(alpha=0.8, tol=1e-05)

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
# Divide the dataset into a train/cross-validation set and a test set



X_train_cv, X_test, y_train_cv, y_test = train_test_split(X, y, test_size=0.2, 

                                                          random_state=56)
# SVM classifier



evaluate_classifier(svm, kf, X_train_cv, y_train_cv, tfidf)
# Ridge classifier



evaluate_classifier(ridge, kf, X_train_cv, y_train_cv, tfidf)
# kNN classifier



evaluate_classifier(knn, kf, X_train_cv, y_train_cv, tfidf)
# Voting ensemble of 3 classifiers



boost = VotingClassifier(estimators=[

                        ('svc', svm), ('ridge', ridge), ('knn', knn)],

                        voting='hard', n_jobs=-1)
# Evaluate the voting classifier on the test set



start = timer()

metrics = np.zeros(4)

X_train_gen = [x for x in X_train_cv]

tfidf.fit(X_train_gen)

X_train_vec = tfidf.transform(X_train_gen)

boost.fit(X_train_vec, y_train_cv)

X_test_gen = [x for x in X_test]

X_test_vec = tfidf.transform(X_test_gen)

y_pred = boost.predict(X_test_vec)

metrics = get_metrics(y_test, y_pred, metrics)

dt = timer() - start

metrics = metrics * 100

print('Evaluation of voting classifier on the test set finished in {:.2f} s . \n'

      'Average accuracy: {:.2f} % \n'

      'Average precision: {:.2f} % \n'

      'Average recall: {:.2f} % \n'

      'Average F-Measure: {:.2f} % \n'

      .format(dt, metrics[0], metrics[1],

              metrics[2], metrics[3]))
# Concatenate the titles and the contents



df_test['Text'] = df_test['Title'] + ' ' + df_test['Content']

X_final = df_test['Text'].to_dense().values
# Make predictions on the unlabeled data



X_train_gen = [x for x in X]

tfidf.fit(X_train_gen)

X_train_vec = tfidf.transform(X_train_gen)

boost.fit(X_train_vec, y)

X_test_gen = [x for x in X_final]

X_test_vec = tfidf.transform(X_test_gen)

y_pred = boost.predict(X_test_vec)
df_results = pd.DataFrame({'Id':df_test['Id'], 'Predicted':y_pred})
df_results.to_csv('testSet_categories.csv',index=False, header=True)