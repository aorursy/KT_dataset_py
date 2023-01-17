import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
def get_categories(df):
    return df['category'].unique()
df = pd.read_csv('../input/bbc-articles-cleaned/tfidf_dataset.csv')
df.head()
X_data = df[['text']].to_numpy().reshape(-1)
Y_data = df[['category']].to_numpy().reshape(-1)
n_texts = len(X_data)
print('Texts in dataset: %d' % n_texts)

n_categories = len(get_categories(df))
print('Number of categories: %d' % n_categories)

print('Loading train dataset...')
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)

print('Done!')
clf = Pipeline([('vect', CountVectorizer(strip_accents=None, lowercase=False)),
                ('svm', SGDClassifier(alpha=0.001,
                                      loss='log',
                                      penalty='l2',
                                      random_state=42,
                                      tol=0.001))])
clf.fit(X_train, Y_train)
def plot_confusion_matrix(X_test, Y_test, model):
    Y_pred = model.predict(X_test)

    con_mat = confusion_matrix(Y_test, Y_pred)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    label_names = list(range(len(con_mat_norm)))
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=label_names, 
                              columns=label_names)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
Y_pred = clf.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(Y_pred, Y_test))

print('Classification report:')
print(classification_report(Y_test, Y_pred))
plot_confusion_matrix(X_test, Y_test, clf)