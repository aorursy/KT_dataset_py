from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor 

import os 
import gensim
import pandas as pd

import itertools

import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import scikitplot

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

from fastFM import sgd
lemmatizer = WordNetLemmatizer()


def stem(text):
    return lemmatizer.lemmatize(text)


def map_parallel(f, iterable, **kwargs):
    with ProcessPoolExecutor() as pool:
        result = pool.map(f, iterable, **kwargs)
    return result


def retrieve_articles(start, chunksize=1000):
    return arxiv.query(
        search_query=search_query,
        start=start,
        max_results=chunksize
    )
def vectorize_text(examples_df):

    vectorizer = CountVectorizer(min_df=2)
    features = vectorizer.fit_transform(examples_df['summary'])

    le = LabelEncoder()
    ohe = OneHotEncoder()
    labels = le.fit_transform(valid_example_categories).reshape(-1, 1)
    labels_ohe = ohe.fit_transform(labels).todense()
    vectorized_data = {
        'features': features,
        'labels': labels,
        'labels_onehot' : labels_ohe
    }
    return vectorized_data, (ohe, le)


def extract_keywords(text):
    """
    Use gensim's textrank-based approach
    """
    return gensim.summarization.keywords(
        text=text,
        lemmatize=True,
        split=True
    )
class FMClassifier(sgd.FMClassification):
    """
    Wrapper for fastFM estimator that makes it behave like sklearn ones
    """
    
    def fit(self, X, y, *args):
        y = y.copy()
        y[y == 0] = -1
        return super(FMClassifier, self).fit(X, y, *args)

    def predict_proba(self, X):
        probs = super(FMClassifier, self).predict_proba(X)
        return np.tile(probs, 2).reshape(2, probs.shape[0]).T
    

def predict_ovr(model, X):
    """
    predict as multiclass (standard OVR behaves as predicting multilabel)
    """
    return np.argmax(model.predict_proba(X), 1)
def filter_out_small_categories(df, categories, threshold=200):

    class_counts = categories.value_counts()
    too_small_classes = class_counts[class_counts < threshold].index
    too_small_classes

    valid_example_indices = ~categories.isin(too_small_classes)
    valid_examples = df[valid_example_indices]
    valid_example_categories = categories[valid_example_indices]
    
    return valid_examples, valid_example_categories
def report_classification_confusion_matrix(y, y_pred, label_encoder):

    y_test_pred_label_names = label_encoder.inverse_transform(y_pred)
    y_test_label_names = label_encoder.inverse_transform(y.reshape(-1))

    print(classification_report(y_test_label_names, y_test_pred_label_names))
    
    scikitplot.metrics.plot_confusion_matrix(
        y_test_label_names,
        y_test_pred_label_names,
        hide_zeros=True,
        x_tick_rotation=90
    )
    plt.show()
data_path = '../input/matrix_factorization_arxiv_query_result.json'
articles_df = pd.read_json(data_path)
articles_df.head()
articles_df[['title', 'authors', 'published', 'summary']].head()
articles_df.info()
categories = articles_df['arxiv_primary_category'].apply(itemgetter('term'))

main_categories = categories.apply(lambda s: s.split('.')[0].split('-')[0])
main_categories_counts = main_categories.value_counts(ascending=True)
main_categories_counts.plot.barh()
plt.show()
main_categories_counts[main_categories_counts > 200].plot.barh()
plt.show()
categories.value_counts(ascending=True)[-10:].plot.barh()
plt.show()
%%time

articles_df['summary_keywords'] = list(
    map_parallel(extract_keywords, articles_df['summary'])
)
n_examples = 20 

for __, row in itertools.islice(articles_df.iterrows(), n_examples):
  print(20 * '*')
  print(row['title'])
  print(20 * '*')
  print('keywords:', row['summary_keywords'])
  print()
article_keyword_lengths = articles_df['summary_keywords'].apply(len)
article_keyword_lengths.plot.hist(bins=article_keyword_lengths.max(), title='Number of summary keywords')
valid_examples, valid_example_categories = filter_out_small_categories(articles_df, main_categories)
valid_examples.shape
vectorized_data, (ohe, le) = vectorize_text(valid_examples)
fm = FMClassifier(
    rank=50,
    n_iter=10000,
    step_size=0.0001,
    l2_reg_w=0.01,
    l2_reg_V=0.01
)
fm_multiclass = OneVsRestClassifier(fm)
x_train, x_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    vectorized_data['features'],
    vectorized_data['labels_onehot'],
    vectorized_data['labels'],
    stratify=vectorized_data['labels'],
    test_size=0.2,
    random_state=0
)
%%time

fm_multiclass.fit(x_train, y_train)
y_test_pred = predict_ovr(fm_multiclass, x_test)
print(
    'train score:', accuracy_score(y_train_labels, predict_ovr(fm_multiclass, x_train)), '\n'
    'test score: ', accuracy_score(y_test_labels, y_test_pred)
)
report_classification_confusion_matrix(y_test_labels, y_test_pred, le)