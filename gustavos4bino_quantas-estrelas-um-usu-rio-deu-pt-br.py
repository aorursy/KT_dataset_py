import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

%matplotlib inline
path = '../input/'
reviews = pd.read_csv(os.path.join(path, 'olist_order_reviews_dataset.csv'))
reviews.head()
orders = pd.read_csv(os.path.join(path, 'olist_orders_dataset.csv'))
orders.head()
columns_to_stay = ['order_id','order_delivered_carrier_date', 'order_delivered_customer_date']
orders = orders[columns_to_stay]
orders.head()
columns_to_remove = ['review_id']
reviews = reviews.drop(columns_to_remove, axis=1)
reviews.head()
reviews.dtypes
reviews['review_creation_date'] = reviews['review_creation_date'].apply(pd.to_datetime)
reviews['review_answer_timestamp'] = reviews['review_answer_timestamp'].apply(pd.to_datetime)
reviews['review_score'].value_counts()
print(
    reviews['review_comment_title'].isna().sum(),
    reviews['review_comment_message'].isna().sum()
)
def nan_to_empty_string(row):
    if row is np.NaN:
        row = ''.strip()
    else:
        pass
    return row

reviews['review_comment_title'] = reviews['review_comment_title'].apply(nan_to_empty_string)
reviews['review_comment_message'] = reviews['review_comment_message'].apply(nan_to_empty_string)
orders.dtypes
orders['order_delivered_carrier_date'] = orders['order_delivered_carrier_date'].apply(pd.to_datetime)
orders['order_delivered_customer_date'] = orders['order_delivered_customer_date'].apply(pd.to_datetime)
orders['Time waiting'] = orders['order_delivered_customer_date'] - orders['order_delivered_carrier_date']
orders['Time waiting'] = orders['Time waiting'].apply(lambda x: x.total_seconds())

orders = orders.drop(['order_delivered_customer_date', 'order_delivered_carrier_date'], axis=1)
orders['Time waiting'].isna().sum()
reviews['review timestamp'] = reviews['review_answer_timestamp'] - reviews['review_creation_date']
reviews['review timestamp'] = reviews['review timestamp'].apply(lambda x: x.total_seconds())

reviews = reviews.drop(['review_answer_timestamp', 'review_creation_date'], axis=1)
reviews['review timestamp'].isna().sum()
def join_text(row):
    return '{} {}'.format(
        row['review_comment_title'],
        row['review_comment_message'])

reviews['review'] = reviews[['review_comment_title','review_comment_message']].apply(join_text, axis=1)
reviews = reviews.drop(['review_comment_title','review_comment_message'], axis=1)
orders.head()
print(
    orders.shape,
    reviews.shape
)
review_merge = reviews.merge(
    orders,
    how='inner',
    on='order_id'
)

review_merge.head()
sem_data_entrega = review_merge[review_merge['Time waiting'].isna()]
sem_data_entrega.head()
sns.countplot(sem_data_entrega['review_score'])
media_time_waiting = review_merge['Time waiting'].mean()
review_merge['Time waiting'].fillna(media_time_waiting, inplace=True)
review_merge['Time waiting'].isna().sum()
review_merge.head()
review_merge['review len'] = review_merge['review'].apply(lambda x: len(x.strip()))
review_merge.head()
review_merge.drop('order_id', axis=1, inplace=True)
review_merge['review len'].max()
sns.lineplot(y='review len',x='review_score', data=review_merge)
review_merge.corr()
sns.countplot(review_merge['review_score'])
sns.distplot(review_merge['Time waiting'])
sns.lmplot(x='Time waiting', y='review_score',data= review_merge)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer, Normalizer
from nltk.corpus import stopwords
review_merge.head()
def _get_numeric(data):
    return data[['Time waiting', 'review len']]
    
def _get_text(data):
    return data[['review']].apply(lambda x: ''.join(x), axis=1)


get_numeric = FunctionTransformer(_get_numeric, validate=False)
get_text = FunctionTransformer(_get_text, validate=False)
get_numeric.transform(review_merge.head())
get_text.transform(review_merge.head())
from sklearn.naive_bayes import MultinomialNB
TOKEN_PATTERN = r'\w+'

numeric_pipeline = Pipeline([
    ('selector', get_numeric),
    ('scaler', Normalizer())
])

text_pipeline = Pipeline([
    ('selector', get_text),
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        token_pattern=TOKEN_PATTERN,
        stop_words=stopwords.words('portuguese'))),
    ('dim_reduction', SelectKBest(chi2, k=300))
])

final_pipe = Pipeline([
    ('feature_union', FeatureUnion(transformer_list=[
        ('numeric', numeric_pipeline),
        ('text', text_pipeline)
    ])),
    ('sgd', SGDClassifier(max_iter=100, tol=0.001))
])
NUMERO_OBS = 100000
X = review_merge.drop('review_score', axis=1)[:NUMERO_OBS]
y = review_merge['review_score'].tolist()[:NUMERO_OBS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    shuffle=True,
    test_size=0.3
)
final_pipe.fit(X_train, y_train)
predicted = final_pipe.predict(X_test)
print(classification_report(y_test, predicted))
print("Acurácia de treino: ", final_pipe.score(X_train, y_train))
print("Acurácia de teste: ", final_pipe.score(X_test, y_test))