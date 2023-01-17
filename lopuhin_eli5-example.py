import eli5
from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin
def load_data(filename="../input/kaggledays-warsaw/train.csv"):
    data = pd.read_csv(filename, sep="\t", index_col='id')
    msg = "Reading the data ({} rows). Columns: {}"
    print(msg.format(len(data), data.columns))
    try:
        y = data.loc[:, "answer_score"]
    except KeyError: # There are no answers in the test file
        return data, None
    return data, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(
        np.mean((np.log1p(y) - np.log1p(y0)) ** 2)
    )
X_train.head()
%%time

COLUMNS = list(X_train.columns)
default_preprocessor = TfidfVectorizer().build_preprocessor()


def field_extractor(field):
    field_idx = COLUMNS.index(field)
    return lambda x: default_preprocessor(x[field_idx])


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, fn=lambda x: x):
        super().__init__()
        self.columns = columns
        self.field_idx = [COLUMNS.index(c) for c in columns]
        self.fn = fn

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, data, *args, **kwargs):
        if isinstance(data, list):
            data = np.array(data)
        return self.fn(data[:, self.field_idx])
    
    def get_feature_names(self):
        return self.columns

    
vectorizer = FeatureUnion([
    # ('q_score', FeatureSelector(['question_score'], fn=lambda x: np.log1p(x.astype(int)))),
    # ('subreddit', CountVectorizer(token_pattern='\w+', preprocessor=field_extractor('subreddit'))),
    ('question', TfidfVectorizer(max_features=10000, token_pattern="\w+", preprocessor=field_extractor('question_text'))),
    ('answer', TfidfVectorizer(max_features=10000, token_pattern="\w+", preprocessor=field_extractor('answer_text'))),
    ])

model = SGDRegressor(max_iter=5)
model.fit(vectorizer.fit_transform(X_train.values), np.log1p(y_train.values));

print("Valid RMSLE:", rmsle(y_test, np.expm1(model.predict(vectorizer.transform(X_test.values)))))
eli5.explain_weights(model, vectorizer, top=50)
# eli5.explain_weights(model, vectorizer, top=100, feature_filter=lambda x: not x.startswith('subreddit_'))
test_sample = X_test.sample(n=10, random_state=42)
for row in test_sample.values:
    display(eli5.show_prediction(model, row, vec=vectorizer))