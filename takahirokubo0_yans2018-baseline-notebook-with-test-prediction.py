import os
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_dev = pd.read_feather("../input/dev_feature.feather")
df_train = pd.read_feather("../input/train_feature.feather")
df_test = pd.read_feather("../input/test_feature.feather")
df_test.head(5)
def preprocess(input, only_char=True, lower=True, stemming=False):
    input = re.sub(r"[^\x00-\x7F]+"," ", input)
    if lower: input = input.lower()
    if only_char:
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(input)
        input = " ".join(tokens)
    tokens = word_tokenize(input)
    tokens = [w for w in tokens if len(w) > 1]
    tokens = ["0" if w.isdigit() else w for w in tokens]
    return " ".join(tokens)
def test_preprocess():
    text = "New variants can be readily derived from BAHSIC by combining the two building blocks of BAHSIC 2017."
    print(preprocess(text, only_char=True, lower=True))

test_preprocess()
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


class VectorizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, preprocessor, sublinear_tf=True, max_df=0.8, min_df=0.025,
                 ngram_range=(1, 1), n_components=-1):
        self.preprocessor = preprocessor
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.text_columns = ["title", "abstract", "conclusion", "evaluation"]
        self.excludes = ["conference", "name", "label"]
        self.scaler = StandardScaler()
        self.vectorizer = None
        self.vector_size = 0
        self.handf_size = 0
    
    @property
    def vocab(self):
        return self.vectorizer.vocabulary_
    
    def _extract_handf(self, X):
        return X.drop(columns=(self.text_columns + self.excludes))
    
    def fit(self, X, y=None):
        # Make Text Encoder
        all_text = X[self.text_columns].apply(lambda x: " ".join(x), axis=1)
        vectorizer = TfidfVectorizer(
                        sublinear_tf=self.sublinear_tf, max_df=self.max_df, min_df=self.min_df,
                        preprocessor = self.preprocessor, ngram_range = self.ngram_range,
                        analyzer="word", stop_words="english")
        vectorizer.fit(all_text)
        self.vectorizer = vectorizer
        self.vector_size = len(vectorizer.vocabulary_)
        
        # Make Hand feature scaler
        _X = self._extract_handf(X)
        self.handf_size = _X.shape[1]
        self.scaler.fit(_X.astype(np.float64))
        return self

    def transform(self, X):
        text_part = X[self.text_columns].apply(lambda x: " ".join(x), axis=1)
        encoded = self.vectorizer.transform(text_part)
        sparse = True
        if self.n_components > 0:
            svd = TruncatedSVD(n_components=self.n_components)
            encoded = svd.fit_transform(encoded)
            self.vector_size = self.n_components
            sparse = False

        _X = self.scaler.transform(self._extract_handf(X).astype(np.float64))
        if sparse:
            features = hstack((encoded, _X))
        else:
            features = np.hstack((encoded, _X))
        return features
class SOTATransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.pattern = ["state of the art", "state-of-the-art", "sota"]
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sota = X["abstract"].apply(lambda x: max([(1 if p in x.lower() else 0) for p in self.pattern]))
        X["abstract_contains_sota"] = sota
        return X
def test_sota_transformer(df):
    t = SOTATransformer()
    print(t.transform(df)["abstract_contains_sota"])

test_sota_transformer(pd.DataFrame({"abstract": ["state of the art is awesome", "SOTA is wonderful", "state-of-the-art done", "no state"]}))
def test_vectorizer(df):
    v = VectorizerTransformer(preprocess, n_components=50)
    features = v.fit_transform(df)
    print("Done")
    print((v.vector_size, v.handf_size))
    print(v.vocab)

test_vectorizer(df_dev)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


model = LogisticRegression(penalty="l1", class_weight="balanced")
"""
No SVC score is below
              precision    recall  f1-score   support

           0       0.85      0.74      0.79       466
           1       0.57      0.72      0.63       220

   micro avg       0.73      0.73      0.73       686
   macro avg       0.71      0.73      0.71       686
weighted avg       0.76      0.73      0.74       686
"""
# model = SVC(C=1.0, class_weight="balanced")
from sklearn.pipeline import Pipeline


pipe = Pipeline(steps=[
            ("fe", VectorizerTransformer(preprocess)),
            ("model", model)])
label_train = df_train["label"]
label_dev = df_dev["label"]

pipe.fit(pd.concat([df_train, df_dev], axis=0), pd.concat([label_train, label_dev], axis=0))
from sklearn.model_selection import GridSearchCV
# Grid Search

def hyper_parameter_search():
    param_grid = [
        {
            "fe__max_df": [0.8, 0.9, 1.0],
            "fe__min_df": [0.005, 0.001, 0.02, 0.025]
        }
    ]
    """
    param_grid = [
        {
            "model__C": [1.0, 10],
            "model__kernel": ["linear", "rbf"]
        }
    ]
    """
    
    grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid, verbose=10)
    grid.fit(df_train, label_train)

    print(grid.cv_results_)
    return grid.best_estimator_

# pipe = hyper_parameter_search()

"""
result = {'params': [
            {'fe__max_df': 0.1, 'fe__min_df': 0.01, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 0.1, 'fe__min_df': 0.01, 'fe__ngram_range': (1, 2)}, 
            {'fe__max_df': 0.1, 'fe__min_df': 0.025, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 0.1, 'fe__min_df': 0.025, 'fe__ngram_range': (1, 2)}, 
            {'fe__max_df': 0.1, 'fe__min_df': 0.05, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 0.1, 'fe__min_df': 0.05, 'fe__ngram_range': (1, 2)}, 
            {'fe__max_df': 0.5, 'fe__min_df': 0.01, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 0.5, 'fe__min_df': 0.01, 'fe__ngram_range': (1, 2)}, 
            {'fe__max_df': 0.5, 'fe__min_df': 0.025, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 0.5, 'fe__min_df': 0.025, 'fe__ngram_range': (1, 2)}, 
            {'fe__max_df': 0.5, 'fe__min_df': 0.05, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 0.5, 'fe__min_df': 0.05, 'fe__ngram_range': (1, 2)}, 
            {'fe__max_df': 1.0, 'fe__min_df': 0.01, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 1.0, 'fe__min_df': 0.01, 'fe__ngram_range': (1, 2)}, 
            {'fe__max_df': 1.0, 'fe__min_df': 0.025, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 1.0, 'fe__min_df': 0.025, 'fe__ngram_range': (1, 2)}, 
            {'fe__max_df': 1.0, 'fe__min_df': 0.05, 'fe__ngram_range': (1, 1)}, 
            {'fe__max_df': 1.0, 'fe__min_df': 0.05, 'fe__ngram_range': (1, 2)}],
          'mean_test_score': [
                0.72503644, 0.72922741, 0.72120991, 0.72139213, 0.71282799,
                0.7141035 , 0.73706268, 0.73742711, 0.7372449 , 0.7361516 ,
                0.73414723, 0.73414723, 0.73706268, 0.73669825, 0.73906706,
                0.7361516 , 0.73414723, 0.73432945],
          'rank_test_score': [
                14, 13, 16, 15, 18,
                17,  4,  2,  3,  7,
                10, 10,  4,  6,  1,
                7, 10, 9]}
"""
from sklearn.metrics import classification_report


predicteds = pipe.predict(df_dev)
print(classification_report(label_dev, predicteds))
file_names = "data_yans/arxiv.cs.cl-lg_2007-2017/test/parsed_pdfs/" + df_test["name"] + ".json"
answer = pd.DataFrame({
    "file_names": file_names,
    "conference": df_test["conference"],
    "label": pipe.predict(df_test)
})
answer.head(5)
answer.to_csv("answer_file.csv", index=False)
print(os.listdir("."))