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
def preprocess(input, only_char=True, lower=False, as_array=False):
    input = re.sub(r"[^\x00-\x7F]+"," ", input)
    if lower: input = input.lower()
    if only_char:
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(input)
        input = " ".join(tokens)
    tokens = word_tokenize(input)
    tokens = [w for w in tokens if len(w) > 1]
    tokens = ["0" if w.isdigit() else w for w in tokens]
    if not as_array:
        return " ".join(tokens)
    else:
        return tokens
import requests
from tqdm import tqdm_notebook


def download_from_url(url, path):
    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(path):
        first_byte = os.path.getsize(path)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size

    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm_notebook(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(path, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

download_from_url("https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/glove.840B.300d.txt.zip", "./glove.840B.300d.txt.zip")
from zipfile import ZipFile

if not os.path.exists("./glove.840B.300d.txt"):
    with ZipFile("./glove.840B.300d.txt.zip") as z:
        z.extractall("glove.840B.300d.txt", "glove.840B.300d.txt")
    print(os.listdir("."))
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


class VectorizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, preprocessor, embedding_path, sublinear_tf=True, max_df=1.0, min_df=0.025,
                 ngram_range=(1, 1), n_components=-1):
        self.preprocessor = preprocessor
        self.embedding_path = embedding_path
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.text_columns = ["title", "abstract", "conclusion", "evaluation"]
        self.excludes = ["conference", "name", "label"]
        self.scaler = StandardScaler()
        self.embedding = {}
        self.vector_size = 0
        self.handf = []
    
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
                        preprocessor=self.preprocessor, ngram_range=self.ngram_range,
                        analyzer="word", stop_words="english")
        vectorizer.fit(all_text)
        with open(self.embedding_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                if word in vectorizer.vocabulary_:
                    self.embedding[word] = coefs
                    self.vector_size = len(coefs)
        
        # Make Hand feature scaler
        _X = self._extract_handf(X)
        self.handf = _X.columns
        self.scaler.fit(_X.astype(np.float64))
        return self

    def text_to_vec(self, text_series):
        text_part = " ".join(text_series)
        tokens = self.preprocessor(text_part, as_array=True)
        vector = np.zeros(self.vector_size)
        for w in tokens:
            if w in self.embedding:
                vector += self.embedding[w]
        if len(tokens) > 0:
            return vector / len(tokens)
        else:
            return vector
    
    def transform(self, X):
        encoded = X[self.text_columns].apply(self.text_to_vec, axis=1)
        encoded = np.vstack(encoded.values)
        _X = self.scaler.transform(self._extract_handf(X).astype(np.float64))
        features = np.hstack((encoded, _X))
        return features
from sklearn.linear_model import LogisticRegression


model = LogisticRegression(penalty="l1", class_weight="balanced")
from sklearn.pipeline import Pipeline


pipe = Pipeline(steps=[
            ("fe", VectorizerTransformer(preprocess, "glove.840B.300d.txt/glove.840B.300d.txt")),
            ("model", model)])
label_train = df_train["label"]
label_dev = df_dev["label"]

pipe.fit(df_train, label_train)
from sklearn.metrics import classification_report

print(len(pipe.named_steps["fe"].embedding))

predicteds = pipe.predict(df_dev)
print(classification_report(label_dev, predicteds))
file_names = "data_yans/arxiv.cs.cl-lg_2007-2017/test/parsed_pdfs/" + df_test["name"] + ".json"
answer = pd.DataFrame({
    "file_names": file_names,
    "conference": df_test["conference"],
    "label": pipe.predict(df_test)
})
answer.head(5)
answer.to_csv("answer_file_w2v.csv", index=False)
print(os.listdir("."))