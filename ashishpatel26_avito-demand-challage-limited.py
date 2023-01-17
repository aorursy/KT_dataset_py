import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 500)
data = pd.read_csv("../input/train.csv")
data.head()
cols_to_drop = ["item_id", "user_id", "city", "param_1", "param_2", "param_3", "title",
    "activation_date", "item_seq_number", "image", "image_top_1"]
data = data.drop(labels=cols_to_drop, axis=1)
data.head()
from sklearn.preprocessing import LabelEncoder
parent_category = LabelEncoder()
parent_category.fit(data["parent_category_name"])
data["parent_category_name"] = parent_category.transform(data["parent_category_name"])
category = LabelEncoder()
category.fit(data["category_name"])
data["category_name"] = category.transform(data["category_name"])
user_type = LabelEncoder()
user_type.fit(data["user_type"])
data["user_type"] = user_type.transform(data["user_type"])
region = LabelEncoder()
region.fit(data["region"])
data["region"] = region.transform(data["region"])
data = data.dropna()
data.head()
data["description"].fillna("", inplace=True)
from nltk import casual_tokenize
def tokenize(text):
    tokens = casual_tokenize(str(text))
    clean_stuff = [word.lower() for word in tokens if word.isalpha()]
    line = " ".join(clean_stuff)
    return line
%%time
data["description"] = data["description"].apply(tokenize)
cat_num_cols = ["region", "parent_category_name", "category_name", "price", "user_type"]
X_cat_num = data[cat_num_cols].values
y = data["deal_probability"].values
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
tfidf = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words=stopwords.words("russian"),
    max_features=10000
)
%%time
tfidf.fit(data["description"].values)
X_texts = tfidf.transform(data["description"].values)
from scipy.sparse import hstack
X = hstack((X_texts, X_cat_num))
X.shape
import os
from sklearn.externals import joblib
try:
    os.mkdir("./models")
except:
    pass
joblib.dump(X, "./models/X.pkl")
joblib.dump(y, "./models/y.pkl")
joblib.dump(tfidf, "./models/tfidf.pkl")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.metrics import mean_squared_error, make_scorer
from math import sqrt
def rmse_func(y_calc, y_test):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

rmse = make_scorer(rmse_func, greater_is_better=False)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
%env JOBLIB_TEMP_FOLDER=/tmp
params_svr = {"C": np.arange(0.1, 100.1, 0.1)}
svr = GridSearchCV(
    SVR(),
    param_grid=params_svc,
    scoring=rmse,
    cv=5,
    verbose=1,
    n_jobs=-1
)
svr.fit(X_train, y_train)
print("SVR results:\n\t- best params: {}\n\t- best score: {}".format(svc.best_params_, svc.best_score_))
result = svr.predict(y)
ids = data['item_id']
ids['deal_probability'] = result
ids.to_csv("submit2.csv",index=True,header=True)