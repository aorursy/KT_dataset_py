# !pip install kaggle
# !pip install xgboost 
# !pip install graphviz
import matplotlib.pyplot as plt
import xgboost
import kaggle
import pandas
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
train = pandas.read_csv("../input/train.csv")
train["train_test"] = "train"
test = pandas.read_csv("../input/test.csv")
test["train_test"] = "test"
data = pandas.concat([train, test])
data[:3]
data = pandas.get_dummies(data)
data[:3]
data.fillna(value=-1, inplace=True)
X_train = data[(data["train_test_train"] == 1)].drop(["train_test_test", "train_test_train", "SalePrice"], axis=1)
y_train = data[(data["train_test_train"] == 1)][["SalePrice"]]
X_test = data[(data["train_test_test"] == 1)].drop(["train_test_test", "train_test_train", "SalePrice"], axis=1)
# y_test = data[(data["train_test_test"] == 1)][["SalePrice"]]
# clf = xgboost.XGBClassifier(n_jobs=-1)#DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# # print(classification_report(y_test, y_pred))

# # lang_stopwords = stopwords.words("arabic")

# # tweet_tok = TweetTokenizer(strip_handles = True, reduce_len = True)

svc_params  = {
    'fit_intercept': True,
    'max_iter': 1000,#4700,
    'tol': 0.0070961981175485384,
    'C': 1.85101225107496248,
    'class_weight': 'balanced',
    'penalty': 'l2',
    'multi_class': 'ovr'
}

# # tfidf
# tfidf = TfidfVectorizer(**tf_params)
# linearsvm
linear = LinearSVC(**svc_params)

pipeline = Pipeline([
#     ('tfidf', tfidf), 
    ('linear_svc', linear)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
# Generate Submission File 
sample_submission = pandas.DataFrame({ 'Id': list(range(1461, 2920)),'SalePrice': y_pred })
sample_submission.to_csv("sample_submission.csv", index=False)

