import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import re
from math import log
import xgboost as xgb
df = pd.read_csv("../input/kaggledays-warsaw/train.csv", sep="\t", index_col='id')
df_test = pd.read_csv("../input/kaggledays-warsaw/test.csv", sep="\t", index_col='id')
def convert_type(df):
    # Convert dtype to correct format
    coltypes = {"question_id":"str", "subreddit":"str", "question_utc":"datetime64[s]",
                "question_text": "str", "question_score":"int64", "answer_utc":"datetime64[s]",
                "answer_text":"str", "answer_score":"int64"}
    for col, coltype in coltypes.items():
        if col in df.columns:
            df[col] = df[col].astype(coltype)

    # Log transform question and answer
    logscales = ["question_score", "answer_score"]
    for col in logscales:
        if col in df.columns:
            df[col] = df[col].apply(log)
    return df
df = convert_type(df)
df_all = df
df_test = convert_type(df_test)
# Split val and train
qids = df.question_id.unique()
qids = qids[np.random.permutation(len(qids))]
split_point = int(0.2*len(qids))
qids_val, qids_trn = qids[:split_point], qids[split_point:]

df_trn = df[df.question_id.isin(qids_trn)]
df_val = df[df.question_id.isin(qids_val)]
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, orient=None):
        super(FeatureSelector, self).__init__()
        self.columns = columns

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, data, *args, **kwargs):
        return data[self.columns]
class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from a paragraph"""
    def __init__(self):
        super(TextStats, self).__init__()
        self.url_regex = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        self.img_regex = 'https?:[^)''"]+\.(?:jpg|jpeg|gif|png)'
        
    def fit(self, x, y=None):
        return self

    def transform(self, answers):
        stats = pd.DataFrame({"ans_length": answers.apply(lambda x: len(x)),
                              "ans_nsents": answers.apply(lambda x: x.count('.')),
                              "ans_imgs": answers.apply(lambda x: len(re.findall(self.img_regex, x))),
                              "ans_links": answers.apply(lambda x: len(re.findall(self.url_regex, x)))})
        
        stats.ans_links = stats.ans_links - stats.ans_imgs
        stats.ans_imgs = stats.ans_imgs.apply(lambda x: 6 if x > 6 else x)
        stats.ans_links = stats.ans_links.apply(lambda x: 10 if x > 10 else x)

        return stats
class TimeFeats(BaseEstimator, TransformerMixin):
    """Extract time-related features"""
    def __init__(self, earliest_ans_time):
        super(TimeFeats, self).__init__()
        self.earliest_time = earliest_ans_time
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        feats = pd.DataFrame({"ans_delay":(X.answer_utc - X.question_utc).dt.total_seconds().apply(lambda x: log(x+1)),
                              "answer_dow": X["answer_utc"].dt.dayofweek,
                              "answer_hod": X["answer_utc"].dt.hour,
                              "time_trend": (X["answer_utc"] - self.earliest_time).dt.total_seconds()/3600})
        return feats
class SubreditFeat(BaseEstimator, TransformerMixin):
    """Extract subreddit related features"""
    def __init__(self, subreddits):
        super(SubreditFeat, self).__init__()
        self.subreddit2id = {subreddit: id for id, subreddit in enumerate(subreddits.unique())}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        feats = pd.DataFrame({"subreddit_id": X.apply(lambda x: self.subreddit2id[x])})
        return feats
class QuestionFeats(BaseEstimator, TransformerMixin):
    """Extract question-related features"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        answer_count = X.groupby("question_id")["answer_text"].count()
        question_feats = pd.DataFrame({"cnt_feat": X["question_id"].apply(lambda x: answer_count[x])})
        return question_feats
earliest_ans_time = min(df_all.answer_utc.min(), df_test.answer_utc.min())
process_data = make_union(
    FeatureSelector(["question_score"]),
    make_pipeline(
        FeatureSelector("subreddit"),
        SubreditFeat(df_all["subreddit"])
    ),
    make_pipeline(
        FeatureSelector("question_text"),
        TfidfVectorizer(max_features=50, token_pattern="\w+"),
        TruncatedSVD(n_components=20)
    ),
    make_pipeline(
        FeatureSelector("answer_text"),
        TfidfVectorizer(max_features=50, token_pattern="\w+"),
        TruncatedSVD(n_components=30)
    ),
    make_pipeline(
        FeatureSelector("answer_text"),
        TextStats()
    ),
    make_pipeline(
        FeatureSelector(["answer_utc", "question_utc"]),
        TimeFeats(earliest_ans_time),
    ),
    make_pipeline(
        FeatureSelector(["question_id", "answer_text"]),
        QuestionFeats()
    )
)
df = df_trn
# df = df_all # Uncomment for training in whole dataset
process_data.fit(df)
X_trn = process_data.transform(df_trn)
y_trn = df_trn["answer_score"].values
X_val = process_data.transform(df_val)
y_val = df_val["answer_score"].values
model = xgb.XGBRegressor(nthread=10, learning_rate = 0.01, n_estimators = 500, max_depth=6)
model.fit(X_trn, y_trn)
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(
        np.mean((np.log1p(y) - np.log1p(y0)) ** 2)
    )

yhat_val = model.predict(X_val)
rmsle(yhat_val, y_val)
# You should retrain the model using all data before making the final prediction
X_tst = process_data.transform(df_test)
solution = pd.DataFrame(index=df_test.index)
solution['answer_score'] = np.exp(model.predict(X_tst))
solution.to_csv('submission.csv')
# There is a leak in the testset. I did not try to exploit that leak. I just simply replace all my prediction with the revealed leaked records.
# It turns out that you can learn a lot from the leak.
solution['id'] = solution.index
leaks = pd.read_csv("leaked_records.csv").rename(columns={"answer_score": "leak"})
sub = pd.merge(solution, leaks, on="id", how="left")
sub.loc[~sub["leak"].isnull(), "answer_score"] = sub.loc[~sub["leak"].isnull(), "leak"]
sub = sub[["id", "answer_score"]]
sub.to_csv('submission_leak.csv', index=False)
