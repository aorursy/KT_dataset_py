import pandas as pd
import numpy as np
f = "../input/us_names.csv"
df = pd.read_csv(f)
df.drop(columns=['Unnamed: 0'], inplace=True)
df['Label'] = df[['F', 'M']].idxmax(axis=1)
df['Label'] = df['Label'].apply(lambda x: 1 if x == 'F' else 0)
# Shuffling
df = df[['name', 'Label']].sample(frac=1)
import phonetics
df['vowel_ending'] = df['name'].apply(lambda x: x[-1].lower() in set('aeiouy'))
df['last'] = df['name'].apply(lambda x: x[-1].lower())
df['last_allit'] = df['name'].apply(lambda x: x[-1] == x[-2])
df['phonetics'] = df['name'].apply(lambda x: phonetics.metaphone(x))
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

class DFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
metaphone_pipeline = Pipeline([
    ('selector', DFSelector('phonetics')),
    ('encoder', CountVectorizer(ngram_range=(3, 4), analyzer='char', lowercase=False))
])

cat_pipeline = Pipeline([
    ('selector', DFSelector(["vowel_ending", "last", "last_allit"])),
    ('encoder1', OneHotEncoder())
])

cnt_pipeline = Pipeline([
    ('selector_cnt', DFSelector('name')),
    ('vec', CountVectorizer(ngram_range=(2, 4), analyzer='char_wb', lowercase=False)),
])

from sklearn.pipeline import FeatureUnion
preprocessing = FeatureUnion(transformer_list=[
    ("metaphone_pipeline", metaphone_pipeline),
    ("cat_pipeline", cat_pipeline),
    ("cnt_pipeline", cnt_pipeline)
])
%%time
X, y = preprocessing.fit_transform(df), df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = y_train.values
y_test = y_test.values
# Get Feature Importances
def feature_importance():
    clf = ExtraTreesClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
    ranks = clf.feature_importances_
    feature_scores = {i: ranks[i] for i in (np.nonzero(ranks)[0])}
    id2feature = {}
    # fetch feature names for each transformer
    for _, t in preprocessing.transformer_list:
        encoder = t.steps[1][1]
        id2feature.update({i: v for i, v in enumerate(encoder.get_feature_names(), start=len(id2feature))})
    
    scores = []
    for idx, score in feature_scores.items():
        decoded_feature = id2feature[idx]
        scores.append((decoded_feature,score, idx))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
fi = feature_importance()
svd = TruncatedSVD(200)
X_train_reduced = svd.fit_transform(X_train)
X_test_reduced = svd.transform(X_test)
%matplotlib inline
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
plt.title('Truncated Data')
colors = {k: sns.xkcd_rgb[v] for k, v in [(0, 'bluish'), (1, 'barbie pink')]}
for data, label in zip(X_train_reduced[:1000], y_train[:1000]):
    plt.scatter(data[0], data[1], color=colors[label])
import xgboost as xgb
xgb_params = {'eta': 0.1, 
                  'max_depth': 5, 
                  'subsample': 0.5, 
                  'colsample_bytree': 0.5, 
                  'objective': 'binary:hinge', 
                  'eval_metric': 'error', 
                  'seed': 1234
                 }
#for eli5
d_train = xgb.DMatrix(X_train_reduced, y_train)
d_test = xgb.DMatrix(X_test_reduced, y_test)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
watchlist = [(d_train, 'train'), (d_test, 'valid')]
model_xgb = xgb.train(xgb_params, d_train, 500, watchlist, verbose_eval=1, early_stopping_rounds=20)
predictions = model_xgb.predict(d_test)
wrong = np.where(predictions!=y_test)
correct = predictions.shape[0] - wrong[0].shape[0]
ratio = correct / predictions.shape[0]

print("{} of {}".format(correct, predictions.shape[0]))
print("{:.2%}".format(ratio))
%%time
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train_reduced, y_train)
predictions = clf.predict(X_test_reduced)
wrong = np.where(predictions!=y_test)
correct = predictions.shape[0] - wrong[0].shape[0]
ratio = correct / predictions.shape[0]

print("{} of {}".format(correct, predictions.shape[0]))
print("{:.2%}".format(ratio))