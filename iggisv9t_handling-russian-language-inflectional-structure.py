import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from scipy import sparse
from category_encoders.hashing import HashingEncoder
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.columns
cat_feats = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']
text_feats = ['title', 'description']
num_feats = ['price', 'item_seq_number']
allcols = cat_feats + text_feats + num_feats
merged = pd.concat((train[allcols], test[allcols]), axis=0)
merged['price'] = merged['price'].apply(np.log1p)
import pymorphy2
import re

morph = pymorphy2.MorphAnalyzer()
retoken = re.compile(r'[\'\w\-]+')
s = merged['description'].tail().values[-1]
print(s)
def tokenize_normalize(text):
    text = retoken.findall(text.lower())
    text = [morph.parse(x)[0].normal_form for x in text]
    return ' '.join(text)
tokenize_normalize(s)
# some descriptions only consist of a digits
merged['description'] = merged['description'].astype(str)
%%time
merged['description_norm'] = merged['description'].apply(tokenize_normalize)
%%time
tfidf = TfidfVectorizer(ngram_range=(1, 3), encoding='KOI8-R', min_df=100, max_df=0.999)
tfidf_matrices = []
for feat in ['description_norm', 'title']:
    tfidf_matrices.append(tfidf.fit_transform(merged[feat].fillna('').values))
%%time
tfidf_matrices = sparse.hstack(tfidf_matrices, format='csr')
print(tfidf_matrices.shape)
%%time
he = HashingEncoder()
cat_df = he.fit_transform(merged[cat_feats].values)
cat_df.head()
full_matrix = sparse.hstack([cat_df.values, tfidf_matrices, merged[num_feats].fillna(-1).values], format='csr')
import gc
del tfidf_matrices, merged, cat_df
gc.collect()
%%time
model = LGBMRegressor(max_depth=4, learning_rate=0.3, n_estimators=550)
res = cross_val_score(model, full_matrix[:train.shape[0]], train['deal_probability'].values, cv=4, scoring='neg_mean_squared_error')
res = [np.sqrt(-r) for r in res]
print(np.mean(res), np.std(res))
model.fit(full_matrix[:train.shape[0]], train['deal_probability'].values)
preds = model.predict(full_matrix[train.shape[0]:])
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10, 7))
plt.hist(preds, bins=50);
sub = pd.read_csv('../input/sample_submission.csv')
sub['deal_probability'] = preds
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('first_attempt.csv', index=False)
sub.head()