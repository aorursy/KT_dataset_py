import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from scipy import sparse
from category_encoders.hashing import HashingEncoder
import os
from scipy.sparse import hstack, csr_matrix
import tqdm
print(os.listdir("../input"))
%%time

train = pd.read_csv('../input/train.csv',nrows =10000)
test = pd.read_csv('../input/test.csv',nrows=10000)
cat_feats = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']
text_feats = ['title', 'description']
num_feats = ['price', 'item_seq_number']
allcols = cat_feats + text_feats + num_feats
merged = pd.concat((train[allcols], test[allcols]), axis=0)
merged['price'] = merged['price'].apply(np.log1p)
merged.head()
merged.isnull().sum()
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words = None,
    encoding='KOI8-R',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    dtype=np.float32,
    max_features=9000
)
# Character Stemmer
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    encoding='KOI8-R',
    analyzer='char',
    ngram_range=(1, 4),
    dtype=np.float32,
    max_features=5000
)
%%time
tfidf_matrices_1 = []
for feat in text_feats:
    tfidf_matrices_1.append(word_vectorizer.fit_transform(merged[feat].fillna('').values))
%%time
tfidf_matrices_2 = []
for feat in text_feats:
    tfidf_matrices_2.append(char_vectorizer.fit_transform(merged[feat].fillna('').values))
%%time
tfidf_matrices = sparse.hstack(tfidf_matrices_1,format='csr')
%%time
he = HashingEncoder()
cat_df = he.fit_transform(merged[cat_feats].values)

full_matrix = sparse.hstack([cat_df.values, tfidf_matrices, merged[num_feats].fillna(-1).values], format='csr')

%%time
model = LGBMRegressor(max_depth=4, learning_rate=0.3, n_estimators=500)
res = cross_val_score(model, full_matrix[:train.shape[0]], train['deal_probability'].values, cv=5, scoring='neg_mean_squared_error')
res = [np.sqrt(-r) for r in res]
print(np.mean(res), np.std(res))
model.fit(full_matrix[:train.shape[0]], train['deal_probability'].values)
preds = model.predict(full_matrix[train.shape[0]:])
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10, 7))
plt.hist(preds, bins=50);
#sub = pd.read_csv('../input/sample_submission.csv')
#sub['deal_probability'] = preds
#sub['deal_probability'].clip(0.0, 1.0, inplace=True)
#sub.to_csv('../input/first_attempt.csv', index=False)
#sub.head()
