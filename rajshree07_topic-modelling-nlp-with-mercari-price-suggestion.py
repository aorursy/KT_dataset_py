#Import Packages
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.linear_model import Ridge
import os
print(os.listdir("../input"))


# Observe the training set
train = pd.read_table('../input/train.tsv')
train.head()

print("The size of the training data is: " + str(train.shape))
print(train.dtypes)
train.astype('object').describe().transpose()
# Observe test set
test = pd.read_table('../input/test.tsv')
test.head()

test.shape
# Get 10% of the Training Data
reduced_X_train = train.sample(frac=0.1).reset_index(drop=True)
reduced_y_train = np.log1p(reduced_X_train['price'])
# Fast Cleaning of Data
reduced_X_train['category_name'] = reduced_X_train['category_name'].fillna('Other').astype(str)
reduced_X_train['brand_name'] = reduced_X_train['brand_name'].fillna('missing').astype(str)
reduced_X_train['shipping'] = reduced_X_train['shipping'].astype(str)
reduced_X_train['item_condition_id'] = reduced_X_train['item_condition_id'].astype(str)
reduced_X_train['item_description'] = reduced_X_train['item_description'].fillna('None')

%%time
from sklearn.decomposition import LatentDirichletAllocation

# Initialize CountVectorizer
cvectorizer = CountVectorizer(max_features=20000,
                              stop_words='english', 
                              lowercase=True)

# Fit it to our dataset
cvz = cvectorizer.fit_transform(reduced_X_train['item_description'])

# Initialize LDA Model with 10 Topics
lda_model = LatentDirichletAllocation(n_topics=10,
                                      random_state=42)

# Fit it to our CountVectorizer Transformation
X_topics = lda_model.fit_transform(cvz)

# Define variables
n_top_words = 10
topic_summaries = []

# Get the topic words
topic_word = lda_model.components_
# Get the vocabulary from the text features
vocab = cvectorizer.get_feature_names()

# Display the Topic Models
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' | '.join(topic_words)))
# Definte RMSLE Cross Validation Function
def rmsle_cv(model):
    kf = KFold(shuffle=True, random_state=42).get_n_splits(reduced_X_train['item_description'])
    rmse= np.sqrt(-cross_val_score(model, reduced_X_train['item_description'], reduced_y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse.mean())
from sklearn.linear_model import Ridge

vec = CountVectorizer()
clf = Ridge(random_state=42)
pipe = make_pipeline(vec, clf)
pipe.fit(reduced_X_train['item_description'], reduced_y_train)

cv_rmsle = rmsle_cv(pipe)

print("The Validation Score is: " + str(cv_rmsle))
import eli5
eli5.show_weights(pipe, vec=vec, top=100, feature_filter=lambda x: x != '<BIAS>')
eli5.show_prediction(clf, doc=reduced_X_train['item_description'][1297], vec=vec)
vec = CountVectorizer(stop_words='english')
clf = Ridge(random_state=42)
pipe = make_pipeline(vec, clf)
pipe.fit(reduced_X_train['item_description'], reduced_y_train)

cv_sw_rmsle = rmsle_cv(pipe)

print("The Validation Score is: " + str(cv_sw_rmsle))
eli5.show_prediction(clf, doc=reduced_X_train['item_description'][1297], vec=vec)
vec = TfidfVectorizer()
clf = Ridge(random_state=42)
pipe = make_pipeline(vec, clf)
pipe.fit(reduced_X_train['item_description'], reduced_y_train)

tfidf_rmsle = rmsle_cv(pipe)

print("The Validation Score is: " + str(tfidf_rmsle))
eli5.show_prediction(clf, doc=reduced_X_train['item_description'][1297], vec=vec)
vec = TfidfVectorizer(stop_words='english')
clf = Ridge(random_state=42)
pipe = make_pipeline(vec, clf)
pipe.fit(reduced_X_train['item_description'], reduced_y_train)

tfidf_sw_rmsle = rmsle_cv(pipe)

print("The Validation Score is: " + str(tfidf_sw_rmsle))
eli5.show_prediction(clf, doc=reduced_X_train['item_description'][1297], vec=vec)
vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
clf = Ridge(random_state=42)
pipe = make_pipeline(vec, clf)
pipe.fit(reduced_X_train['item_description'], reduced_y_train)

tfidf_sw_ng_rmsle = rmsle_cv(pipe)

print("The Validation Score is: " + str(tfidf_sw_ng_rmsle))
eli5.show_prediction(clf, doc=reduced_X_train['item_description'][1297], vec=vec)
print ("RMSLE Score: " + str(cv_rmsle) + " | CountVectorizer")
print ("RMSLE Score: " + str(cv_sw_rmsle) + " | CountVectorizer | Stop Words")
print ("RMSLE Score: " + str(tfidf_rmsle) + " | TF-IDF")
print ("RMSLE Score: " + str(tfidf_sw_rmsle) + " | TF-IDF | Stop Words")
print ("RMSLE Score: " + str(tfidf_sw_ng_rmsle) + " | TF-IDF | Stop Words | N-Grams")
from sklearn.pipeline import FeatureUnion

default_preprocessor = CountVectorizer().build_preprocessor()

def build_preprocessor(field):
    field_idx = list(reduced_X_train.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('category_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('category_name'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=55000,
        stop_words='english',
        preprocessor=build_preprocessor('item_description'))),
])
# Create Transformed Train Set
reduced_Xt_train = vectorizer.fit_transform(reduced_X_train.values)

def get_rmsle(y, pred): return np.sqrt(mean_squared_error(y, pred))
%%time

# Create 3-Fold CV
cv = KFold(n_splits=3, shuffle=True, random_state=42)
for train_ids, valid_ids in cv.split(reduced_Xt_train):
    # Define LGBM Model
    model_ridge = Ridge(solver = "lsqr", fit_intercept=True, random_state=42)
    
    # Fit LGBM Model
    model_ridge.fit(reduced_Xt_train[train_ids], reduced_y_train[train_ids])
    
    # Predict & Evaluate Training Score
    y_pred_train = model_ridge.predict(reduced_Xt_train[train_ids])
    rmsle_train = get_rmsle(y_pred_train, reduced_y_train[train_ids])
    
    # Predict & Evaluate Validation Score
    y_pred_valid = model_ridge.predict(reduced_Xt_train[valid_ids])
    rmsle_valid = get_rmsle(y_pred_valid, reduced_y_train[valid_ids])
    
    print(f'LGBM Training RMSLE: {rmsle_train:.5f}')
    print(f'LGBM Validation RMSLE: {rmsle_valid:.5f}')
%%time
from sklearn.linear_model import Lasso

# Create 3-Fold CV
cv = KFold(n_splits=3, shuffle=True, random_state=42)
for train_ids, valid_ids in cv.split(reduced_Xt_train):
    # Define LGBM Model
    model_LASSO = Lasso(fit_intercept=True, random_state=42)
    
    # Fit LGBM Model
    model_LASSO.fit(reduced_Xt_train[train_ids], reduced_y_train[train_ids])
    
    # Predict & Evaluate Training Score
    y_pred_train = model_LASSO.predict(reduced_Xt_train[train_ids])
    rmsle_train = get_rmsle(y_pred_train, reduced_y_train[train_ids])
    
    # Predict & Evaluate Validation Score
    y_pred_valid = model_LASSO.predict(reduced_Xt_train[valid_ids])
    rmsle_valid = get_rmsle(y_pred_valid, reduced_y_train[valid_ids])
    
    print(f'LASSO Training RMSLE: {rmsle_train:.5f}')
    print(f'LASSO Validation RMSLE: {rmsle_valid:.5f}')
%%time
import lightgbm as lgb

# Create 3-Fold CV
cv = KFold(n_splits=3, shuffle=True, random_state=42)
for train_ids, valid_ids in cv.split(reduced_Xt_train):
    # Define LGBM Model
    model_lgb = lgb.LGBMRegressor(num_leaves=31, n_jobs=-1, learning_rate=0.1, n_estimators=500, random_state=42)
    
    # Fit LGBM Model
    model_lgb.fit(reduced_Xt_train[train_ids], reduced_y_train[train_ids])
    
    # Predict & Evaluate Training Score
    y_pred_train = model_lgb.predict(reduced_Xt_train[train_ids])
    rmsle_train = get_rmsle(y_pred_train, reduced_y_train[train_ids])
    
    # Predict & Evaluate Validation Score
    y_pred_valid = model_lgb.predict(reduced_Xt_train[valid_ids])
    rmsle_valid = get_rmsle(y_pred_valid, reduced_y_train[valid_ids])
    
    print(f'LGBM Training RMSLE: {rmsle_train:.5f}')
    print(f'LGBM Validation RMSLE: {rmsle_valid:.5f}')
#Create Train/Test Split
train_X, test_X, train_y, test_y = train_test_split(reduced_Xt_train, reduced_y_train, test_size=0.2, random_state=211)
# Define LGBM Model
model_lgb = lgb.LGBMRegressor(num_leaves=31, n_jobs=-1, learning_rate=0.1, n_estimators=500, random_state=40)

# Fit LGBM Model
model_lgb.fit(train_X, train_y)

# Predict with LGBM Model
lgbm_y_pred = model_lgb.predict(test_X)
# Define Ridge Model
model_ridge = Ridge(solver = "lsqr", fit_intercept=True, random_state=42)
    
# Fit Ridge Model
model_ridge.fit(train_X, train_y)
    
# Evaluate Training Score
ridge_y_pred = model_ridge.predict(test_X)
ensemble_y_pred = (lgbm_y_pred+ridge_y_pred)/2

ensemble_rmsle = get_rmsle(ensemble_y_pred, test_y)

print(f'Ensemble RMSLE: {ensemble_rmsle:.5f}')
#Ensemble Predictions without Inverse Log Transformation
ensemble_y_pred[0:20]
# Ensemble Predictions (Inverse Log - Exponential)
ensemble_y = (np.expm1(lgbm_y_pred)+np.expm1(ridge_y_pred))/2
ensemble_y[200:220]
# Test Predictions 
np.expm1(test_y[200:220])