import os
from time import time

import pandas as pd
import numpy as np

# for classification
from sklearn.ensemble import RandomForestClassifier

# for cross validation
from sklearn.model_selection import StratifiedKFold
# for searching for best params
from sklearn.model_selection import GridSearchCV

# for FastText vectorization
import gensim

import warnings
warnings.simplefilter(action='ignore')
# config
DATA_DIR = '../input/tweet-sentiment-extraction'
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
SUBMISSION_FILE = 'submission.csv'

RANDOM_STATE = 0
train_data = pd.read_csv(os.path.join(DATA_DIR, TRAIN_DATA_FILE)).fillna('')
test_data = pd.read_csv(os.path.join(DATA_DIR, TEST_DATA_FILE)).fillna('')
train_data = train_data[['textID', 'text', 'sentiment', 'selected_text']]
train_data[17:22]
test_data.head()
# create 2 target columns for 2 models 
starts = []
ends = []
for text, selected_text in zip(train_data['text'], train_data['selected_text']):
  start = text.find(selected_text)
  starts.append(start)
  ends.append(start + len(selected_text))

train_data['start_idx'] = starts
train_data['end_idx'] = ends

train_data.head(3)
DIM = 200  # vector size
print('Creating corpus for FastText model...')
t0 = time()
corpus = [text.lower().split() for text in train_data['text']]
corpus.extend([text.lower().split() for text in test_data['text']])
corpus.extend(train_data['sentiment'].unique().tolist())
print(f'Done in {time() - t0} seconds')

print('Building FastText model from corpus...')
t0 = time()
fast_text = gensim.models.FastText(corpus, size=DIM, min_count=1, min_n=1)
print(f'Done in {time() - t0} seconds')
# source: https://github.com/nstsj/compling_nlp_hse_course/blob/master/notebooks/Embeddings.ipynb
def get_embedding(text, model, dim):
    """Return FastText embeddings."""
    from collections import Counter
    text = text.lower().split()
    # text = text.split()
    
    words = Counter(text)
    total = len(text)
    vectors = np.zeros((len(words), dim))
    
    for i,word in enumerate(words):
        try:
            v = model[word]
            vectors[i] = v*(words[word]/total)
        except (KeyError, ValueError):
            raise
            # continue
    
    if vectors.any():
        vector = np.average(vectors, axis=0)
    else:
        vector = np.zeros((dim))
    
    return vector
print('Create embeddings for \'text\' ...')
X_text_ft = np.zeros((train_data.shape[0], DIM))

t0 = time()
for i, text in enumerate(train_data['text'].values):
    X_text_ft[i] = get_embedding(text, fast_text, DIM)
print(f'Done in {time() - t0} seconds')

X_sentiment_ft = np.zeros((train_data.shape[0], DIM))

print('... and \'sentiment\' columns in train data')
t0 = time()
for i, text in enumerate(train_data['sentiment'].values):
    X_sentiment_ft[i] = get_embedding(text, fast_text, DIM)
print(f'Done in {time() - t0} seconds')

print(X_text_ft.shape, X_sentiment_ft.shape)

print('Concatenated text and sentiment vectors:')
train_data_ft = np.concatenate([X_text_ft, X_sentiment_ft], axis=1)
print(train_data_ft.shape)
print('Create embeddings for \'text\' ...')
text_ft = np.zeros((test_data.shape[0], DIM))

t0 = time()
for i, text in enumerate(test_data['text'].values):
    text_ft[i] = get_embedding(text, fast_text, DIM)
print(f'Done in {time() - t0} seconds')

print('... and \'sentiment\' columns in test data')
sentiment_ft = np.zeros((test_data.shape[0], DIM))

t0 = time()
for i, text in enumerate(test_data['sentiment'].values):
    sentiment_ft[i] = get_embedding(text, fast_text, DIM)
print(f'Done in {time() - t0} seconds')

print(text_ft.shape, sentiment_ft.shape)

print('Concatenated text and sentiment vectors:')
kaggle_test_ft = np.concatenate([text_ft, sentiment_ft], axis=1)
print(kaggle_test_ft.shape)
# Build models
def create_model_starts():
    clf = RandomForestClassifier(
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=50,
        random_state=RANDOM_STATE
        )
    return clf

def create_model_ends():
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_leaf=17,
        min_samples_split=2,
        random_state=RANDOM_STATE
        )
    return clf
# # Grid Search for Models
# params = {
#     'n_estimators': [20, 50, 80, 120],
#     'max_depth': [7, 20, None],
#     'min_samples_leaf': [2, 10, 17, 25],
#     'min_samples_split': [2, 3],
#     'class_weight': [None, 'balanced']
# }

# starts_model = create_model_starts()
# print('Searching for best params for StartsModel...')
# gs_starts = GridSearchCV(starts_model,
#                          params,
#                          cv=5,
#                          verbose=2,
#                          n_jobs=-1
#                          )

# t0 = time()
# gs_starts.fit(train_data_ft, train_data['start_idx'])
# print(f'Done in {time() - t0} seconds')

# print('Best parameters for StartsModel')
# print(gs_starts.best_params_)
# # # # {'clf__C': 0.001, 'clf__class_weight': None, 'reduce__n_components': 2}

# # # 'Done in 2589.2142584323883 seconds' LOG_REG
# # # {'clf__C': 0.001, 'clf__class_weight': None}


# # 'Done in 9542.264384746552 seconds'
# # {'clf__max_depth': 20,
# #  'clf__min_samples_leaf': 2,
# #  'clf__min_samples_split': 2,
# #  'clf__n_estimators': 50}

# ends_model = create_model_ends()
# print('Searching for best params for EndssModel...')
# gs_ends = GridSearchCV(ends_model,
#                        params,
#                        cv=5,
#                        verbose=2,
#                        n_jobs=-1
#                        )

# t0 = time()
# gs_ends.fit(train_data_ft, train_data['end_idx'])
# print(f'Done in {time() - t0} seconds')

# print('Best parameters for StartsModel')
# print(gs_ends.best_params_)
def jaccard(top_selected):
    str1, str2 = top_selected
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
import warnings
warnings.simplefilter(action='ignore')

jac = []  # container for Jaccard scores per fold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# five-fold cross validation
for fold, (train_idx, test_idx) in enumerate(skf.split(train_data['text'], train_data['sentiment'])):
    
    print(f'>>> FOLD {fold + 1}')

    print('Training model 1...')
    model_starts = create_model_starts()
    t0 = time()
    model_starts.fit(train_data_ft[train_idx], train_data['start_idx'].loc[train_idx])
    print(f'Done in {time() - t0} seconds')

    print('Training model 2...')
    model_ends = create_model_ends()
    t0 = time()
    model_ends.fit(train_data_ft[train_idx], train_data['end_idx'].loc[train_idx])
    print(f'Done in {time() - t0} seconds')

    # Predict, evaluate
    print('Evaluating...')
    res = pd.DataFrame()
    res['pred_starts'] = model_starts.predict(train_data_ft[test_idx])  # predict starts
    res['pred_ends'] = model_ends.predict(train_data_ft[test_idx])      # predict ends
    
    columns = ['text', 'sentiment', 'selected_text']
    res[columns] = train_data[columns].loc[test_idx].reset_index(drop=True)

    # res['pred_select'] = res[['text', 'pred_starts', 'pred_ends']].apply(slice_text, axis=1)
    res['pred_select'] = res[['text', 'pred_starts', 'pred_ends']].apply(lambda x: x[0][x[1]:x[2]], axis=1)
    # Handle cases where start >= end
    condition = res['pred_starts'] >= res['pred_ends']
    res.loc[:, 'pred_select'][condition] = res.loc[:, 'text'][condition]

    res['score'] = res[['pred_select', 'selected_text']].apply(jaccard, axis=1)

    print(res.groupby('sentiment')['score'].mean())
    mean_jac = res['score'].mean()
    print(f"Mean score in Fold {fold + 1}: {mean_jac}")

    jac.append(mean_jac)

total_score = np.mean(jac)
print('>' * 10)
print(f'Total Jaccard score for 5 folds = {total_score}')
# Train models on all data
print('Training models on all data...')
t0 = time()
model_starts = create_model_starts()
model_starts.fit(train_data_ft, train_data['start_idx'])

model_ends = create_model_ends()
model_ends.fit(train_data_ft, train_data['end_idx'])

print(f'Done in {time() - t0} seconds')
temp_df = pd.DataFrame()
temp_df['pred_starts'] = model_starts.predict(kaggle_test_ft)  # predict starts
temp_df['pred_ends'] = model_ends.predict(kaggle_test_ft)      # predict ends
    
# columns = ['text', 'sentiment']
temp_df['text'] = test_data['text']
print(temp_df.head())
temp_df['selected_text'] = temp_df[[
                          'text', 
                          'pred_starts', 
                          'pred_ends']
                                ].apply(lambda x: x[0][x[1]:x[2]], axis=1)

# Handle cases where start >= end
condition = temp_df['pred_starts'] >= temp_df['pred_ends']
temp_df.loc[:, 'selected_text'][condition] = temp_df.loc[:, 'text'][condition]

submission_df = pd.DataFrame() 
submission_df['textID'] = test_data['textID']
submission_df['selected_text'] = temp_df['selected_text']
submission_df.to_csv(SUBMISSION_FILE, index = False)
pd.set_option('max_colwidth', 80)
test_data.head()
submission_df.head(5)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
plt.hist(res['selected_text'].map(len), alpha=0.8, bins=40)
plt.title('Distribution of Lengths of TRUE Substrings in Training Data')
plt.xlabel('Char length')
plt.ylabel('How often')
plt.show()
plt.hist(res['pred_select'].map(len), alpha=0.8, bins=40)
plt.title('Distribution of Lengths of PREDICTED Substrings in Training Data')
plt.xlabel('Char length')
plt.ylabel('How often')
plt.show()
plt.hist(submission_df['selected_text'].map(len), alpha=0.8, bins=40)
plt.title('Distribution of Lengths of PREDICTED Substrings in Testing Data')
plt.xlabel('Char length')
plt.ylabel('How often')
plt.show()