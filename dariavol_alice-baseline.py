import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# Read the training and test data sets, change paths if needed
train_df = pd.read_csv('../input/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../input/test_sessions.csv',
                      index_col='session_id')

# Convert time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()
# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)
test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)

# Load websites dictionary
with open(r"../../mlcourse.ai/data/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
sites_dict.head()
time_df = pd.DataFrame(index=train_df.index)
time_df['target'] = train_df['target']
# Create a separate dataframe where we will work with timestamps
time_df = pd.DataFrame(index=train_df.index)
time_df['target'] = train_df['target']

# Find sessions' starting and ending
time_df['min'] = train_df[times].min(axis=1)
time_df['max'] = train_df[times].max(axis=1)

# Calculate sessions' duration in seconds
time_df['seconds'] = (time_df['max'] - time_df['min']) / np.timedelta64(1, 's')

time_df.head()
# Our target variable
y_train = train_df['target']

# United dataframe of the initial data 
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Index to split the training and test data sets
idx_split = train_df.shape[0]
# Dataframe with indices of visited websites in session
full_sites = full_df[sites]
full_sites.head()
# sequence of indices
sites_flatten = full_sites.values.flatten()

# and the matrix we are looking for 
# (make sure you understand which of the `csr_matrix` constructors is used here)
# a further toy example will help you with it
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0]  + 10, 10)))[:, 1:]
def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed, solver='liblinear').fit(X[:idx, :], y[:idx])
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)
    
    return score
%%time
# Select the training set from the united dataframe (where we have the answers)
X_train = full_sites_sparse[:idx_split, :]

# Calculate metric on the validation set
#print(get_auc_lr_valid(X_train, y_train))
# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
# Train the model on the whole training data set
# Use random_state=17 for repeatability
# Parameter C=1 by default, but here we set it explicitly
lr = LogisticRegression(C=1.0, random_state=17, solver='liblinear').fit(X_train, y_train)

# Make a prediction for test data set
X_test = full_sites_sparse[idx_split:,:]
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, 'baseline_1.csv')
# Dataframe for new features
full_new_feat = pd.DataFrame(index=full_df.index)

# Add start_month feature
full_new_feat['start_month'] = full_df['time1'].apply(lambda ts: 
                                                      100 * ts.year + ts.month).astype('int')
# Add the new feature to the sparse matrix
tmp = full_new_feat[['start_month']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# Compute the metric on the validation set
#print(get_auc_lr_valid(X_train, y_train))
# Add the new standardized feature to the sparse matrix
tmp = StandardScaler().fit_transform(full_new_feat[['start_month']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# Compute metric on the validation set
#print(get_auc_lr_valid(X_train, y_train))
full_new_feat['start_hour'] = full_df[times].min(axis=1).apply(lambda ts: ts.hour).astype('int64')
full_new_feat['morning'] = full_new_feat['start_hour'].apply(lambda x: 1 if x <= 11 else 0)
tmp = full_new_feat[['start_hour','morning']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# Compute the metric on the validation set
#print(get_auc_lr_valid(X_train, y_train))
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 
                                                           'start_hour', 
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 
                             tmp_scaled[:idx_split,:]]))

# Capture the quality with default parameters
#score_C_1 = get_auc_lr_valid(X_train, y_train)
#print(score_C_1)
from tqdm import tqdm

# List of possible C-values
Cs = np.logspace(-3, 1, 10)
scores = []
for C in tqdm(Cs):
    scores.append(get_auc_lr_valid(X_train, y_train, C=C))
plt.plot(Cs, scores, 'ro-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('AUC-ROC')
plt.title('Regularization Parameter Tuning')
# horizontal line -- model quality with default C value
plt.axhline(y=score_C_1, linewidth=.5, color='b', linestyle='dashed') 
plt.show()
# Prepare the training and test data
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 'start_hour', 
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 
                             tmp_scaled[:idx_split,:]]))
X_test = csr_matrix(hstack([full_sites_sparse[idx_split:,:], 
                            tmp_scaled[idx_split:,:]]))

# Train the model on the whole training data set using optimal regularization parameter
lr = LogisticRegression(C=C, random_state=17, solver='liblinear').fit(X_train, y_train)

# Make a prediction for the test set
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the submission file
write_to_submission_file(y_test, 'baseline_2.csv')
train_df_my = train_df
train_df_my.head()
train_df_my['day_of_week'] = train_df_my['time1'].apply(lambda ts: ts.dayofweek).astype('float64')
train_df_my[train_df_my['target'] == 1].groupby('day_of_week').size().plot(kind='bar');
train_df_my['start_month'] = train_df_my['time1'].apply(lambda ts: 
                                                      100 * ts.year + ts.month).astype('float64')
train_df_my[train_df_my['target'] == 1].groupby('start_month').size().plot(kind='bar');
train_df_my['start_hour'] = train_df_my['time1'].apply(lambda ts: ts.hour).astype('int8')
train_df_my['morning'] = train_df_my['start_hour'].apply(lambda x: 1 if x <= 11 else 0)
train_df_my[train_df_my['target'] == 1].groupby('start_hour').size().plot(kind='bar');
train_df_my[train_df_my['target'] == 0].groupby('start_hour').size().plot(kind='bar');
train_df_my[train_df_my['target'] == 1].groupby('morning').size().plot(kind='bar');
train_df_my['day_of_month'] = train_df_my['time1'].apply(lambda ts: ts.day).astype('float64')
train_df_my[train_df_my['target'] == 1].groupby('day_of_month').size().plot(kind='bar');
train_df_my[train_df_my['target'] == 0].groupby('day_of_month').size().plot(kind='bar');
top_sites_for_notA = pd.Series(train_df[train_df['target'] == 0][sites].values.flatten()
                     ).value_counts().sort_values(ascending=False).head(80)
print(top_sites_for_A)
sites_dict.loc[top_sites_for_A.index]
train_df_my['top_sites'] = 0
for site in sites:
    train_df_my['top_sites'] += train_df_my[site].isin(top_sites_for_notA).astype('int8')
train_df_my[train_df_my['target'] == 1].groupby('top_sites').size()
train_df_my[train_df_my['target'] == 0].groupby('top_sites').size()
train_df_my['diff'] = 0

full_new_feat['day_of_week'] = full_df['time1'].apply(lambda ts:ts.dayofweek).astype('int8')
full_new_feat = pd.concat([full_new_feat, pd.get_dummies(full_new_feat['day_of_week'], prefix = 'day_of_week')], axis = 1)
full_new_feat['seconds'] = (full_df[times].max(axis = 1) - full_df[times].min(axis = 1)) / np.timedelta64(1, 's')
full_new_feat['start_hour'].unique()
full_new_feat = pd.concat([full_new_feat, pd.get_dummies(full_new_feat['start_hour'], prefix = 'start_hour')], axis = 1)
full_new_feat['top_sites'] = 0
for site in sites:
    full_new_feat['top_sites'] += full_df[site].isin(top_sites_for_notA).astype('int8')
full_new_feat['top_sites'] = np.sqrt(full_new_feat['top_sites'] * 10)
full_new_feat['top'] = full_new_feat['top_sites'].apply(lambda x: 1 if x > 0 else 0)
full_new_feat['day_of_month'] = full_df['time1'].apply(lambda ts:ts.day).astype('int8')
full_to_text = full_df[sites].apply(
    lambda x: " ".join([str(a) for a in x.values if a != 0]), axis=1)\
               .values.reshape(len(full_df[sites]), 1)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
pipeline = Pipeline([
    ("vectorize", CountVectorizer()),
    ("tfidf", TfidfTransformer())
])
pipeline.fit(full_to_text.ravel())

X_full_sparse = pipeline.transform(full_to_text.ravel())

X_full_sparse.shape
data = pd.DataFrame(index=full_df.index)
data = full_df[sites]
data['list_of_words'] = full_df['site1'].apply(str)
data['list_of_words'] += ','
for i in range(2, 10):
    data['list_of_words'] += full_df['site%s'% i].apply(str)
    data['list_of_words'] += ','
full_df['list_of_words'] = data['list_of_words'].apply(lambda x: x.split(','))
from gensim.models import word2vec
text_model = word2vec.Word2Vec(data['list_of_words'], size=500, window=5, workers=-1)
w2v = dict(zip(text_model.wv.index2word, text_model.wv.syn0))
class sense_vectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(w2v.values())))

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[word] for word in words if word in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit(self, X):
        return self 
full_new_feat.head()
feats = ['start_month', 'start_hour', 'seconds', 'day_of_week', 'top']
feats += ['day_of_week_' + str(i) for i in range(7)]
feats += ['start_hour_' + str(i) for i in range(7, 24)]
tmp_scaled = StandardScaler().fit_transform(full_new_feat[feats])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 
                             tmp_scaled[:idx_split,:]]))

# Capture the quality with default parameters
#score_C_1 = get_auc_lr_valid(X_train, y_train)
#print(score_C_1)
from tqdm import tqdm

# List of possible C-values
Cs = np.logspace(-3, 1, 10)
scores = []
for C in tqdm(Cs):
    scores.append(get_auc_lr_valid(X_train, y_train, C=C))
plt.plot(Cs, scores, 'ro-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('AUC-ROC')
plt.title('Regularization Parameter Tuning')
# horizontal line -- model quality with default C value
plt.axhline(y=score_C_1, linewidth=.5, color='b', linestyle='dashed') 
plt.show()
max(zip(scores, Cs))
C_optim = max(zip(scores, Cs))[1]
# Prepare the training and test data
tmp_scaled = StandardScaler().fit_transform(full_new_feat[feats])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 
                             tmp_scaled[:idx_split,:]]))
X_test = csr_matrix(hstack([full_sites_sparse[idx_split:,:], 
                            tmp_scaled[idx_split:,:]]))

# Train the model on the whole training data set using optimal regularization parameter
lr = LogisticRegression(C=C_optim, random_state=17, solver='liblinear').fit(X_train, y_train)

# Make a prediction for the test set
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the submission file
write_to_submission_file(y_test, 'm_h_s_dw_sh_t1.csv')