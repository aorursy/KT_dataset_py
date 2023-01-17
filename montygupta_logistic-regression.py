import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import string

from collections import Counter, defaultdict

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

from pyemd import emd
import gensim
from gensim.similarities import WmdSimilarity
from gensim.models import Word2Vec
from gensim import corpora
import gensim.downloader as api
from gensim.matutils import softcossim

from scipy.spatial.distance import cosine,cityblock,jaccard,canberra,euclidean,minkowski,braycurtis

from fuzzywuzzy import fuzz
from tqdm import tqdm_notebook

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("train.csv")
df = df.dropna(how="any").reset_index(drop=True)
df_test_total = pd.read_csv("test.csv")
#df = df.head(30)
df_test = df_test_total[:200000]
# EDA
is_dup = df['is_duplicate'].value_counts()
print (is_dup)
plt.figure(figsize=(8,4))
sns.barplot(is_dup.index, is_dup.values, alpha=0.8)
plt.ylabel('No of Occurrences', fontsize=12)
plt.xlabel('Is Duplicate', fontsize=12)
plt.show()
is_dup / is_dup.sum()
# length of the questions

df['q1_word_len'] = df['question1'].str.split().str.len()
df['q2_word_len'] = df['question2'].str.split().str.len()
df['q1_char_len'] = df['question1'].str.len()
df['q2_char_len'] = df['question2'].str.len()
df.head()
# test data

df_test['q1_word_len'] = df_test['question1'].str.split().str.len()
df_test['q2_word_len'] = df_test['question2'].str.split().str.len()
df_test['q1_char_len'] = df_test['question1'].str.len()
df_test['q2_char_len'] = df_test['question2'].str.len()
df_test.head()
# Plot of words

cnt_words = df['q1_word_len'] + df['q2_word_len']
cnt_words = cnt_words.value_counts()
plt.figure(figsize=(18,6))
sns.barplot(cnt_words.index, cnt_words.values, alpha=0.8)
plt.ylabel('No of Occurrences', fontsize=12)
plt.xlabel('No of words in the question', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
# Plot of characters

cnt_chars = df['q1_char_len'] + df['q2_char_len']
cnt_chars = cnt_chars.value_counts()
plt.figure(figsize=(18,6))
sns.barplot(cnt_chars.index, cnt_chars.values, alpha=0.8)
plt.ylabel('No of Occurrences', fontsize=12)
plt.xlabel('No of chars in the question', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
# Feature Extraction on grams
ques_stopwords = set(stopwords.words('english')) |  set(string.punctuation) 
stemmer = PorterStemmer()

def feature_extraction(row):
    que1 = str(row['question1'])
    que2 = str(row['question2'])
    out_list = []
    # get unigram features #
    unigrams_que1 = [stemmer.stem(word) for word in que1.lower().split() if word not in ques_stopwords]
    unigrams_que2 = [stemmer.stem(word) for word in que2.lower().split() if word not in ques_stopwords]
    common_unigrams_len = len(set(unigrams_que1).intersection(set(unigrams_que2)))
    common_unigrams_ratio = float(common_unigrams_len) / max(len(set(unigrams_que1).union(set(unigrams_que2))),1)
    out_list.extend([common_unigrams_len, common_unigrams_ratio])

    # get bigram features #
    bigrams_que1 = [i for i in ngrams(unigrams_que1, 2)]
    bigrams_que2 = [i for i in ngrams(unigrams_que2, 2)]
    common_bigrams_len = len(set(bigrams_que1).intersection(set(bigrams_que2)))
    common_bigrams_ratio = float(common_bigrams_len) / max(len(set(bigrams_que1).union(set(bigrams_que2))),1)
    out_list.extend([common_bigrams_len, common_bigrams_ratio])

    # get trigram features #
    trigrams_que1 = [i for i in ngrams(unigrams_que1, 3)]
    trigrams_que2 = [i for i in ngrams(unigrams_que2, 3)]
    common_trigrams_len = len(set(trigrams_que1).intersection(set(trigrams_que2)))
    common_trigrams_ratio = float(common_trigrams_len) / max(len(set(trigrams_que1).union(set(trigrams_que2))),1)
    out_list.extend([common_trigrams_len, common_trigrams_ratio])
    return out_list
# Use grams_feature_extraction to extract features 

df['common_grams'] = df.apply(lambda row: feature_extraction(row), axis=1)
columns = ['common_unigrams_len', 'common_unigrams_ratio', 
  'common_bigrams_len', 'common_bigrams_ratio', 
  'common_trigrams_len', 'common_trigrams_ratio']
df1 = pd.DataFrame(df['common_grams'].tolist(), columns=columns)
df = pd.concat([df,df1], axis=1)
df.drop('common_grams', axis=1, inplace=True) 
df
# Use grams_feature_extraction to extract features for test data

df_test['common_grams'] = df_test.apply(lambda row: feature_extraction(row), axis=1)
columns = ['common_unigrams_len', 'common_unigrams_ratio', 
  'common_bigrams_len', 'common_bigrams_ratio', 
  'common_trigrams_len', 'common_trigrams_ratio']
df1 = pd.DataFrame(df_test['common_grams'].tolist(), columns=columns)
df_test = pd.concat([df_test,df1], axis=1)
df_test.drop('common_grams', axis=1, inplace=True) 
df_test
df.to_csv('df_new', index=False, encoding='utf-8')
df_test.to_csv('df_new_test', index=False, encoding='utf-8')
df = pd.read_csv("df_new")
df_test = pd.read_csv("df_new_test")
# EXA on unigrams
cnt_srs = df['common_unigrams_len'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel("No of Occurrences", fontsize=12)
plt.xlabel("Common unigrams count", fontsize=12)
plt.show()
# EDA on the feature

plt.figure(figsize=(12,6))
sns.boxplot(x='is_duplicate', y='common_unigrams_len', data=df)
plt.xlabel('Is duplicate', fontsize=12)
plt.ylabel('Common unigrams count', fontsize=12)
plt.show()
# EDA on unigram ratio
plt.figure(figsize=(12,6))
sns.boxplot(x='is_duplicate', y='common_unigrams_ratio', data=df)
plt.xlabel("Is duplicate", fontsize=12)
plt.ylabel("Unigram common ratio")
plt.show
n = 10
sns.pairplot(df[['q1_char_len','q2_char_len','q1_word_len','q2_word_len','is_duplicate','common_unigrams_len']][0:n])                
                
col_mask=df.isnull().any(axis=0) 
#col_mask
row_mask=df.isnull().any(axis=1)
row_mask
df = df.dropna()
scaler = MinMaxScaler().fit(df[['q1_word_len','q2_word_len','q1_char_len','q2_char_len','common_unigrams_len', 'common_unigrams_ratio', 
  'common_bigrams_len', 'common_bigrams_ratio', 'common_trigrams_len', 'common_trigrams_ratio']])
X = scaler.transform(df[['q1_word_len','q2_word_len','q1_char_len','q2_char_len','common_unigrams_len', 'common_unigrams_ratio', 
  'common_bigrams_len', 'common_bigrams_ratio', 'common_trigrams_len', 'common_trigrams_ratio']])
y = df['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

clf = LogisticRegression()
grid = {
    'C': [1e-6, 1e-3, 1e0],
    'penalty': ['l1','l2']
}
cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=1, verbose=1)
cv.fit(X_train, y_train)
print(cv.best_params_)
print(cv.best_estimator_.coef_)
cv.error_score
y_pred = cv.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
print(y_pred)
print('accuracy of logistic regression classifier for train set: {:.2f}'.format(cv.score(X_test,y_test)))

# Graph based features


ques = pd.concat([df[['question1', 'question2']], \
    df_test_total[['question1', 'question2']]], axis=0).reset_index(drop='index')

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])

    
def q1_freq(row):
    return (len(q_dict[row['question1']]))


def q2_freq(row):
    return (len(q_dict[row['question2']]))


def q1_q2_intersect(row):
    return (len(
        set(q_dict[row['question1']]).intersection(
            set(q_dict[row['question2']]))))

df['q1_q2_intersect'] = df.apply(q1_q2_intersect, axis=1, raw=True)
df['q1_freq'] = df.apply(q1_freq, axis=1, raw=True)
df['q2_freq'] = df.apply(q2_freq, axis=1, raw=True)

# test data
df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)
df.info()
df_test.info()
df.to_csv('df_new', index=False, encoding='utf-8')
df_test.to_csv('df_test_new', index=False, encoding='utf-8')
df = pd.read_csv("df_new")
df_test = pd.read_csv("df_test_new")
col_mask=df.isnull().any(axis=0) 
print(col_mask)
col_test_mask=df_test.isnull().any(axis=0) 
print(col_test_mask)
df_test.head()
cnt_srs = df['q1_q2_intersect'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, np.log1p(cnt_srs.values), alpha=0.8)
plt.xlabel('Q1-Q2 neighbour instersection count', fontsize=12)
plt.ylabel('Log of Number of Occurrences', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
grouped_df = df.groupby('q1_q2_intersect')['is_duplicate'].aggregate(np.mean).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['q1_q2_intersect'].values,
             grouped_df['is_duplicate'].values, alpha=0.8)
plt.ylabel('Mean is duplicate', fontsize=12)
plt.xlabel('Q1-Q2 neighbor intersection count', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
pvt_df = df.pivot_table(index='q1_freq', columns='q2_freq', values='is_duplicate')
plt.figure(figsize=(12,12))
sns.heatmap(pvt_df)
plt.title("Mean is_duplicate across q1 and q2 fequency")
plt.show()
cols_to_use = ['q1_q2_intersect', 'q1_freq', 'is_duplicate']
temp_df = df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8,8))

#Heatmap
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Leaky variables correlation map", fontsize=15)
plt.show()

corr_mat = df[cols_to_use].corr()
corr_mat.head()
scaler = MinMaxScaler().fit(df[['q1_word_len','q2_word_len','q1_char_len','q2_char_len','common_unigrams_len', 'common_unigrams_ratio', 
  'common_bigrams_len', 'common_bigrams_ratio', 'common_trigrams_len', 'common_trigrams_ratio','q1_q2_intersect', 'q1_freq', 'q2_freq']])
X = scaler.transform(df[['q1_word_len','q2_word_len','q1_char_len','q2_char_len','common_unigrams_len', 'common_unigrams_ratio', 
  'common_bigrams_len', 'common_bigrams_ratio', 'common_trigrams_len', 'common_trigrams_ratio','q1_q2_intersect', 'q1_freq', 'q2_freq']])
y = df['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

clf = LogisticRegression()
grid = {
    'C': [1e-6, 1e-3, 1e0],
    'penalty': ['l1','l2']
}
cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=1, verbose=1)
cv.fit(X_train, y_train)
print(cv.best_params_)
print(cv.best_estimator_.coef_)
cv.error_score
y_pred = cv.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
print('accuracy of logistic regression classifier for train set: {:.2f}'.format(cv.score(X_test,y_test)))

scaler = MinMaxScaler().fit(df_test[['q1_word_len','q2_word_len','q1_char_len','q2_char_len','common_unigrams_len', 'common_unigrams_ratio', 
  'common_bigrams_len', 'common_bigrams_ratio', 'common_trigrams_len', 'common_trigrams_ratio','q1_q2_intersect', 'q1_freq', 'q2_freq']])
X_test = scaler.transform(df_test[['q1_word_len','q2_word_len','q1_char_len','q2_char_len','common_unigrams_len', 'common_unigrams_ratio', 
  'common_bigrams_len', 'common_bigrams_ratio', 'common_trigrams_len', 'common_trigrams_ratio','q1_q2_intersect', 'q1_freq', 'q2_freq']])

test_pred = cv.predict(X_test)
print(test_pred)
df_test.info()
#df.to_csv('df_new', index=False, encoding='utf-8')
df1 = pd.DataFrame(test_pred, columns=columns)
submission = pd.concat([df_test['test_id'],df1], axis=1)
submission.head()
col_mask=submission.isnull().any(axis=0) 
col_mask
is_dup = submission['is_duplicate'].value_counts()
plt.figure(figsize=(8,4))
sns.barplot(is_dup.index, is_dup.values, alpha=0.8)
plt.ylabel('No of Occurrences', fontsize=12)
plt.xlabel('Is Duplicate', fontsize=12)
plt.show()

submission.to_csv('submission.csv', index=False, encoding='utf-8')
norm_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
def wmd(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    stop_words = stopwords.words('english')
    q1 = [w for w in q1 if w not in stop_words]
    q2 = [w for w in q2 if w not in stop_words]
    return model.wmdistance(q1, q2)

def norm_wmd(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    stop_words = stopwords.words('english')
    q1 = [w for w in q1 if w not in stop_words]
    q2 = [w for w in q2 if w not in stop_words]
    return norm_model.wmdistance(q1, q2)

def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())
norm_model.init_sims(replace=True)
df['norm_wmd'] = df.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
# Advanced Features
df['fuzz_ratio'] = df.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Advanced Features
df_test['fuzz_ratio'] = df_test.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
df_test['fuzz_partial_ratio'] = df_test.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
df_test['fuzz_partial_token_set_ratio'] = df_test.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
df_test['fuzz_partial_token_sort_ratio'] = df_test.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
df_test['fuzz_token_set_ratio'] = df_test.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
df_test['fuzz_token_sort_ratio'] = df_test.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

question1_vectors = np.zeros((df.shape[0], 300))
for i, q in enumerate(tqdm_notebook(df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)
    
question2_vectors  = np.zeros((df.shape[0], 300))
for i, q in enumerate(tqdm_notebook(df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

df = df[pd.notnull(df['cosine_distance'])]
df = df[pd.notnull(df['jaccard_distance'])]
# test dataframe
question1_vectors = np.zeros((df_test.shape[0], 300))
for i, q in enumerate(tqdm_notebook(df_test.question1.values)):
    question1_vectors[i, :] = sent2vec(q)
    
question2_vectors  = np.zeros((df_test.shape[0], 300))
for i, q in enumerate(tqdm_notebook(df_test.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

df_test['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_test['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_test['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_test['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_test['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_test['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df_test['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

df_test = df_test[pd.notnull(df_test['cosine_distance'])]
df_testf = df_test[pd.notnull(df_test['jaccard_distance'])]
df.drop(['question1', 'question2'], axis=1, inplace=True)
df.info()
df_test.drop(['question1', 'question2'], axis=1, inplace=True)
df_test.info()
X = df.loc[:, df.columns != 'is_duplicate']
y = df.loc[:, df.columns == 'is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = MinMaxScaler().fit(df)
X = scaler.transform(df)
y = df['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

clf = LogisticRegression()
grid = {
    'C': [1e-6, 1e-3, 1e0],
    'penalty': ['l1','l2']
}
cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=1, verbose=1)
cv.fit(X_train, y_train)
print(cv.best_params_)
print(cv.best_estimator_.coef_)
cv.error_score
y_pred = cv.predict(X)
cf_matrix = confusion_matrix(y, y_pred)
print(cf_matrix)
y_pred = cv.predict(X)
y_pred
print('accuracy of logistic regression classifier for train set: {:.2f}'.format(cv.score(X,y)))
test_pred = cv.predict(df_test)
print(test_pred)
model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train.values.ravel()) 
prediction = model.predict(X_test)
cm = confusion_matrix(y_test, prediction)  
print(cm)  
print('Accuracy', accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))
test_pred = model.predict(df_test)
print(test_pred)

