import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import log_loss
%matplotlib inline

# # removing the characters other than alpha-numeric
# def get_data(filename, header=1):
#     if header==1:
#         df = pd.read_csv(filename)
#     else:
#         df = pd.read_csv(filename, header=0)
#     # For beginning, transform train['FullDescription'] to lowercase using text.lower()
#     # Then replace everything except the letters and numbers to spaces.
#     # it will facilitate the further division of the text into words.
#     df['question1'] = df['question1'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)
#     df['question2'] = df['question2'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)
    
#     return df
data = pd.read_csv("../input/train.csv")
train = data
test = pd.read_csv("../input/test.csv")
data
print (len(data))
print('Duplicate pairs: {}%'.format(data['is_duplicate'].mean()*100))
all_q = pd.Series(data['qid1'].tolist() + data['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(np.unique(all_q))))
print('Number of questions that appear multiple times: {}'.format(np.sum(all_q.value_counts() > 1)))
print (all_q[:10])
plt.figure(figsize=(12, 5))
plt.hist(all_q.value_counts(), bins=50)
# print (all_q.value_counts())
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print()

# print (all_q.value_counts())
print (data['is_duplicate'].value_counts())
print (data['is_duplicate'].value_counts()/data.shape[0])
print (data.shape)
train_qs = pd.Series(data['question1'].tolist() + data['question2'].tolist()).astype(str)
test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist()).astype(str)
dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
print (train_qs[:10])
print (dist_train[:10])
dist_train.describe()
dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))
print (dist_train[:10])
qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
math = np.mean(train_qs.apply(lambda x: '[math]' in x))
fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))

print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
print('Questions with [math] tags: {:.2f}%'.format(math * 100))
print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
print('Questions with numbers: {:.2f}%'.format(numbers * 100))
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
# print (len(stops))
data.head()
a=data['question1']
b=data['question2']
a=a[0].split()
b=b[0].split()
c=[w for w in a if w in b]
d=[w for w in b if w in a]
print (a,b)
print (c,d)
# # Stemming
# from stemming.porter2 import stem
# train_qs_stem = [' '.join([stem(word) for word in sentence.split(" ")]) for sentence in train_qs]
# test_qs_stem = [' '.join([stem(word) for word in sentence.split(" ")]) for sentence in test_qs]
# words common in both the quesitons
from stemming.porter2 import stem
import re

regex = re.compile('[^a-zA-Z0-9]')

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
#     print (len(shared_words_in_q1) ,len(shared_words_in_q2))
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

# removing characters other than alphanumeric
# def word_match_share(row):
#     res = 0
#     q1 = regex.sub(' ', str(row['question1']).lower()).split()
#     q2 = regex.sub(' ', str(row['question2']).lower()).split()
# #     q1 = str(row['question1']).lower().split()
# #     q2 = str(row['question1']).lower().split()
#     q1 = [x for x in q1 if x not in stops]
#     q2 = [x for x in q2 if x not in stops]
    
#     if len(q1)==0 or len(q2)==0:
#         return 0
#     for word in q1:
#         if word in q2:
#             res += 1
#     for word in q2:
#         if word in q1:
#             res += 1
#     return res/(len(q1)+len(q2))


# using stemming
# def word_match_share(row):
#     '''(Using stemming)
#     Takes input as the row of training dataset
#     if both questions contain only stopwords
#         return 0
#     else
#         return the ratio of non-stopwords common in both questions by total number of words
#     '''
#     q1words = {}
#     q2words = {}
#     for word in str(row['question1']).lower().split():
#         if word not in stops:
#             word = stem(word)
#             q1words[word] = 1
#     for word in str(row['question2']).lower().split():
#         if word not in stops:
#             word = stem(word)
#             q2words[word] = 1
#     if len(q1words)==0 or len(q2words)==0:
#         return 0

#     shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
#     shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
#     res = (len(shared_words_in_q1)+len(shared_words_in_q2))/(len(q1words)+len(q2words))
#     return res
plt.figure(figsize=(15, 5))
train_word_match = data.apply(word_match_share, axis=1, raw=True)
print (train_word_match[:10])
# print (data['is_duplicate'] == 0)
plt.hist(train_word_match[data['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
plt.hist(train_word_match[data['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)
from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=50000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
# print (" ".join(train_qs))
print (words[:10])
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
a=weights.items()
# print (weights)
# print (weights)
print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
print('\nLeast common words and weights: ')
(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

# # using stemming
# def tfidf_word_match_share(row):
#     '''assigns weights to words in the quetions based on tfidf value
#     if the words are only stopwords
#         return 0
#     '''
#     q1words = {}
#     q2words = {}
#     for word in str(row['question1']).lower().split():
#         if word not in stops:
#             word = stem(word)
#             q1words[word] = 1
#     for word in str(row['question2']).lower().split():
#         if word not in stops:
#             word = stem(word)
#             q2words[word] = 1
#     if len(q1words)==0 or len(q2words)==0:
#         return 0

#     shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words]+[weights.get(w, 0) for w in q2words.keys() if w in q1words]
#     total_weights = [weights.get(w, 0) for w in q1words]+[weights.get(w, 0) for w in q2words]

#     res = np.sum(shared_weights)/np.sum(total_weights)
#     return res
plt.figure(figsize=(15, 5))
df_train=data
df_test=test
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)
plt.show()
from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))
print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))
# # common letters in that
# ->>>>>>>>>>> increased the log loss
# def letter_match_share(row):
#     q1words = {}
#     q2words = {}
#     for word in str(row['question1']).lower().split():
#         if word not in stops:
#             q1words[word] = 1
#     for word in str(row['question2']).lower().split():
#         if word not in stops:
#             q2words[word] = 1
#     if len(q1words) == 0 or len(q2words) == 0:
#         # The computer-generated chaff includes a few questions that are nothing but stopwords
#         return 0
#     d1=defaultdict(int)
#     d2=defaultdict(int)
#     l1=0
#     l2=0
#     for w in q1words:
#         for l in w:
#             d1[l]+=1
#     for w in q2words:
#         for l in w:
#             d2[l]+=1
#     for l in d1:
#         l1+=d1[l]
#     for l in d2:
#         l2+=d2[l]
#     com_l=0
#     for l in d1:
#         com_l=min(d1[l],d2[l])
# #     print (len(shared_words_in_q1) ,len(shared_words_in_q2))
#     R = (2*com_l)/(l1+l2)
#     return R


# plt.figure(figsize=(15, 5))
# train_letter_match = data.apply(letter_match_share, axis=1, raw=True)
# print (train_letter_match[:10])
# # print (data['is_duplicate'] == 0)
# plt.hist(train_letter_match[data['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
# plt.hist(train_letter_match[data['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
# plt.legend()
# plt.title('Label distribution over letter_match_share', fontsize=15)
# plt.xlabel('letter_match_share', fontsize=15)

# x_train=pd.DataFrame()
# print (x_train.head())
# print (x_train)
x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
# x_train['letter_match'] = train_letter_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
# x_test['letter_match'] = df_test.apply(letter_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
# print (x_train.shape)
y_train = df_train['is_duplicate']
# y= np.array(np.unique(y_train, return_counts=True)).T
# print (y)
x_train.head()
# adding 1->2,3,4  4->2 3 6  intersection = 2,3

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
ques = pd.concat([train[['question1', 'question2']],test[['question1', 'question2']]], axis=0).reset_index(drop='index')

print (ques.shape)


q_dict = defaultdict(set)
# print (q_dict)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

x_train['q1_q2_intersect'] = train.apply(q1_q2_intersect, axis=1, raw=True)
x_test['q1_q2_intersect'] = test.apply(q1_q2_intersect, axis=1, raw=True)

plt.figure(figsize=(15, 5))
# print (x_train.head())
plt.hist((x_train.q1_q2_intersect[df_train['is_duplicate'] == 0]/60).fillna(0), bins=50, normed=True, label='Not Duplicate')
plt.hist((x_train.q1_q2_intersect[df_train['is_duplicate'] == 1]/60).fillna(0), bins=50, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over intersectiom', fontsize=15)
plt.xlabel('intersection', fontsize=15)
plt.show()
# adding freq of each question

train_orig =  pd.read_csv('../input/train.csv', header=0)
test_orig =  pd.read_csv('../input/test.csv', header=0)

# tic0=timeit.default_timer()
df1 = train_orig[['question1']].copy()
df2 = train_orig[['question2']].copy()
df1_test = test_orig[['question1']].copy()
df2_test = test_orig[['question2']].copy()

df2.rename(columns = {'question2':'question1'},inplace=True)
df2_test.rename(columns = {'question2':'question1'},inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
#train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
train_questions.drop_duplicates(subset = ['question1'],inplace=True)

train_questions.reset_index(inplace=True,drop=True)
questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
train_cp = train_orig.copy()
test_cp = test_orig.copy()
train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

test_cp['is_duplicate'] = -1
test_cp.rename(columns={'test_id':'id'},inplace=True)
comb = pd.concat([train_cp,test_cp])

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()

def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
#map to frequency space
comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

train_comb = comb[comb['is_duplicate'] >= 0][['q1_freq','q2_freq']]
test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]


print (x_train.shape,train_comb.shape )                                          
x_train['q1_freq']=train_comb['q1_freq']  
x_train['q2_freq']=train_comb['q2_freq']

                                         
x_test['q1_freq']=test_comb['q1_freq']  
x_test['q2_freq']=test_comb['q2_freq']
x_train.head()
print(x_test['q1_freq'].describe())

plt.figure(figsize=(15, 5))
# print (x_train.head())
plt.hist((x_train.q1_freq[df_train['is_duplicate'] == 0]/60).fillna(0), bins=50, normed=True, label='Not Duplicate')
plt.hist((x_train.q1_freq[df_train['is_duplicate'] == 1]/60).fillna(0), bins=50, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over freq', fontsize=15)
plt.xlabel('freq', fontsize=15)
plt.show()
pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]
print (pos_train.shape,neg_train.shape)
# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.2
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))
print (pos_train.shape,neg_train.shape)
x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

print (x_train.shape,len(y_train))
x_train.head()
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=22)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.3
params['max_depth'] = 5
params['silent'] = 1

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=10)

# model =  xgb.XGBClassifier()
# learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1]
# param_grid = dict(learning_rate=learning_rate)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=22)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
# grid_result = grid_search.fit(x_train, y_train)
# [399]	train-logloss:0.361613	valid-logloss:0.364888 in ->>1

# not rebalancing [399]	train-logloss:0.459056	valid-logloss:0.458105
# [399]	train-logloss:0.339633	valid-logloss:0.339191

# using magic feature [399]	train-logloss:0.263183	valid-logloss:0.264937
#->https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
# using intersection vala thing
# [399]	train-logloss:0.212057	valid-logloss:0.214368
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)
pp=p_test[p_test>=0.5]
print (pp.shape[0]/p_test.shape[0])
sub = pd.DataFrame()
# print (df_test.head())
sub['id'] = df_test['id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)
sub.head()