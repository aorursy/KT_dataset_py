import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

data_paths = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_paths[filename] = os.path.join(dirname, filename)

        # print(os.path.join(dirname, filename))

        

train_df = pd.read_csv(data_paths['train.csv'])

test_df = pd.read_csv(data_paths['test.csv'])

sub_df = pd.read_csv(data_paths['sample_submission.csv'])

print('Train data shape:', train_df.shape)

print('Columns in Train:', train_df.columns)
train_df.sample(5, random_state = 1)
for i in [34, 55345, 124786]:

    display(train_df.loc[i, 'comment_text'])
comment_lens = train_df['comment_text'].str.len()

print('Central Tendencies on lengths of comment_text\n', comment_lens.describe())

ax = comment_lens.hist()
import matplotlib.pyplot as plt

import seaborn as sns



drop_col = ['id', 'is_clean']  # columns not neccessary - can be dropped

text_col = ['comment_text']  # text feature

label_col = [col for col in train_df.columns if col not in text_col + drop_col] # target variables



labels_per_comment = train_df[label_col].sum(axis = 1) # clac no.of labels for each comment



# add a new column to indicate if a comment is toxic (bad) or not (clean).

train_df['is_clean'] = 0

train_df.loc[labels_per_comment == 0, 'is_clean'] = 1

# train_df['is_clean'].value_counts()



print("Total Clean comments (All 0's in a target row) in train:",len(train_df[train_df['is_clean'] == 1]))

print("Total unclean/bad comments (atleast one 1 in a target row)in train:",len(train_df[train_df['is_clean'] != 1]))

print("Total label tags (total counts of 1's in target columns):",train_df[label_col].sum().sum())
tags_count = labels_per_comment.value_counts()



# plotting the label counts

plt.figure(figsize=(8,4))

ax = sns.barplot(tags_count.index, tags_count.values, alpha=0.7)

plt.title("Tags Counts v/s Occurences in Train Data")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('# Tag Count', fontsize=12)



#adding the text labels

rects = ax.patches

labels = tags_count.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 10, 

            label, ha='center', va='bottom')



plt.show()
label_counts = train_df[label_col].sum()



# plotting the label counts

plt.figure(figsize=(8,4))

ax = sns.barplot(label_counts.index, label_counts.values, alpha=0.7)

plt.title("Counts Per Class")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Label', fontsize=12)



#adding the text labels

rects = ax.patches

labels = label_counts.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 10, 

            label, ha='center', va='bottom')



plt.show()
import random

for label in label_col:

    label_df = train_df[train_df[label]==1].reset_index(drop = 1)

    print('\n' + label + ' - comment sample :')

    print(label_df.loc[random.randint(0, len(label_df)-1), 'comment_text'])

    print('\n' + '-'*50)
import re



#Total chars:

train_df['total_len'] = train_df['comment_text'].apply(len)

test_df['total_len'] = test_df['comment_text'].apply(len)



#Sentence count in comment: '\n' is split & count number of sentences in each comment

train_df['sent_count'] = train_df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

test_df['sent_count'] = test_df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)



#Word count in each comment:

train_df['word_count'] = train_df["comment_text"].apply(lambda x: len(str(x).split()))

test_df['word_count'] = test_df["comment_text"].apply(lambda x: len(str(x).split()))





plt.figure(figsize=(18,6))

plt.suptitle("Are longer comments more toxic?",fontsize=18)

plt.tight_layout()



# total lengths (characters)

plt.subplot(131)

ax=sns.kdeplot(train_df[train_df.is_clean == 0].total_len, label="UnClean",shade=True,color='r')

ax=sns.kdeplot(train_df[train_df.is_clean == 1].total_len, label="Clean")

plt.legend()

plt.ylabel('Number of occurances', fontsize=12)

plt.xlabel('# of Chars', fontsize=12)

# plt.title("# Chars v/s Toxicity", fontsize=12)



# words

plt.subplot(132)

ax=sns.kdeplot(train_df[train_df.is_clean == 0].word_count, label="UnClean",shade=True,color='r')

ax=sns.kdeplot(train_df[train_df.is_clean == 1].word_count, label="Clean")

plt.legend()

plt.xlabel('# of Words', fontsize=12)

# plt.title("# Words v/s comment Toxicity", fontsize=12)



## sentences

plt.subplot(133)

ax=sns.kdeplot(train_df[train_df.is_clean == 0].sent_count, label="UnClean",shade=True,color='r')

ax=sns.kdeplot(train_df[train_df.is_clean == 1].sent_count, label="Clean")

plt.legend()

plt.xlabel('# of Sentences', fontsize=12)

# plt.title("# Sentences v/s comment Toxicity", fontsize=12)



plt.show()
import string



#Captial letters:

train_df['capitals'] = train_df['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()))

test_df['capitals'] = test_df['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()))



# #Captials ratio:

# train_df['capitals_percent'] = train_df['capitals']/train_df['total_len'] * 100

# test_df['capitals_percent'] = test_df['capitals']/train_df['total_len'] * 100



# punct count:

train_df['punct_count'] = train_df['comment_text'].apply(lambda x: sum(1 for c in x if c in string.punctuation))

test_df['punct_count'] = test_df['comment_text'].apply(lambda x: sum(1 for c in x if c in string.punctuation))



# smilies:

smilies = (':-)', ':)', ';-)', ';)')

train_df['smilies_count'] = train_df['comment_text'].apply(lambda comment: sum(comment.count(s) for s in smilies))

test_df['smilies_count'] = test_df['comment_text'].apply(lambda comment: sum(comment.count(s) for s in smilies))



#----------plotting------------



plt.figure(figsize=(18,6))

plt.suptitle("did Presence of special characters vary with Toxicity ?\n",fontsize=18)

plt.tight_layout()



# words

plt.subplot(131)

ax=sns.kdeplot(train_df[train_df.is_clean == 0].capitals, label="UnClean",shade=True,color='r')

ax=sns.kdeplot(train_df[train_df.is_clean == 1].capitals, label="Clean")

plt.legend()

plt.ylabel('Number of occurances', fontsize=12)

plt.xlabel('# Capital letters', fontsize=12)

# plt.title("# Captials v/s Toxicity", fontsize=12)



# words

plt.subplot(132)

ax=sns.kdeplot(train_df[train_df.is_clean == 0].punct_count, label="UnClean",shade=True,color='r')

ax=sns.kdeplot(train_df[train_df.is_clean == 1].punct_count, label="Clean")

plt.legend()

plt.xlabel('# of Punctuations', fontsize=12)

# plt.title("#Punctuations v/s comment Toxicity", fontsize=12)



## sentences

plt.subplot(133)

ax=sns.kdeplot(train_df[train_df.is_clean == 0].smilies_count, label="UnClean",shade=True,color='r')

ax=sns.kdeplot(train_df[train_df.is_clean == 1].smilies_count, label="Clean")

plt.legend()

plt.xlabel('# of Smilies', fontsize=12)

# plt.title("#Smilies v/s comment Toxicity", fontsize=12)



plt.show()
#Unique word count:

train_df['unique_word_count'] = train_df["comment_text"].apply(lambda x: len(set(str(x).split())))

test_df['unique_word_count'] = test_df["comment_text"].apply(lambda x: len(set(str(x).split())))



#Unique ratio:

train_df['unique_word_percent'] = train_df['unique_word_count']/train_df['word_count'] * 100

test_df['unique_word_percent'] = test_df['unique_word_count']/train_df['word_count'] * 100



#----------plotting------------



# comments with unique word count percentage < 25%...they can be spam/referal links/marketing links



plt.figure(figsize=(15,5))

plt.suptitle("Comments with less-unique-words(spam) are more toxic?",fontsize = 18)



plt.subplot(121)

plt.title("% of unique words in comments")

ax=sns.kdeplot(train_df[train_df.is_clean == 0].unique_word_percent, label="UnClean",shade=True,color='r')

ax=sns.kdeplot(train_df[train_df.is_clean == 1].unique_word_percent, label="Clean")

plt.legend()

plt.ylabel('Number of occurances', fontsize=12)

plt.xlabel('Percent unique words', fontsize=12)



plt.subplot(122)

sns.violinplot(y = 'unique_word_count',x='is_clean', data = train_df[train_df['unique_word_percent'] < 25], 

               split=True,inner="quart")

plt.xlabel('is_Clean', fontsize=12)

plt.ylabel('# of words', fontsize=12)

plt.title("# unique words v/s Toxicity")

plt.show()



# train_df[train_df['word_unique_percent'] < 25]
## lets have a look how clean & unclean spam comment looks like



print("Clean Spam example:")

print(train_df[train_df['unique_word_percent'] < 10][train_df['is_clean'] == 1].comment_text.iloc[3])

print('-'*50)

print("Toxic Spam example:")

print(train_df[train_df['unique_word_percent'] < 10][train_df['is_clean'] == 0].comment_text.iloc[25])
train_df.to_csv('train_feateng.csv', index = None)

test_df.to_csv('test_feateng.csv', index = None)
# from nltk.corpus import stopwords

# from nltk import pos_tag

# from nltk.stem.wordnet import WordNetLemmatizer 

# from nltk.tokenize import word_tokenize # Tweet tokenizer does not split at apostophes which is what we want

# from nltk.tokenize import TweetTokenizer





# lemma = WordNetLemmatizer()

# tokenizer=TweetTokenizer()

# eng_stopwords = list(stopwords.words('english'))



# def simple_preprocess(comment):

#     """

#     This function receives comments and returns clean word-list

#     """

#     #Convert to lower case 

#     comment=comment.lower()

#     #remove \n

#     comment=re.sub("\\n","",comment)



#     #Split the sentences into words and lemmatize

#     words=tokenizer.tokenize(comment)

#     words=[lemma.lemmatize(word, "v") for word in words]

#     words = [w for w in words if not w in eng_stopwords]

    

#     clean_sent=" ".join(words)

    

#     return(clean_sent)



# train_df['comment_text'] = train_df['comment_text'].apply(lambda x: simple_preprocess(x))

# test_df['comment_text'] = test_df['comment_text'].apply(lambda x: simple_preprocess(x))
def get_topn_tfidf_feat_byClass(X_tfidf, y_train, feature_names, labels, topn):

    

    feat_imp_dfs = {}

    

    for label in labels:

        # get indices of rows where label is true

        label_ids = y_train.index[y_train[label] == 1]

        # get subset of rows

        label_rows = X_tfidf[label_ids].toarray()

        # calc mean feature importance

        feat_imp = label_rows.mean(axis = 0)

        # sort by column dimension and get topn feature indices

        topn_ids = np.argsort(feat_imp)[::-1][:topn]

        # combine tfidf value with feature name

        topn_features = [(feature_names[i], feat_imp[i]) for i in topn_ids]

        # df

        topn_df = pd.DataFrame(topn_features, columns = ['word_feature', 'tfidf_value'])

        # save 

        feat_imp_dfs[label] = topn_df

    return feat_imp_dfs
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from nltk import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize # Tweet tokenizer does not split at apostophes which is what we want

from nltk.tokenize import TweetTokenizer





tfidf = TfidfVectorizer(ngram_range = (1,1), min_df = 100, 

                        strip_accents='unicode', analyzer='word',

                        use_idf=1,smooth_idf=1,sublinear_tf=1,

                        stop_words = 'english')

X_unigrams = tfidf.fit_transform(train_df['comment_text'])

X_unigrams.shape, len(tfidf.get_feature_names())





feature_names = np.array(tfidf.get_feature_names())

imp_dfs = get_topn_tfidf_feat_byClass(X_unigrams, train_df, feature_names, label_col, topn = 10)



plt.figure(figsize=(15,10))



for i, label in enumerate(label_col):

    plt.subplot(3, 2, i + 1)

    sns.barplot(imp_dfs[label].word_feature[:10], imp_dfs[label].tfidf_value[:10], alpha = 0.8)

    plt.title("Important UniGrams for the class:{}".format(label))

    plt.tight_layout()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range = (2,2), min_df = 100, 

                        strip_accents='unicode', analyzer='word',

                        use_idf=1,smooth_idf=1,sublinear_tf=1,

                        stop_words = 'english')

X_bigrams = tfidf.fit_transform(train_df['comment_text'])

X_bigrams.shape, len(tfidf.get_feature_names())



feature_names = np.array(tfidf.get_feature_names())

imp_dfs = get_topn_tfidf_feat_byClass(X_bigrams, train_df, feature_names, label_col, topn = 10)



plt.figure(figsize=(15,12))



for i, label in enumerate(label_col):

    plt.subplot(3, 2, i + 1)

    by_class = sns.barplot(imp_dfs[label].word_feature[:10], imp_dfs[label].tfidf_value[:10], alpha = 0.8)

    plt.title("Important BiGrams for the class:{}".format(label))

    for item in by_class.get_xticklabels():

        item.set_rotation(45)

    plt.tight_layout()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_df['comment_text'], 

                                                  train_df[label_col], test_size=0.2, random_state=2019)

X_test = test_df['comment_text']

print('Data points in train data after splitting:', len(X_train))

print('Data points in valiadtion data:', len(X_val))

print('Data points in test data:', len(X_test))
from sklearn.metrics import log_loss, roc_auc_score

y_val_naive1 = np.random.rand(y_val.shape[0], y_val.shape[1])

print('Naive Baseline:', 'Random Guessing')

print('ROC-AUC score :', roc_auc_score(y_val, y_val_naive1))

print('Log Loss:', log_loss(y_val, y_val_naive1))
y_val_naive2 = np.zeros(y_val.shape)

y_val_naive2[:] = 0.5

print('Naive Baseline:', 'Random Guessing')

print('ROC-AUC score :', roc_auc_score(y_val, y_val_naive2))

print('Log Loss:', log_loss(y_val, y_val_naive2))
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_df['comment_text'], 

                                                  train_df[label_col], test_size=0.2, random_state=2019)

X_test = test_df['comment_text']



tfidf = TfidfVectorizer(ngram_range = (1,2), min_df = 9, strip_accents='unicode', analyzer='word',

                        use_idf=1, smooth_idf=1, sublinear_tf=1,stop_words = 'english')

X_train_tf = tfidf.fit_transform(X_train)

X_val_tf = tfidf.transform(X_val)

X_test_tf = tfidf.transform(X_test)

feature_names = tfidf.get_feature_names()



print('Final Data dimensions after transformations:', X_train_tf.shape, y_train.shape, X_val_tf.shape, y_val.shape)
from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import MultinomialNB

model = OneVsRestClassifier(MultinomialNB(), n_jobs = -1)

model.fit(X_train_tf, y_train)

print('model: Naive Bayes')

print('mean ROC-AUC on train set:', roc_auc_score(y_train, model.predict_proba(X_train_tf)))

y_pred_nb = model.predict_proba(X_val_tf)

print('mean ROC-AUC on validation set:', roc_auc_score(y_val, y_pred_nb))
model = MultinomialNB()



train_rocs = []

valid_rocs = []



preds_train = np.zeros(y_train.shape)

preds_valid = np.zeros(y_val.shape)

preds_test = np.zeros((len(test_df), len(label_col)))

print('model: Naive Bayes')

for i, label_name in enumerate(label_col):

    print('\nClass:= '+label_name)

    # fit

    model.fit(X_train_tf,y_train[label_name])

    

    # train

    preds_train[:,i] = model.predict_proba(X_train_tf)[:,1]

    train_roc_class = roc_auc_score(y_train[label_name],preds_train[:,i])

    print('Train ROC AUC:', train_roc_class)

    train_rocs.append(train_roc_class)



    # valid

    preds_valid[:,i] = model.predict_proba(X_val_tf)[:,1]

    valid_roc_class = roc_auc_score(y_val[label_name],preds_valid[:,i])

    print('Valid ROC AUC:', valid_roc_class)

    valid_rocs.append(valid_roc_class)

    

    # test predictions

    preds_test[:,i] = model.predict_proba(X_test_tf)[:,1]

    

print('\nmean column-wise ROC AUC on Train data: ', np.mean(train_rocs))

print('mean column-wise ROC AUC on Val data:', np.mean(valid_rocs))
from sklearn.linear_model import LogisticRegression

model = OneVsRestClassifier(LogisticRegression(), n_jobs = -1)

model.fit(X_train_tf, y_train)

print('model: Logistic Regression')

print('mean ROC-AUC on train set:', roc_auc_score(y_train, model.predict_proba(X_train_tf)))

y_pred_log = model.predict_proba(X_val_tf)

print('mean ROC-AUC on validation set:', roc_auc_score(y_val, y_pred_log))
from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV



model = LinearSVC()



train_rocs = []

valid_rocs = []



preds_train = np.zeros(y_train.shape)

preds_valid = np.zeros(y_val.shape)

preds_test = np.zeros((len(test_df), len(label_col)))

print('model: Linear SVM')

for i, label_name in enumerate(label_col):

    print('\nClass:= '+label_name)

    

    # fit

    model.fit(X_train_tf,y_train[label_name])

    

    # calibration classifier fit

    model = CalibratedClassifierCV(model, cv = 'prefit')

    model.fit(X_train_tf, y_train[label_name])

    

    # train

    preds_train[:,i] = model.predict_proba(X_train_tf)[:,1]

    train_roc_class = roc_auc_score(y_train[label_name],preds_train[:,i])

    print('Train ROC AUC:', train_roc_class)

    train_rocs.append(train_roc_class)



    # valid

    preds_valid[:,i] = model.predict_proba(X_val_tf)[:,1]

    valid_roc_class = roc_auc_score(y_val[label_name],preds_valid[:,i])

    print('Valid ROC AUC:', valid_roc_class)

    valid_rocs.append(valid_roc_class)

    

    # test predictions

    preds_test[:,i] = model.predict_proba(X_test_tf)[:,1]

    

print('\nmean column-wise ROC AUC on Train data: ', np.mean(train_rocs))

print('mean column-wise ROC AUC on Val data:', np.mean(valid_rocs))
from sklearn.ensemble import RandomForestClassifier

model = OneVsRestClassifier(RandomForestClassifier(), n_jobs = -1)

print('model: Random Forest')

model.fit(X_train_tf, y_train)

print('mean ROC-AUC on train set:', roc_auc_score(y_train, model.predict_proba(X_train_tf)))

y_pred_rf = model.predict_proba(X_val_tf)

print('mean ROC-AUC on validation set:', roc_auc_score(y_val, y_pred_rf))
from lightgbm import LGBMClassifier

model = OneVsRestClassifier(LGBMClassifier(), n_jobs = -1)

print('model: Lightgbm')

model.fit(X_train_tf, y_train)

print('mean ROC-AUC on train set:', roc_auc_score(y_train, model.predict_proba(X_train_tf)))

y_pred_log = model.predict_proba(X_val_tf)

print('mean ROC-AUC on validation set:', roc_auc_score(y_val, y_pred_log))
from xgboost import XGBClassifier

model = OneVsRestClassifier(XGBClassifier(), n_jobs = -1)

print('model: XGBoost')

model.fit(X_train_tf, y_train)

print('mean ROC-AUC on train set:', roc_auc_score(y_train, model.predict_proba(X_train_tf)))

y_pred_lgb = model.predict_proba(X_val_tf)

print('mean ROC-AUC on validation set:', roc_auc_score(y_val, y_pred_lgb))
X_train_val = train_df['comment_text']

y_train_val = train_df[label_col]



X_train_val = tfidf.fit_transform(train_df['comment_text'])

X_test = tfidf.transform(test_df['comment_text'])





model = OneVsRestClassifier(LogisticRegression(), n_jobs = -1)

model.fit(X_train_val, y_train_val)

print('model: Logistic Regression')

print('mean ROC-AUC on train set:', roc_auc_score(y_train_val, model.predict_proba(X_train_val)))

y_test_pred = model.predict_proba(X_test)
## making a submission file

sub_df.iloc[:,1:] = y_test_pred

sub_df.head()

from IPython.display import FileLink

sub_df.to_csv('submission.csv', index = None)

FileLink('submission.csv')
# sorted_features = sorted(zip(model.coef_.ravel(), feature_names))
#