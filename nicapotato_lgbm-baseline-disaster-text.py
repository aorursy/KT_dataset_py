import time

notebookstart= time.time()



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_colwidth = 500

import os

import gc

import re

print("Data:\n",os.listdir("../input"))



# Models Packages

from sklearn import metrics

from sklearn import feature_selection

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, cross_val_score, KFold



# Gradient Boosting

import lightgbm as lgb



# Tf-Idf

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import FeatureUnion

from scipy.sparse import hstack, csr_matrix

from nltk.corpus import stopwords 



# Viz

import seaborn as sns

import matplotlib.pyplot as plt
N_ROWS = None

TEXT_COLUMN = 'text'

TARGET_COLUMN = 'target'
print("Read Data")

train_df = pd.read_csv('../input/nlp-getting-started/train.csv', nrows = N_ROWS, index_col = 'id')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv', nrows = N_ROWS, index_col = 'id')



y = train_df[TARGET_COLUMN].values

traindex = train_df.index.copy()

testdex = test_df.index.copy()



print("Train Shape: {} Rows, {} Columns".format(*train_df.shape))

display(train_df.sample(5))

print("Test Shape: {} Rows, {} Columns".format(*test_df.shape))

display(test_df.sample(5))

print('Dependent Variable Factor Ratio: ',train_df[TARGET_COLUMN].value_counts(normalize=True).to_dict())
# Join Train / Test

df = pd.concat([train_df,test_df],axis=0,sort=False)

display(df.sample(5))
print("Text Features")

df['keyword'] = df['keyword'].str.replace("%20", " ")

df['hashtags'] = df['text'].apply(lambda x: " ".join(re.findall(r"#(\w+)", x)))

df['hash_loc_key'] = df[['hashtags', 'location','keyword']].astype(str).apply(lambda x: " ".join(x), axis=1)



df['hash_loc_key'] = df["hash_loc_key"].astype(str).str.lower().fillna('missing')

df['hash_loc_key'] = df["hash_loc_key"].astype(str).str.lower().fillna('missing')



textfeats = ['hash_loc_key', 'text']

for cols in textfeats:

    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words

    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))

    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words



print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")

string_stopwords = set(stopwords.words('english'))



tfidf_para = {

#     "stop_words": string_stopwords,

#     "analyzer": 'word',

#     "token_pattern": r'\w{1,}',

#     "sublinear_tf": True,

#     "dtype": np.float32,

#     "norm": 'l2',

#     "min_df":5,

#     "max_df":.9,

#     "smooth_idf":False

}





def get_col(col_name): return lambda x: x[col_name]

vectorizer = FeatureUnion([

        ('text',TfidfVectorizer(

            ngram_range=(1, 1),

            max_features=5000,

            **tfidf_para,

            preprocessor=get_col('text'))),

        ('hash_loc_key',TfidfVectorizer(

            ngram_range=(1, 1),

            max_features=5000,

            **tfidf_para,

            preprocessor=get_col('hash_loc_key')))

    ])

    

start_vect=time.time()

vectorizer.fit(df.to_dict('records'))

tfidf_sparse = vectorizer.transform(df.to_dict('records'))

tfvocab = vectorizer.get_feature_names()

print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))
dense_vars =  [

    'text_num_words',

    'text_num_unique_words',

    'text_words_vs_unique',

    'hash_loc_key_num_words',

    'hash_loc_key_num_unique_words',

    'hash_loc_key_words_vs_unique'

]



# Dense Features Correlation Matrix

f, ax = plt.subplots(figsize=[5,5])

sns.heatmap(df[dense_vars].corr(),

            annot=False, fmt=".2f",

            cbar_kws={'label': 'Correlation Coefficient'},

            cmap="plasma",ax=ax, linewidths=.5)



ax.set_title("Dense Features Correlation Matrix")

plt.show()
print("Modeling Stage")

X = hstack([csr_matrix(df.loc[traindex,dense_vars].values),tfidf_sparse[0:traindex.shape[0]]]) # Sparse Matrix

testing = hstack([csr_matrix(df.loc[testdex,dense_vars].values),tfidf_sparse[traindex.shape[0]:]])

tfvocab = df[dense_vars].columns.tolist() + tfvocab

for shape in [X,testing]:

    print("{} Rows and {} Cols".format(*shape.shape))

print("Feature Names Length: ",len(tfvocab))
print("\nModeling Stage")

print("Light Gradient Boosting Regressor")



def lgb_f1_score(y_hat, data):

    y_true = data.get_label()

    y_hat = np.where(y_hat < 0.5, 0, 1)  

    return 'f1', metrics.f1_score(y_true, y_hat, average='macro'), True



lgbm_params = {

        "objective": "binary",

        "boosting_type": "gbdt",

        "num_threads": -1,

#         "bagging_fraction": 0.8,

#         "feature_fraction": 0.8,

#         "learning_rate": 0.05,

#         "max_depth": 9,

#         "num_leaves": 150,

#         "min_split_gain": .1,

#         "reg_alpha": .1

    }



# Training and Validation Set

modelstart = time.time()

n_folds = 4

folds = KFold(n_splits=n_folds, shuffle=True, random_state=1)

oof_preds = np.zeros(traindex.shape[0])

fold_preds = np.zeros(testdex.shape[0])



# Fit 5 Folds

modelstart = time.time()

for trn_idx, val_idx in folds.split(X):

    

    lgtrain = lgb.Dataset(X.tocsr()[trn_idx], y[trn_idx])

    lgvalid = lgb.Dataset(X.tocsr()[val_idx], y[val_idx])

    

    clf = lgb.train(

        params=lgbm_params,

        train_set=lgtrain,

        valid_sets=lgvalid,

        num_boost_round=3500, 

        early_stopping_rounds=200,

        feval=lgb_f1_score,

        verbose_eval=150

    )

    oof_preds[val_idx] = clf.predict(X.tocsr()[val_idx])

    fold_preds += clf.predict(testing) 

    print('Metric:', metrics.f1_score(

        y[val_idx], np.where(oof_preds[val_idx] < 0.5, 0, 1), average='macro'))

print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
# OOF F1 Cutoff

save_f1_opt = []

cutoff = .5

for cutoff in np.arange(.38,.62, .01):

    save_f1_opt.append([cutoff, f1_score(y, (oof_preds > cutoff).astype(int), average='macro')])

f1_pd = pd.DataFrame(save_f1_opt, columns = ['cutoff', 'f1_score'])



best_cutoff = f1_pd.loc[f1_pd['f1_score'].idxmax(),'cutoff']

print("F1 Score: {:.4f}, Optimised Cufoff: {:.2f}".format(f1_pd.loc[f1_pd['f1_score'].idxmax(),'f1_score'], best_cutoff))



f,ax = plt.subplots(1,2,figsize = [10,4])



ax[0].plot(f1_pd['cutoff'], f1_pd['f1_score'], c = 'red')

ax[0].set_ylabel("F1 Score")

ax[0].set_xlabel("Cutoff")

ax[0].axvline(x=best_cutoff, c='black')

ax[0].set_title("Macro Avg F1 Score and Cutoff on OOF")



train_df['oof_preds'] = oof_preds

train_df['error'] = train_df['target'] - train_df['oof_preds']



sns.distplot(train_df['error'], ax = ax[1])

ax[1].set_title("Classification Errors: Target - Pred Probability")

ax[1].axvline(x=.5, c='black')

ax[1].axvline(x=-.5, c='black')

plt.tight_layout(pad=1)

plt.show()
print("OOF Classification Report for Optimised Threshold: {:.3f}".format(best_cutoff))

print(classification_report(y, (oof_preds > best_cutoff).astype(int), digits = 4))

print("\nOOF Non-Optimised Cutoff (.5)")

print(classification_report(y, (oof_preds > .5).astype(int), digits = 4))



cnf_matrix = confusion_matrix(y, (oof_preds > .5).astype(int))

print("OOF Confusion Matrix")

print(cnf_matrix)

print("OOF Normalised Confusion Matrix")

print((cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]).round(3))
print("Look at False Negative")

display(train_df.sort_values(by = 'error', ascending=False).iloc[:20])



print("Look at False Positives")

display(train_df.sort_values(by = 'error', ascending=True).iloc[:20])
submission = pd.DataFrame.from_dict({

    'id': test_df.index,

    TARGET_COLUMN:  np.where((fold_preds/n_folds) < 0.5, 0, 1)

})

submission.to_csv('lgbm_submission.csv', index=False)

print(submission[TARGET_COLUMN].value_counts(normalize = True).to_dict())

submission.head()
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))