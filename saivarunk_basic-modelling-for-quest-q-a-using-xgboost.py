import numpy as np

import pandas as pd

import xgboost as xgb



from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
INPUT_PATH = "/kaggle/input/google-quest-challenge/"
train = pd.read_csv(INPUT_PATH + "train.csv")

test = pd.read_csv(INPUT_PATH + "test.csv")

sample_submission = pd.read_csv(INPUT_PATH + "sample_submission.csv")
print("{} observations, {} columns".format(train.shape[0], train.shape[1]))

train.head()
print("{} observations, {} columns".format(test.shape[0], test.shape[1]))

test.head()
print("{} observations, {} columns".format(sample_submission.shape[0], sample_submission.shape[1]))

sample_submission.head()
columns = train.columns

columns
target_features = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
training_features = ['question_title', 'question_body', 'answer']
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )



tsvd = TruncatedSVD(n_components = 50)



train_question_title_doc = vec.fit_transform(train['question_title'].values)

test_question_title_doc = vec.transform(test['question_title'].values)



train_question_title_doc = tsvd.fit_transform(train_question_title_doc)

test_question_title_doc = tsvd.transform(test_question_title_doc)
vec_qbody = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )



tsvd = TruncatedSVD(n_components = 50)



train_question_body_doc = vec_qbody.fit_transform(train['question_body'].values)

test_question_body_doc = vec_qbody.transform(test['question_body'].values)



train_question_body_doc = tsvd.fit_transform(train_question_body_doc)

test_question_body_doc = tsvd.transform(test_question_body_doc)
vec_answer = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )



tsvd = TruncatedSVD(n_components = 50)



train_answer_doc = vec_answer.fit_transform(train['answer'].values)

test_answer_doc = vec_answer.transform(test['answer'].values)



train_answer_doc = tsvd.fit_transform(train_answer_doc)

test_answer_doc = tsvd.transform(test_answer_doc)
X_train = np.concatenate([train_question_title_doc, train_question_body_doc, train_answer_doc], axis=1)

X_test = np.concatenate([test_question_title_doc, test_question_body_doc, test_answer_doc], axis=1)
printt(X_train.shape)

printt(X_test.shape)
def train_xbg_model(target_feature):

    xgb_model = xgb.XGBRegressor(learning_rate = 0.1, n_estimators=1000,

                           max_depth=5, min_child_weight=1,

                           gamma=0, subsample=0.8,

                           colsample_bytree=0.8, objective= "binary:logistic",  

                           nthread=-1, scale_pos_weight=1, random_state=2019, seed=2019)

    xgb_model.fit(X_train, train[target_feature])

    y_pred = xgb_model.predict(X_test)

    return y_pred
for feature in target_features:

    print("------------------------------")

    print(f"Traning for {feature}")

    sample_submission[feature] = train_xbg_model(feature)
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)