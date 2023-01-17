

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")
target_cols = ['question_asker_intent_understanding',

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
train.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
tfidf = TfidfVectorizer(ngram_range=(1, 3))

tsvd = TruncatedSVD(n_components = 50)

question_title = tfidf.fit_transform(train["question_title"].values)

question_title_test = tfidf.transform(test["question_title"].values)

question_title = tsvd.fit_transform(question_title)

question_title_test = tsvd.transform(question_title_test)



question_body = tfidf.fit_transform(train["question_body"].values)

question_body_test = tfidf.transform(test["question_body"].values)

question_body = tsvd.fit_transform(question_body)

question_body_test = tsvd.transform(question_body_test)



answer = tfidf.fit_transform(train["answer"].values)

answer_test = tfidf.transform(test["answer"].values)

answer = tsvd.fit_transform(answer)

answer_test = tsvd.transform(answer_test)
train["len_user_name"]= train.question_user_name.apply(lambda x : len(x.split()))

test["len_user_name"]= test.question_user_name.apply(lambda x : len(x.split()))



#train["len_user_name"]= train.question_user_name.apply(lambda x : 1 if len(x)<5 else 0)

#test["len_user_name"]= test.question_user_name.apply(lambda x : 1 if len(x)<5 else 0)
#train["cat_host"]= train["category"]+train["host"]+str(train["len_user_name"])

#test["cat_host"]= test["category"]+test["host"]+str(test["len_user_name"])





category_means_map = train.groupby("len_user_name")[target_cols].mean().T.to_dict()

category_te = train["len_user_name"].map(category_means_map).apply(pd.Series)

category_te_test = test["len_user_name"].map(category_means_map).apply(pd.Series)
train_features = np.concatenate([question_title, question_body, answer#, category_te.values

                                ], axis = 1)

test_features = np.concatenate([question_title_test, question_body_test, answer_test#, category_te_test.values

                               ], axis = 1)
from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn.model_selection import KFold

from keras.callbacks.callbacks import EarlyStopping

from scipy.stats import spearmanr



num_folds = 10

fold_scores = []

kf = KFold(n_splits = num_folds, shuffle = True, random_state = 42)

test_preds = np.zeros((len(test_features), len(target_cols)))

for train_index, val_index in kf.split(train_features):

    train_X = train_features[train_index, :]

    train_y = train[target_cols].iloc[train_index]

    

    val_X = train_features[val_index, :]

    val_y = train[target_cols].iloc[val_index]

    

    model = Sequential([

        Dense(128, input_shape=(train_features.shape[1],)),

        Activation('relu'),

        Dense(64),

        Activation('relu'),

        Dense(len(target_cols)),

        Activation('sigmoid'),

    ])

    

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    model.compile(optimizer='adam',

                  loss='binary_crossentropy')

    

    model.fit(train_X, train_y, epochs = 50, validation_data=(val_X, val_y), callbacks = [es])

    preds = model.predict(val_X)

    overall_score = 0

    for col_index, col in enumerate(target_cols):

        overall_score += spearmanr(preds[:, col_index], val_y[col].values).correlation/len(target_cols)

        print(col, spearmanr(preds[:, col_index], val_y[col].values).correlation)

    fold_scores.append(overall_score)



    test_preds += model.predict(test_features)/num_folds

    

print(fold_scores)
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
sub.shape
for col_index, col in enumerate(target_cols):

    sub[col] = test_preds[:, col_index]
sub.to_csv("submission.csv", index = False)