from pathlib import Path



import numpy as np

import pandas as pd



RANDOM_STATE = 42
### Features taken from  https://www.kaggle.com/labdmitriy/baseline-linear



### consider upvoting it. 
train_df = pd.read_csv("../input/google-quest-challenge/train.csv")

test_df = pd.read_csv("../input/google-quest-challenge/test.csv")
train_df['question_title_word_len'] = train_df['question_title'].str.split().str.len()

test_df['question_title_word_len'] = test_df['question_title'].str.split().str.len()



train_df['question_body_word_len'] = train_df['question_body'].str.split().str.len()

test_df['question_body_word_len'] = test_df['question_body'].str.split().str.len()



train_df['answer_word_len'] = train_df['answer'].str.split().str.len()

test_df['answer_word_len'] = test_df['answer'].str.split().str.len()
target_cols = ['question_asker_intent_understanding', 'question_body_critical', 

               'question_conversational', 'question_expect_short_answer', 

               'question_fact_seeking', 'question_has_commonly_accepted_answer', 

               'question_interestingness_others', 'question_interestingness_self', 

               'question_multi_intent', 'question_not_really_a_question', 

               'question_opinion_seeking', 'question_type_choice', 

               'question_type_compare', 'question_type_consequence', 

               'question_type_definition', 'question_type_entity', 

               'question_type_instructions', 'question_type_procedure', 

               'question_type_reason_explanation', 'question_type_spelling', 

               'question_well_written', 'answer_helpful', 

               'answer_level_of_information', 'answer_plausible', 

               'answer_relevance', 'answer_satisfaction', 

               'answer_type_instructions', 'answer_type_procedure', 

               'answer_type_reason_explanation', 'answer_well_written']



cols = train_df.loc[:, ~train_df.columns.isin(target_cols)].columns.tolist()
from scipy import stats

import category_encoders as ce



from sklearn.base import clone

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, RobustScaler, KBinsDiscretizer, QuantileTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.multioutput import MultiOutputRegressor

from sklearn.impute import SimpleImputer

from sklearn.metrics import make_scorer



from sklearn.linear_model import Ridge

from sklearn.svm import LinearSVR
def spearman_corr(y_true, y_pred):

        if np.ndim(y_pred) == 2:

            corr = np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])

        else:

            corr = stats.spearmanr(y_true, y_pred)[0]

        return corr

    

custom_scorer = make_scorer(spearman_corr, greater_is_better=True)
X = train_df[cols]

y = train_df[target_cols].values



X.shape, y.shape
title_col = 'question_title'

title_transformer = Pipeline([

    ('tfidf', TfidfVectorizer())

])



body_col = 'question_body'

body_transformer = Pipeline([

    ('tfidf', TfidfVectorizer())

])



num_cols = [

    'question_title_word_len', 

    'question_body_word_len', 

    'answer_word_len', 

]

num_transformer = Pipeline([

    ('impute', SimpleImputer(strategy='constant', fill_value=0)),

    ('scale', PowerTransformer(method='yeo-johnson'))

])





cat_transformer = Pipeline([

    ('impute', SimpleImputer(strategy='constant', fill_value='')),

    ('encode', OneHotEncoder(handle_unknown='ignore'))

])





preprocessor = ColumnTransformer(

    transformers = [

        ('title', title_transformer, title_col),

        ('body', body_transformer, body_col),

        ('num', num_transformer, num_cols)

    ]

)



pipeline = Pipeline([

    ('preprocessor', preprocessor),

    ('estimator', LinearSVR())

])
preprocessor.fit_transform(X, y)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)



param_grid = {

    'estimator': [

        Ridge(random_state=RANDOM_STATE),

    ],

    'estimator__alpha': [20],

    

    'preprocessor__title__tfidf__lowercase': [False],

    'preprocessor__title__tfidf__max_df': [0.3],

    'preprocessor__title__tfidf__min_df': [1],

    'preprocessor__title__tfidf__binary': [True],

    'preprocessor__title__tfidf__use_idf': [True],

    'preprocessor__title__tfidf__smooth_idf': [False],

    'preprocessor__title__tfidf__sublinear_tf': [False],

    'preprocessor__title__tfidf__ngram_range': [(1, 1)], 

    'preprocessor__title__tfidf__stop_words': [None],

    'preprocessor__title__tfidf__token_pattern': ['(?u)\\b\\w+\\b'],

    

    'preprocessor__body__tfidf__lowercase': [False],

    'preprocessor__body__tfidf__max_df': [0.3],

    'preprocessor__body__tfidf__min_df': [1],

    'preprocessor__body__tfidf__binary': [True],

    'preprocessor__body__tfidf__use_idf': [False],

    'preprocessor__body__tfidf__smooth_idf': [False],

    'preprocessor__body__tfidf__sublinear_tf': [False],

    'preprocessor__body__tfidf__ngram_range': [(1, 1)],

    'preprocessor__body__tfidf__stop_words': [None],

    'preprocessor__body__tfidf__token_pattern': ['(?u)\\b\\w+\\b'],



    'preprocessor__num__impute__strategy': ['constant'],

    'preprocessor__num__scale': [PowerTransformer()],

    

   # 'preprocessor__cat__impute__strategy': ['constant'],

   # 'preprocessor__cat__encode': [ce.BackwardDifferenceEncoder()],

    

}



grid_search = GridSearchCV(pipeline, param_grid, scoring=custom_scorer, 

                           cv=cv, n_jobs=-1, refit=True, return_train_score=True, verbose=2)



grid_search.fit(X, y)
best_estimator = clone(grid_search.best_estimator_)

best_estimator.fit(X, y)
X_test = test_df[cols]



y_pred_test = best_estimator.predict(X_test)

y_pred_test.shape
submission_df = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')

submission_df[target_cols] = y_pred_test

submission_df
sub_file_name = 'submission.csv'

submission_df.to_csv(sub_file_name, index=False)