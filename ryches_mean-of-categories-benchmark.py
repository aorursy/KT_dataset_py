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
category_means_map = train.groupby("category")[target_cols].mean().T.to_dict()
preds = train["category"].map(category_means_map).apply(pd.Series)
from scipy.stats import spearmanr
overall_score = 0

for col in target_cols:

    overall_score += spearmanr(preds[col], train[col]).correlation/len(target_cols)

    print(col, spearmanr(preds[col], train[col]).correlation)
overall_score
test_preds = test["category"].map(category_means_map).apply(pd.Series)
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
for col in target_cols:

    sub[col] = test_preds[col]
sub.to_csv("submission.csv", index = False)