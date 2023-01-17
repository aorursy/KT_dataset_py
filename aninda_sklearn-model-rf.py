# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/google-quest-challenge/train.csv',index_col='qa_id')

test=pd.read_csv('/kaggle/input/google-quest-challenge/test.csv',index_col='qa_id')

print("Shape of the training data is:",data.shape)

data.head()
#Column names

print("The column names are:",[data.columns])
val=data[:1000]

train=data[1000:]

print("Training data shape is:",train.shape)

print("Validation data shape is:",val.shape)
cols=['question_title', 'question_body', 'question_user_name', 'question_user_page', 'answer', 'answer_user_name', 'answer_user_page', 'url', 'category', 'host']

target_cols=['question_asker_intent_understanding', 'question_body_critical', 'question_conversational', 'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer', 'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare', 'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions', 'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling', 'question_well_written', 'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance', 'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written']
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestRegressor
X=train[cols].apply(lambda x: " ".join(x),axis=1)

print(X.shape)

y=train[target_cols]

print(y.shape)
print ('\nTransforming the training data...\n')

tfidf = TfidfVectorizer(stop_words='english')

train_tfidf = tfidf.fit_transform(X)

print (train_tfidf.shape)
reg=RandomForestRegressor()

reg.fit(train_tfidf,y)
X_val=val[cols].apply(lambda x: " ".join(x),axis=1)

print(X_val.shape)
print ('\nTransforming the validation data...\n')

val_tfidf = tfidf.transform(X_val)

print (val_tfidf.shape)
labels=reg.predict(val_tfidf)

labels.shape
from scipy.stats import spearmanr

score = 0

for i in range(30):

    score += np.nan_to_num(spearmanr(val[target_cols].values[:, i], labels[:,i]).correlation) / 30

score
test=pd.read_csv('/kaggle/input/google-quest-challenge/test.csv',index_col='qa_id')

test.shape
X_test=test.apply(lambda x: " ".join(x),axis=1)

print(X_test.shape)
print ('\nTransforming the test data...\n')

test_tfidf = tfidf.transform(X_test)

print (test_tfidf.shape)
labels_test=reg.predict(test_tfidf)
submission=pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')

print(submission.shape)

submission.loc[:,target_cols]=labels_test

print(submission.shape)

submission.head()
submission.to_csv('submission.csv', index=False)