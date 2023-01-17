# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl
# reading the dataset from raw csv file

import datatable as dt



train = dt.fread("../input/riiid-test-answer-prediction/train.csv").to_pandas()
# saving the dataset in .jay (binary format)

# dt.fread("../input/riiid-test-answer-prediction/train.csv").to_jay("train.jay")

# train = dt.fread("train.jay").to_pandas()
data_types_dict = {

    'row_id': 'int64',

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

    'content_type_id': 'int8',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float32',

    'prior_question_had_explanation': 'boolean'

}

train = train.astype(data_types_dict)
# print('Part of missing values for every column')

# print(len(train))

# print(train.isnull().sum())
train.info()
train_part_len = 87000000
def convertBoolean(x):

    if str(x) == "False":

        return 0

    elif str(x) == "True":

        return 1

    else:

        return 0



# TOEIC Section

def TOEICSection(part):

    if part >= 1 and part <= 4:

        return "Listening"

    elif part >= 5 and part <= 7:

        return "Reading"

    else:

        return "Missing"

    

def tagsCount(tags):

    arr = str(tags).split(" ")

    return len(arr)
questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')
tag = questions["tags"].str.split(" ", expand = True)

tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
questions =  pd.concat([questions,tag],axis=1)

questions['tags1'] = pd.to_numeric(questions['tags1'], errors='coerce')

questions['tags2'] = pd.to_numeric(questions['tags2'], errors='coerce')

questions['tags3'] = pd.to_numeric(questions['tags3'], errors='coerce')

questions['tags4'] = pd.to_numeric(questions['tags4'], errors='coerce')

questions['tags5'] = pd.to_numeric(questions['tags5'], errors='coerce')

questions['tags6'] = pd.to_numeric(questions['tags6'], errors='coerce')
def split_X_Y(_train):

    _train = _train[_train.content_type_id == False]

    #arrange by timestamp

    _train = _train.sort_values(['timestamp'], ascending=True)



    _train.drop(['timestamp','content_type_id'], axis=1,   inplace=True)

    

    X = _train.iloc[train_part_len:,:]



    X = X[X.answered_correctly!= -1 ]

    X = X.sort_values(['user_id'])

    Y = X[["answered_correctly"]]

    X = X.drop(["answered_correctly"], axis=1)



    return X, Y
import category_encoders as ce

count_en = ce.CountEncoder()
def addFeatures(_train, _questions):

    train_questions_only_df = _train[_train['answered_correctly']!=-1]

    grouped_by_user_df = train_questions_only_df.groupby('user_id')

    user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'sum']}).copy()

    user_answers_df.columns = ['mean_answered_correctly_user', 'questions_answered', 'sum_answered_correctly_user']

    

    grouped_by_content_df = train_questions_only_df.groupby('content_id')

    content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'] }).copy()

    content_answers_df.columns = ['mean_answered_correctly_content', 'question_asked']

    

    questions_df = _questions.merge(content_answers_df, left_on = 'question_id', right_on = 'content_id', how = 'left')

    bundle_dict = questions_df['bundle_id'].value_counts().to_dict()

    questions_df['right_answers'] = questions_df['mean_answered_correctly_content'] * questions_df['question_asked']

    questions_df['bundle_size'] = questions_df['bundle_id'].apply(lambda x: bundle_dict[x])



    grouped_by_bundle_df = questions_df.groupby('bundle_id')

    bundle_answers_df = grouped_by_bundle_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()

    bundle_answers_df.columns = ['bundle_rignt_answers', 'bundle_questions_asked']

    bundle_answers_df['bundle_accuracy'] = bundle_answers_df['bundle_rignt_answers'] / bundle_answers_df['bundle_questions_asked']



    grouped_by_part_df = questions_df.groupby('part')

    part_answers_df = grouped_by_part_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()

    part_answers_df.columns = ['part_rignt_answers', 'part_questions_asked']

    part_answers_df['part_accuracy'] = part_answers_df['part_rignt_answers'] / part_answers_df['part_questions_asked']



    return user_answers_df, questions_df, bundle_answers_df, part_answers_df
def handleMissing(X, Y, _user_answers, _questions, _bundle_answers, _part_answers):

    X['prior_question_had_explanation'].fillna(False, inplace=True)

    X["prior_question_had_explanation_enc"] = X['prior_question_had_explanation'].apply(convertBoolean)

    

    X = X[['user_id', 'content_id', 'prior_question_elapsed_time','prior_question_had_explanation_enc']] 

    

    X = X.merge(_user_answers, how = 'left', on = 'user_id')

    X = X.merge(_questions, how = 'left', left_on = 'content_id', right_on = 'question_id')

    X = X.merge(_bundle_answers, how = 'left', on = 'bundle_id')

    X = X.merge(_part_answers, how = 'left', on = 'part')

    

    X["toeic_section"] = X['part'].apply(TOEICSection)



    cat_ce = count_en.fit_transform(X["toeic_section"])

    X = X.join(cat_ce.add_suffix("_ce"))



    X["tags_count"] = X['tags'].apply(tagsCount)

    X = X.drop(columns=['user_id', 'content_id', 'bundle_id', 'correct_answer', 'tags', 'toeic_section'])

    X['sum_answered_correctly_user'].fillna(0, inplace=True)

    X.fillna(0.5, inplace=True)

    

    target_columns = ['question_id', 'part']

    target_enc = ce.TargetEncoder(cols=target_columns)

    target_enc.fit(X[target_columns], Y)

    X = X.join(target_enc.transform(X[target_columns]).add_suffix('_target'))

    

    catboost_columns = ['prior_question_elapsed_time', 'prior_question_had_explanation_enc']

    catboost_enc = ce.CatBoostEncoder(cols=catboost_columns)

    catboost_enc.fit(X[catboost_columns], Y)

    X = X.join(catboost_enc.transform(X[catboost_columns]).add_suffix('_catboost'))



    return X, target_enc, catboost_enc
X_full, Y_full = split_X_Y(train)
X_full.info()

# print('Part of missing values for every column')

# print(X_full.isnull().sum() / len(X_full))
user_answers_df, questions_df, bundle_answers_df, part_answers_df = addFeatures(train, questions)
X_full, target_enc, catboost_enc = handleMissing(X_full, Y_full, user_answers_df, questions_df, bundle_answers_df, part_answers_df)
X_full.info()
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier, plot_importance

import matplotlib.pyplot as plt
model = XGBClassifier(

    tree_method="hist",

    learning_rate=0.1,

    gamma=0.2,

    n_estimators=200,

    max_depth=8,

    min_child_weight=40,

    subsample=0.87,

    colsample_bytree=0.95,

    reg_alpha=0.04,

    reg_lambda=0.073,

    objective='binary:logistic',

    nthread=4,

    scale_pos_weight=1,

    seed=27

)
model.fit(X_full, Y_full.values.ravel())
roc_auc_score(Y_full.values, model.predict_proba(X_full)[:,1])
fig, ax = plt.subplots(figsize=(10,8))

plot_importance(model, ax=ax)

plt.show()
import riiideducation

env = riiideducation.make_env()

iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_df['prior_question_had_explanation'].fillna(False, inplace=True)

    test_df["prior_question_had_explanation_enc"] = test_df['prior_question_had_explanation'].apply(convertBoolean)



    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')

    test_df = test_df.merge(bundle_answers_df, how = 'left', on = 'bundle_id')

    test_df = test_df.merge(part_answers_df, how = 'left', on = 'part')

    

    test_df["toeic_section"] = test_df['part'].apply(TOEICSection)

    

    cat_ce = count_en.fit_transform(test_df["toeic_section"])

    test_df = test_df.join(cat_ce.add_suffix("_ce"))

    

    test_df["tags_count"] = test_df['tags'].apply(tagsCount)

    test_df['sum_answered_correctly_user'].fillna(0, inplace=True)

    test_df.fillna(0.5, inplace=True)

    

    target_columns = ['question_id', 'part']

    test_df = test_df.join(target_enc.transform(test_df[target_columns]).add_suffix('_target'))

    catboost_columns = ['prior_question_elapsed_time', 'prior_question_had_explanation_enc']

    test_df = test_df.join(catboost_enc.transform(test_df[catboost_columns]).add_suffix('_catboost'))



    test_df['answered_correctly'] =  model.predict_proba(test_df[X_full.columns])[:,1]

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])