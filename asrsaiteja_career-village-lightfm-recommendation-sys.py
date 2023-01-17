import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from lightfm.data import Dataset # all lightfm imports 

from lightfm import LightFM

from lightfm import cross_validation

from lightfm.evaluation import precision_at_k

from lightfm.evaluation import auc_score



# imports re for text cleaning 

import re

from datetime import datetime, timedelta



# we will ignore pandas warning 

import os



data_paths = {}



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        # print(filename)

        data_paths[filename] = os.path.join(dirname, filename)
df_professionals = pd.read_csv(data_paths['professionals_clean.csv'])

df_questions = pd.read_csv(data_paths['questions_clean.csv'])

df_interactions = pd.read_csv(data_paths['prof_ques_interactions.csv'])



df_professionals.shape, df_questions.shape, df_interactions.shape
df_professionals.columns, df_questions.columns, df_interactions.columns
print('data sample from professional table (users)')

df_professionals.sample(3)
print('data sample from questions table (items)')

df_professionals.sample(3)
print('data sample from interactions table')

df_professionals.sample(3)
def get_unique_tags(df, tags_col):

    

    """

    Generates unique tags for mapping them as features



    Parameters

    ----------

    df: Pandas Dataframe for Users or Q&A. 

    tags_col : name of the tags column    

    

    Returns

    -------

     Array of all unique features (tags).

    """

    features = df[[tags_col]].apply(lambda x: ','.join(x.map(str)), axis=1)

    features = features.str.split(',')

    features = features.apply(pd.Series).stack().reset_index(drop=1)

    return features.unique()







def create_id_level_features(df, id_col_name, features_names):

    

    """

    Generate features that will be ready for feeding into lightfm



    Parameters

    ----------

    df: Pandas Dataframe which contains features

    id_col_name: String

    features_names : List of feature columns name in dataframe



    Returns

    -------

    (user_id, ['feature_1', 'feature_2', 'feature_3'])

    """



    features = df[features_names].apply(lambda x: ','.join(x.map(str)), axis=1)

    features = features.str.split(',')

    features = list(zip(df[id_col_name], features))

    return features
# array of unique tags to stack as columns for the matrix

unique_user_tags = get_unique_tags(df_professionals, tags_col = 'professional_all_tags')

unique_item_tags = get_unique_tags(df_questions, tags_col = 'questions_tag_name')



# generate interaction tuples of (user_id, ques_id, weight)

user_item_interact_tuple = df_interactions.values



# creates tuple of respective (id, tags)

df_questions_features = create_id_level_features(df_questions, 'ques_uid', ['questions_tag_name'],)

df_professionals_features = create_id_level_features(df_professionals, 'prof_uid', ['professional_all_tags'],)
# building the lightfm DataSet, this will transform data into sparse matrices for lightfm internal mapping

dataset = Dataset()

dataset.fit(df_professionals['prof_uid'].unique(), df_questions['ques_uid'].unique(),

            user_features=unique_user_tags, 

            item_features=unique_item_tags)



# list(zip(prof_uid, ques_uid, interaction_weight))

interactions, _ = dataset.build_interactions(user_item_interact_tuple)



# builds sparse matrix (ques_uid, (ques_uids + unique_ques_tags))

questions_features = dataset.build_item_features(df_questions_features)

# builds sparse matrix (prof_uid, (prof_uids + unique_prof_tags))

professional_features = dataset.build_user_features(df_professionals_features)
### Labels 

interactions # Shape: (num of users, num of questions), Cell Value: 1 if that qid answered by uid else 0
### Recommendation Features

questions_features, professional_features # Shape (num of user\items, num of users\items + unique tags)
from lightfm.cross_validation import random_train_test_split

train_interactions, val_interactions = random_train_test_split(interactions, test_percentage=0.1)
from lightfm.evaluation import precision_at_k

from lightfm.evaluation import auc_score



def calculate_auc_score(lightfm_model, interactions_matrix, 

                        professional_features, question_features,):

    """

    Measure the ROC AUC metric for a trained LightFm model. 

    ROU AUC: probability that randomly choosen +ve (attractive) sample ranked higher than 

    the -ve (unattractive) example. Score range: (0, 1.0]



    Parameters

    ----------

    lightfm_model: A fitted lightfm model 

    interactions_matrix: A interactions matrix to evaluate 

    professional_features, question_features : Lightfm dataset - features matrices

        

    Returns

    -------

    String containing AUC score 

    """

    score = auc_score(lightfm_model, interactions_matrix,

                      user_features=professional_features, 

                      item_features=question_features, 

                      num_threads=4).mean()

    return score
## Modelling & Hyperparameter tuning part



model = LightFM(no_components=160,

                learning_rate=0.05,

                loss='warp',

                random_state=2019)



# fit the model

model = model.fit(train_interactions,

                  user_features=professional_features, 

                  item_features=questions_features,

                  epochs=7, num_threads=4, verbose=1)



val_auc = calculate_auc_score(model, val_interactions,

                    professional_features, questions_features,)



print('ROC-AUC on validation dataset:', val_auc)
def recommend_questions(pid):

    """

    Given an Professional Id (user), Will predict the Recommendation based on previous interaction data

    

    Parameters

    ----------

    pid: prof_uid in range (0, 28152)

    """         

    # print their previous answered question title

    answered_qids = df_interactions.loc[df_interactions['prof_uid'] == pid][:3]['ques_uid'].values

    df_previous_questions = df_questions.loc[df_questions['ques_uid'].isin(answered_qids)]



    print('--- Professional Id (' + str(pid) + ") Previously Answered Questions ---")

    display(df_previous_questions[['questions_title', 'questions_tag_name']])

    display(df_professionals.loc[df_professionals.prof_uid == pid])



    # predict the recommendations

    df_use_for_prediction = df_questions.loc[~df_questions['ques_uid'].isin(answered_qids)]

    questions_id_for_predict = df_use_for_prediction['ques_uid'].values.tolist()



    scores = model.predict(pid, questions_id_for_predict,

                           user_features=professional_features,

                           item_features=questions_features,)



    df_use_for_prediction['scores'] = scores

    df_use_for_prediction = df_use_for_prediction.sort_values(by='scores', ascending=False)[:10]



    print('--- Professional Id (' + str(pid) + "): Recommendations --- ")

    display(df_use_for_prediction[['questions_title', 'questions_tag_name']])

    

    return df_use_for_prediction
## UserId:1200,function prints his previously answered questions, his interest tags & then finally recommendations

res_df = recommend_questions(1200) # 1200, 19897, 3
res_df = recommend_questions(19897) # 3
res_df = recommend_questions(3) # 3
res_df = recommend_questions(654)
res_df = recommend_questions(8432)
res_df = recommend_questions(15684)
res_df = recommend_questions(3454)