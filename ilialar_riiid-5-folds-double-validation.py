import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

        

from sklearn.metrics import roc_auc_score



import lightgbm

from lightgbm import LGBMClassifier

from sklearn.model_selection import GroupShuffleSplit, KFold
data_types_dict = {

#     'row_id': 'int64',

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

#     'content_type_id': 'int8',

#     'task_container_id': 'int16',

#     'user_answer': 'int8',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float16',

    'prior_question_had_explanation': 'boolean'

}



train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                       nrows = 10**6,

                       usecols = data_types_dict.keys(),

                       dtype=data_types_dict, 

#                        index_col = 0

                      )
questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')

lectures_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')
class FeatureGenerator:

    def get_questions_feaures(self, questions_df,train_questions_only_df):

        grouped_by_content_df = train_questions_only_df.groupby('content_id')

        content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'] })

        content_answers_df.columns = ['mean_accuracy', 'question_asked']

        questions_df = questions_df.merge(content_answers_df, left_on = 'question_id', right_on = 'content_id', how = 'left')

        bundle_dict = questions_df['bundle_id'].value_counts().to_dict()

        questions_df['right_answers'] = questions_df['mean_accuracy'] * questions_df['question_asked']

        questions_df['bundle_size'] =questions_df['bundle_id'].apply(lambda x: bundle_dict[x])

        questions_df.set_index('question_id', inplace = True)

        return questions_df

    

    def get_users_features(self, train_questions_only_df):

        grouped_by_user_df = train_questions_only_df.groupby('user_id')

        user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count','sum']}).copy()

        user_answers_df.columns = ['mean_user_accuracy', 'questions_answered', 'questions_asked_user']

        return user_answers_df

    

    def get_bundle_features(self, questions_df):

        grouped_by_bundle_df = questions_df.groupby('bundle_id')

        bundle_answers_df = grouped_by_bundle_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()

        bundle_answers_df.columns = ['bundle_right_answers', 'bundle_questions_asked']

        bundle_answers_df['bundle_accuracy'] = bundle_answers_df['bundle_right_answers'] / bundle_answers_df['bundle_questions_asked']

        return bundle_answers_df

    

    def get_part_features(self, questions_df):    

        grouped_by_part_df = questions_df.groupby('part')

        part_answers_df = grouped_by_part_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()

        part_answers_df.columns = ['part_right_answers', 'part_questions_asked']

        part_answers_df['part_accuracy'] = part_answers_df['part_right_answers'] / part_answers_df['part_questions_asked']

        return part_answers_df

        

    def from_df(self, df, questions_df):

        # computes aggregated target features for a given dataset

        self.questions_df = self.get_questions_feaures(questions_df.copy(), df)

        self.user_answers_df = self.get_users_features(df)

        self.bundle_answers_df = self.get_bundle_features(self.questions_df)

        self.part_answers_df = self.get_part_features(self.questions_df)

        return self

        

    def enrich(self,df):

        # adds aggregated featurea to a given dataset

        df = df.merge(self.user_answers_df, how = 'left', on = 'user_id')

        df = df.merge(self.questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')

        df = df.merge(self.bundle_answers_df, how = 'left', on = 'bundle_id')

        df = df.merge(self.part_answers_df, how = 'left', on = 'part')

        return df

    

    def combine(self, fg):

        # combines two FeatureGenerators into one using all the data

        for df1,df2, question_asked_c, right_answer_c, accuracy_c, index_c in [

            (self.questions_df, fg.questions_df, 'question_asked', 'right_answers', 'mean_accuracy', 'question_id'),

            (self.user_answers_df, fg.user_answers_df, 'questions_asked_user', 'questions_answered', 'mean_user_accuracy', 'user_id'),

            (self.bundle_answers_df, fg.bundle_answers_df, 'bundle_questions_asked', 'bundle_right_answers', 'bundle_accuracy', 'bundle_id'),

            (self.part_answers_df, fg.part_answers_df, 'part_questions_asked', 'part_right_answers', 'part_accuracy', 'part'),

        ]:

            df1 = df1.merge(df2[[question_asked_c, right_answer_c]], how = 'outer', on = index_c)

            df1[question_asked_c] = df1[[f'{question_asked_c}_x', f'{question_asked_c}_y']].sum(1)

            df1[right_answer_c] = df1[[f'{right_answer_c}_x', f'{right_answer_c}_y']].sum(1)

            df1.drop([f'{question_asked_c}_x', f'{question_asked_c}_y',f'{right_answer_c}_x', f'{right_answer_c}_y'], 1)

            df1[accuracy_c] = df1[right_answer_c] / df1[question_asked_c]

        

    def normalize(self, factor):

        # normalizes additive features

        for df, features in [

            (self.questions_df, ['question_asked', 'right_answers']),

            (self.user_answers_df, ['questions_asked_user', 'questions_answered']),

            (self.bundle_answers_df, ['bundle_questions_asked', 'bundle_right_answers']),

            (self.part_answers_df, ['part_questions_asked', 'part_right_answers']),            

        ]:

            for c in features:

                df[c] /= factor

    

    def save(self, n, path):

        self.questions_df.to_csv(f'{path}/questions_{n}.csv')

        self.user_answers_df.to_csv(f'{path}/user_answers_{n}.csv')

        self.bundle_answers_df.to_csv(f'{path}/bundle_answers_{n}.csv')

        self.part_answers_df.to_csv(f'{path}/part_answers_{n}.csv')

        

    def load(self, n, path):

        self.questions_df = pd.read_csv(f'{path}/questions_{n}.csv', index_col = 0)

        self.user_answers_df = pd.read_csv(f'{path}/user_answers_{n}.csv', index_col = 0)

        self.bundle_answers_df = pd.read_csv(f'{path}/bundle_answers_{n}.csv', index_col = 0)

        self.part_answers_df = pd.read_csv(f'{path}/part_answers_{n}.csv', index_col = 0)

        return self
# features to use in the model

features = [

    'timestamp', 'prior_question_elapsed_time', 'prior_question_had_explanation', # original data

    'mean_user_accuracy', 'questions_answered', # user data

    'mean_accuracy', 'question_asked','right_answers',# questions answers

    'bundle_size', 'bundle_accuracy', # bundle features

    'part_accuracy', 'part' # part features

           ]

target = 'answered_correctly'
# we will save trained models and fitted feature generators here

os.mkdir('models')
def kfold_enreach(df, n = 5, random_state = 0):

    """Inner cycle of double validation

    For each fold computes the aggregated target-encodign features based on (n-1)/n part of data

    and applies it to the rest 1/n.

    Returns FeatureGenerator effectively trained on whole dataset 

    and the dataset with leak-free target-encoded features

    """

    data_list = []

    

    splitter = KFold(n, shuffle = True)

    # simplified KFold validation for the target encoding

    # can be improved by using the same technique as in 1st level splitting

    for j, (train_idx, valid_idx) in enumerate(splitter.split(df)):

        fg = FeatureGenerator().from_df(df.iloc[train_idx], questions_df)

        valid_df = df.iloc[valid_idx]

        valid_df = fg.enrich(valid_df)

        valid_df = valid_df[[c for c in valid_df.columns if c not in df.columns and c in features]]

        valid_df.index = valid_idx

        data_list.append(valid_df)

        

        if j == 0:

            final_fg = fg

        else:

            final_fg.combine(fg)

        

    # normalize additive columns that were used several times

    final_fg.normalize(n-1)

    

    new_faetures_df =  pd.concat(data_list).sort_index()



    return final_fg, pd.concat([df[[c for c in df.columns if c in features + [target]]], new_faetures_df[[c for c in new_faetures_df.columns if c in features]]], 1)
trained_models = []

scores = []



# we don't know this parameters for the test set so there should be better combination

test_size_final = 0.20

user_percent_having_history = 0.9

average_history_precent = 0.5



test_size = test_size_final / (user_percent_having_history * (1 - average_history_precent))

assert test_size < 1.0



train_df = train_df[train_df[target] != -1]

train_df.reset_index(inplace=True, drop = True)

splitter = GroupShuffleSplit(5, test_size = test_size, random_state = 0)



for j, (train_idx, test_idx) in enumerate(splitter.split(train_df,groups = train_df['user_id'])):

    user_count_dict = train_df['user_id'].iloc[test_idx].value_counts().to_dict()

    user_indices = train_df.groupby('user_id').indices

    # adding some of the early information of test users to train set

    new_train_id = []

    

    for i,user in enumerate(user_count_dict.keys()):

        if i % 10000 == 0: print(i)

        if np.random.rand() < user_percent_having_history:

            samples_to_add = np.random.binomial(user_count_dict[user], average_history_precent)

            if samples_to_add > 0:

                new_train_id.append(user_indices[user][:samples_to_add])

    train_idx = np.hstack(new_train_id + [train_idx])

    test_idx = np.setdiff1d(test_idx,train_idx)

    

    train_fold_df = train_df.iloc[train_idx]

    valid_fold_df = train_df.iloc[test_idx]

    train_fold_df.reset_index(inplace = True, drop = True)

    

    # adding target-encodign features usign double validation

    final_fg, train_fold_df = kfold_enreach(train_fold_df, n = 5, random_state = j + 1)

    valid_fold_df = final_fg.enrich(valid_fold_df)

    

    # I didn't do any params optimisation yet, the current ones are similar to: https://www.kaggle.com/dwit392/expanding-on-simple-lgbm

    params = {

    'objective': 'binary',

    'max_bin': 700,

    'learning_rate': 0.1,

    'num_leaves': 31,

    'num_boost_round': 10000

}

    

    lgbm = LGBMClassifier(

        **params,

    )

    

    fill_dict = {x:0.6 for x in ['mean_user_accuracy','mean_accuracy','bundle_accuracy', 'part_accuracy'] if x in train_fold_df.columns}

    print(fill_dict)

    

    # filling NaNs

    for df in [train_fold_df, valid_fold_df]:

        if 'prior_question_had_explanation' in features:

            df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(value = False).astype(bool)

        df.fillna(fill_dict, inplace = True)

        df.fillna(value = 0, inplace = True)



    lgbm.fit(train_fold_df[features], train_fold_df[target],

            eval_set = [

                (valid_fold_df[features], valid_fold_df[target]),

                (train_fold_df[features], train_fold_df[target]),

            ],

            categorical_feature = ['part'],

            early_stopping_rounds = 10,

            eval_metric='auc',

            )

    

    # saving the trained model

    lgbm.booster_.save_model(f'models/model_{j}.txt')

    final_fg.save(j, 'models')

    trained_models.append({'model': lgbm, 'feature_extractor': final_fg})

    scores.append(lgbm.best_score_['valid_0']['auc'])

    

    

print(np.mean(scores))

print(scores)
load_pretrained_models = True

max_models_num = 10



path = '/kaggle/input/riiid-models/'

if load_pretrained_models:

    trained_models = []

    i = 0

    while os.path.exists(f"{path}/questions_{i}.csv"):

        fg = FeatureGenerator().load(i, f'{path}')

        model = lightgbm.Booster(model_file=f'{path}/model_{i}.txt')

        trained_models.append({'model': model, 'feature_extractor': fg})

        i += 1

        if i == max_models_num:

            break

else:

    for data in trained_models:

        data['model'] = data['model'].booster_
import riiideducation



env = riiideducation.make_env()
iter_test = env.iter_test()
for j, (test_df, sample_prediction_df) in enumerate(iter_test):

    for i, pipeline in enumerate(trained_models):

        # making predictions

        local_test_df = pipeline['feature_extractor'].enrich(test_df.copy())

        local_test_df['prior_question_had_explanation'] = local_test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

        local_test_df.fillna(fill_dict, inplace = True)

        local_test_df.fillna(value = 0, inplace = True)

        

        if i == 0:

            predicition = pipeline['model'].predict(local_test_df[features])

        else:

            predicition += pipeline['model'].predict(local_test_df[features])

        

    predicition /= len(trained_models)



    test_df['answered_correctly'] = predicition

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])