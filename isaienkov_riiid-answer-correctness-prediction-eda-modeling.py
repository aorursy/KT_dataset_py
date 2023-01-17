import numpy as np

import pandas as pd



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go



from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression



import optuna

from optuna.samplers import TPESampler



import gc

from sklearn.model_selection import train_test_split

import riiideducation
train_df = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv', 

    low_memory=False, 

    nrows=10**6, 

    dtype={

        'row_id': 'int64', 

        'timestamp': 'int64', 

        'user_id': 'int32', 

        'content_id': 'int16', 

        'content_type_id': 'int8',

        'task_container_id': 'int16', 

        'user_answer': 'int8', 

        'answered_correctly': 'int8', 

        'prior_question_elapsed_time': 'float32', 

        'prior_question_had_explanation': 'boolean'

    }

)



train_df
print('Part of missing values for every column')

print(train_df.isnull().sum() / len(train_df))
ds = train_df['user_id'].value_counts().reset_index()

ds.columns = ['user_id', 'count']

ds['user_id'] = ds['user_id'].astype(str) + '-'

ds = ds.sort_values(['count'])



fig = px.bar(

    ds.tail(40), 

    x='count', 

    y='user_id', 

    orientation='h', 

    title='Top 40 users by number of actions', 

    height=900, 

    width=700

)



fig.show()
ds = train_df['user_id'].value_counts().reset_index()

ds.columns = ['user_id', 'count']

ds = ds.sort_values('user_id')



fig = px.line(

    ds, 

    x='user_id', 

    y='count', 

    title='User action distribution', 

    height=600, 

    width=900

)



fig.show()
ds = train_df['content_id'].value_counts().reset_index()

ds.columns = ['content_id', 'count']

ds['content_id'] = ds['content_id'].astype(str) + '-'

ds = ds.sort_values(['count'])



fig = px.bar(

    ds.tail(40), 

    x='count', 

    y='content_id', 

    orientation='h', 

    title='Top 40 most useful content_ids', 

    height=900, 

    width=700

)



fig.show()
ds = train_df['content_id'].value_counts().reset_index()

ds.columns = ['content_id', 'count']

ds = ds.sort_values('content_id')



fig = px.line(

    ds, 

    x='content_id', 

    y='count', 

    title='content_id action distribution', 

    height=600, 

    width=900

)



fig.show()
ds = train_df['content_type_id'].value_counts().reset_index()

ds.columns = ['content_type_id', 'percent']

ds['percent'] /= len(train_df)



fig = px.pie(

    ds, 

    names='content_type_id', 

    values='percent', 

    title='Lecures & questions', 

    height=500, 

    width=600

)



fig.show()
ds = train_df['task_container_id'].value_counts().reset_index()

ds.columns = ['task_container_id', 'count']

ds['task_container_id'] = ds['task_container_id'].astype(str) + '-'

ds = ds.sort_values(['count'])



fig = px.bar(

    ds.tail(40), 

    x='count', 

    y='task_container_id', 

    orientation='h', 

    title='Top 40 most useful task_container_ids', 

    height=900, 

    width=700

)



fig.show()
ds = train_df['task_container_id'].value_counts().reset_index()

ds.columns = ['task_container_id', 'count']

ds = ds.sort_values('task_container_id')



fig = px.line(

    ds, 

    x='task_container_id', 

    y='count', 

    title='task_container_id action distribution', 

    height=600, 

    width=800

)



fig.show()
ds = train_df['content_type_id'].value_counts().reset_index()

ds.columns = ['content_type_id', 'percent']

ds['percent'] /= len(train_df)



fig = px.pie(

    ds, 

    names='content_type_id', 

    values='percent', 

    title='Lecures & questions', 

    height=500, 

    width=600

)



fig.show()
ds = train_df['user_answer'].value_counts().reset_index()

ds.columns = ['user_answer', 'percent_of_answers']

ds['percent_of_answers'] /= len(train_df)

ds = ds.sort_values(['percent_of_answers'])



fig = px.bar(

    ds, 

    x='user_answer', 

    y='percent_of_answers', 

    orientation='v', 

    title='Percent of user answers for every option', 

    height=500, 

    width=600

)



fig.show()
ds = train_df['answered_correctly'].value_counts().reset_index()

ds.columns = ['answered_correctly', 'percent_of_answers']

ds['percent_of_answers'] /= len(train_df)

ds = ds.sort_values(['percent_of_answers'])



fig = px.pie(

    ds, 

    names='answered_correctly', 

    values='percent_of_answers', 

    title='Percent of correct answers', 

    height=500, 

    width=600

)



fig.show()
fig = make_subplots(rows=2, cols=3)



traces = [

    go.Bar(

        x=[-1, 0, 1], 

        y=[

            len(train_df[(train_df['user_answer']==item) & (train_df['answered_correctly'] == -1)]),

            len(train_df[(train_df['user_answer']==item) & (train_df['answered_correctly'] == 0)]),

            len(train_df[(train_df['user_answer']==item) & (train_df['answered_correctly'] == 1)])

        ], 

        name='Option: ' + str(item),

        text = [

            str(round(100 * len(train_df[(train_df['user_answer']==item) & (train_df['answered_correctly'] == -1)]) / len(train_df[(train_df['user_answer']==item)]), 2)) + '%',

            str(round(100 * len(train_df[(train_df['user_answer']==item) & (train_df['answered_correctly'] == -0)]) / len(train_df[(train_df['user_answer']==item)]), 2)) + '%',

            str(round(100 * len(train_df[(train_df['user_answer']==item) & (train_df['answered_correctly'] == 1)]) / len(train_df[(train_df['user_answer']==item)]), 2)) + '%',

        ],

        textposition='auto'

    ) for item in train_df['user_answer'].unique().tolist()

]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 3) + 1, (i % 3)  +1)



fig.update_layout(

    title_text='Percent of correct answers for every option',

    height=600,

    width=900

)



fig.show()
fig = px.histogram(

    train_df, 

    x="prior_question_elapsed_time",

    nbins=100,

    width=700,

    height=500,

    title='prior_question_elapsed_time distribution'

)



fig.show()
questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')

questions
print('Part of missing values for every column')

print(questions.isnull().sum() / len(questions))
questions['tag'] = questions['tags'].str.split(' ')

questions = questions.explode('tag')

questions = pd.merge(questions, questions.groupby('question_id')['tag'].count().reset_index(), on='question_id')

questions = questions.drop(['tag_x'], axis=1)

questions.columns = ['question_id', 'bundle_id', 'correct_answer', 'part', 'tags', 'tags_number']

questions = questions.drop_duplicates()

questions
ds = questions['correct_answer'].value_counts().reset_index()

ds.columns = ['correct_answer', 'number_of_answers']

ds['correct_answer'] = ds['correct_answer'].astype(str) + '-'

ds = ds.sort_values(['number_of_answers'])



fig = px.bar(

    ds, 

    x='number_of_answers', 

    y='correct_answer', 

    orientation='h', 

    title='Number of correct answers per group', 

    height=400, 

    width=700

)



fig.show()
ds = questions['part'].value_counts().reset_index()

ds.columns = ['part', 'count']

ds['part'] = ds['part'].astype(str) + '-'

ds = ds.sort_values(['count'])



fig = px.bar(

    ds, 

    x='count', 

    y='part', 

    orientation='h', 

    title='Parts distribution', 

    height=500, 

    width=700

)



fig.show()
ds = questions['tags_number'].value_counts().reset_index()

ds.columns = ['tags_number', 'count']

ds['tags_number'] = ds['tags_number'].astype(str) + '-'

ds = ds.sort_values(['tags_number'])



fig = px.bar(

    ds, 

    x='count', 

    y='tags_number', 

    orientation='h', 

    title='Number tags distribution', 

    height=400, 

    width=700

)



fig.show()
check = questions['tags'].str.split(' ').explode('tags').reset_index()

check = check['tags'].value_counts().reset_index()



check.columns = ['tag', 'count']

check['tag'] = check['tag'].astype(str) + '-'

check = check.sort_values(['count'])



fig = px.bar(

    check.tail(40), 

    x='count', 

    y='tag', 

    orientation='h', 

    title='Top 40 most useful tags', 

    height=900, 

    width=700

)



fig.show()
lectures = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')

lectures
print('Part of missing values for every column')

print(lectures.isnull().sum() / len(lectures))
ds = lectures['tag'].value_counts().reset_index()

ds.columns = ['tag', 'count']

ds['tag'] = ds['tag'].astype(str) + '-'

ds = ds.sort_values(['count'])



fig = px.bar(

    ds.tail(40), 

    x='count', 

    y='tag', 

    orientation='h', 

    title='Top 40 lectures by number of tags', 

    height=800, 

    width=700

)



fig.show()
ds = lectures['part'].value_counts().reset_index()

ds.columns = ['part', 'count']

ds['part'] = ds['part'].astype(str) + '-'

ds = ds.sort_values(['count'])



fig = px.bar(

    ds, 

    x='count', 

    y='part', 

    orientation='h', 

    title='Parts distribution', 

    height=500, 

    width=700

)



fig.show()
ds = lectures['type_of'].value_counts().reset_index()

ds.columns = ['type_of', 'count']

ds = ds.sort_values(['count'])



fig = px.bar(

    ds, 

    x='count', 

    y='type_of', 

    orientation='h', 

    title='type_of column distribution', 

    height=500, 

    width=700

)



fig.show()
test_ex = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv')

test_ex
used_data_types_dict = {

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float16',

    'prior_question_had_explanation': 'boolean'

}
train_df = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv',

    usecols = used_data_types_dict.keys(),

    dtype=used_data_types_dict, 

    index_col = 0

)
features_df = train_df.iloc[:int(9 /10 * len(train_df))]

train_df = train_df.iloc[int(9 /10 * len(train_df)):]
train_df
train_questions_only_df = features_df[features_df['answered_correctly']!=-1]

grouped_by_user_df = train_questions_only_df.groupby('user_id')

user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'std', 'median', 'skew']}).copy()

user_answers_df.columns = ['mean_user_accuracy', 'questions_answered', 'std_user_accuracy', 'median_user_accuracy', 'skew_user_accuracy']
user_answers_df
user_answers_df['median_user_accuracy'].value_counts()
grouped_by_content_df = train_questions_only_df.groupby('content_id')

content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count', 'std', 'median', 'skew'] }).copy()

content_answers_df.columns = ['mean_accuracy', 'question_asked', 'std_accuracy', 'median_accuracy', 'skew_accuracy']
content_answers_df['median_accuracy'].value_counts()
content_answers_df
del features_df

del grouped_by_user_df

del grouped_by_content_df



gc.collect()
features = [

    'mean_user_accuracy', 

    'questions_answered',

    'std_user_accuracy', 

    'median_user_accuracy',

    'skew_user_accuracy',

    'mean_accuracy', 

    'question_asked',

    'std_accuracy', 

    'median_accuracy',

    'prior_question_elapsed_time', 

    'prior_question_had_explanation',

    'skew_accuracy'

]

target = 'answered_correctly'
train_df = train_df[train_df[target] != -1]
train_df = train_df.merge(user_answers_df, how='left', on='user_id')

train_df = train_df.merge(content_answers_df, how='left', on='content_id')

train_df
train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

train_df = train_df.fillna(value = 0.5)
train_df = train_df[features + [target]]
train_df = train_df.replace([np.inf, -np.inf], np.nan)

train_df = train_df.fillna(0.5)

train_df
train_df, test_df = train_test_split(train_df, random_state=666, test_size=0.2)
test_df
sampler = TPESampler(seed=666)



def create_model(trial):

    num_leaves = trial.suggest_int("num_leaves", 2, 31)

    n_estimators = trial.suggest_int("n_estimators", 50, 300)

    max_depth = trial.suggest_int('max_depth', 3, 8)

    min_child_samples = trial.suggest_int('min_child_samples', 100, 1200)

    learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)

    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 90)

    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.0001, 1.0)

    feature_fraction = trial.suggest_uniform('feature_fraction', 0.0001, 1.0)

    model = LGBMClassifier(

        num_leaves=num_leaves,

        n_estimators=n_estimators, 

        max_depth=max_depth, 

        min_child_samples=min_child_samples, 

        min_data_in_leaf=min_data_in_leaf,

        learning_rate=learning_rate,

        feature_fraction=feature_fraction,

        random_state=666

)

    return model



def objective(trial):

    model = create_model(trial)

    model.fit(train_df[features], train_df[target])

    score = roc_auc_score(test_df[target].values, model.predict_proba(test_df[features])[:,1])

    return score



# uncomment to use optuna

# study = optuna.create_study(direction="maximize", sampler=sampler)

# study.optimize(objective, n_trials=70)



# params = study.best_params

# params['random_state'] = 666

params = {

    'bagging_fraction': 0.5817242323514327,

    'feature_fraction': 0.6884588361650144,

    'learning_rate': 0.42887924851375825, 

    'max_depth': 6,

    'min_child_samples': 946, 

    'min_data_in_leaf': 47, 

    'n_estimators': 169,

    'num_leaves': 29,

    'random_state': 666

}

model = LGBMClassifier(**params)

model.fit(train_df[features], train_df[target])
print('LGB score: ', roc_auc_score(test_df[target].values, model.predict_proba(test_df[features])[:,1]))
env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(content_answers_df, how = 'left', on = 'content_id')

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_df.fillna(value = 0.5, inplace = True)



    test_df['answered_correctly'] = model.predict_proba(test_df[features])[:,1]

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])