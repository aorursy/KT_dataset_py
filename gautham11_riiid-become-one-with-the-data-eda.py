import pandas as pd
import numpy as np
from pathlib import Path
import plotly_express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# from https://www.kaggle.com/rohanrao/riiid-with-blazing-fast-rid
!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
# plotly helpers

# plots a histogram and a box plot of a column in dataframe
def distribution_plot(df, column, min_quantile=0, max_quantile=1):
    display(pd.DataFrame(df[column].describe(percentiles=np.arange(.1, 1, .1))).T)
    min_value = df[column].quantile(min_quantile)
    max_value = df[column].quantile(max_quantile)
    df = df[(df[column] >= min_value) & (df[column] <= max_value)]
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Histogram(x=df[column], nbinsx=100), row=1, col=1)
    fig.add_trace(go.Box(y=df[column], orientation='v', name=column), row=1, col=2)
    fig.update_layout(title_text=f'{column} Distribution | min quantile {min_quantile} | max quantile {max_quantile}', showlegend=False)
    fig.show()
    
def p_line(y, x=None, title=None):
    if x is None:
        if hasattr(y, 'index'):
            x = y.index
        else:
            x = list(range(0, len(y)))
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    if title is not None:
        fig.update_layout(title_text=title)
    fig.show()
data_dir = Path('../input/riiid-test-answer-prediction')
# import data files
import datatable as dt

train = dt.fread("../input/riiid-test-answer-prediction/train.csv").to_pandas()
print(train.shape)
train = reduce_mem_usage(train)
questions = pd.read_csv(data_dir/'questions.csv')
lectures = pd.read_csv(data_dir/'lectures.csv')
example_test = pd.read_csv(data_dir/'example_test.csv')
set(example_test['content_id']) - set(questions['question_id'])

train.head()
pd.DataFrame(train.isnull().sum(), columns=['null_count'])
print(f"Number of users in train - {train['user_id'].nunique()}")
content_count = pd.DataFrame(train['content_type_id'].value_counts()).reset_index()
content_count.columns = ['type', 'count']
content_count['type'] = content_count.type.replace({0: 'questions', 1: 'lectures'})
px.pie(content_count, values='count', names='type')
user_interactions_count = train[['row_id','user_id', 'content_type_id']].groupby(['user_id', 'content_type_id'], as_index=False).count()
user_interactions_count = user_interactions_count.rename(columns={'row_id': 'count'})
user_interactions_count = user_interactions_count.pivot(index='user_id', columns='content_type_id', values=['count'])
user_interactions_count = user_interactions_count.fillna(0)
user_interactions_count.columns = ['questions_count', 'lectures_count']
distribution_plot(user_interactions_count,'questions_count', min_quantile=0, max_quantile=.9)
qc_count = user_interactions_count['questions_count'].value_counts(normalize=True)
qc_count.index = pd.Series(qc_count.index).apply(lambda x: f'count_{x}')
p_line(qc_count.head(10), title='percentage of users vs questions count')
distribution_plot(user_interactions_count,'lectures_count', min_quantile=0, max_quantile=.95)
p_line(user_interactions_count['lectures_count'].value_counts(normalize=True).cumsum(), title='User percentage cumsum vs number of lectures')
user_interactions_count.corr()
user_interactions_count['questions_lectures'] = user_interactions_count['questions_count'].astype(str) + '_' + user_interactions_count['lectures_count'].astype(str)
p_line(user_interactions_count['questions_lectures'].value_counts(normalize=True).head(20), title='Percentage of users vs questions_lectures combo count')
user_interactions_count = user_interactions_count.sort_values('questions_count')
user_interactions_count['user_index'] = range(0, len(user_interactions_count))

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=user_interactions_count['user_index'], y=user_interactions_count['questions_count'], name="questions_count"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=user_interactions_count['user_index'], y=user_interactions_count['lectures_count'], name="lectures_count", opacity=.5),
    secondary_y=True,
)
fig.update_xaxes(title_text="user index")
fig.update_yaxes(title_text="number of Questions", secondary_y=False)
fig.update_yaxes(title_text="number of Lectures", secondary_y=True)
fig.show()
user_interactions_count = user_interactions_count.sort_values('lectures_count')
user_interactions_count['user_index'] = range(0, len(user_interactions_count))

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=user_interactions_count['user_index'], y=user_interactions_count['questions_count'], name="questions_count", opacity=.5),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=user_interactions_count['user_index'], y=user_interactions_count['lectures_count'], name="lectures_count"),
    secondary_y=True,
)
fig.update_xaxes(title_text="User index")
fig.update_yaxes(title_text="number of Questions", secondary_y=False)
fig.update_yaxes(title_text="number of Lectures", secondary_y=True)
fig.show()
pd.Series.is_monotonic_increasing
is_increasing = train[['timestamp', 'user_id']].groupby('user_id').agg(lambda x: x.is_monotonic_increasing)
questions.head()
pd.DataFrame(questions.isnull().sum(), columns=['null_count'])
print(f'Number of unique questions in train data - {train.loc[train["content_type_id"] == 0, "content_id"].nunique()}')
print(f'Number of unique questions in questions metadata - {questions["question_id"].nunique()}')
q_not_in_train_data = set(questions['question_id']) - set(train.loc[train['content_type_id'] == 0, 'content_id'])
print(f'Questions in Metadata, but not in train data - {q_not_in_train_data}')

q_not_in_metadata = set(train.loc[train['content_type_id'] == 0, 'content_id']) - set(questions['question_id'])
print(f'Questions in train, but not in metadata - {q_not_in_metadata}')
questions.head()
questions['part'].unique()
parts = questions[['question_id', 'part']].groupby('part', as_index=False).count().rename(columns={'question_id': 'number_of_questions'})
px.bar(parts, x='part', y='number_of_questions')
questions['bundle_id'].nunique()
bundle_count = questions[['question_id', 'bundle_id']].groupby('bundle_id', as_index=False).count().rename(columns={'question_id': 'number_of_questions'})
distribution_plot(bundle_count, 'number_of_questions')
question_answers = train.loc[train['content_type_id'] == 0, ['content_id', 'answered_correctly']].groupby('content_id', as_index=False).mean()
distribution_plot(question_answers, 'answered_correctly')
user_answers = train.loc[train['content_type_id'] == 0, ['user_id', 'answered_correctly']].groupby('user_id', as_index=False).mean()
distribution_plot(user_answers, 'answered_correctly')
px.histogram(user_answers, 'answered_correctly', nbins=100)
test = pd.read_csv(data_dir/'example_test.csv')
test.shape
set(test['user_id']) - set(train['user_id'])
user_question_count = train.loc[train['content_type_id'] == 0, ['row_id', 'user_id', 'content_id']].groupby(['user_id', 'content_id']).count()
p_line(user_question_count['row_id'].value_counts(normalize=True))
questions.isnull().sum()
print(f'{questions["tags"].nunique()} unique combinations')
questions['tags'].value_counts()
questions['n_tags'] = questions['tags'].fillna("").apply(lambda x: len(x.split()))
all_tags = pd.Series(np.concatenate(questions['tags'].fillna("").apply(lambda x: x.split()).values)).apply(lambda x: f'tag_{x}')
all_tags.nunique()
# this is number of question in questions meta-data,not in train data
p_line(all_tags.value_counts(), all_tags.value_counts().index, title='Unique question tag vs question count')
question_answers = question_answers.rename(columns={'answered_correctly': 'question_score'})
user_answers = user_answers.rename(columns={'answered_correctly': 'user_score'})
question_answers['content_type_id'] = 0
user_answers['content_type_id'] = 0
import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()
def harmonic_mean(a, b):
    return (2 * a * b) / (a + b + 1e-6)
for (test_df, sample_prediction_df) in iter_test:
    test_df = test_df.merge(question_answers, on=['content_id', 'content_type_id'], how='left')
    test_df = test_df.merge(user_answers, on=['user_id', 'content_type_id'], how='left')
    test_df['question_score'].fillna(.5, inplace=True)
    test_df['user_score'].fillna(.5, inplace=True)
    test_df['prediction'] = test_df.apply(lambda row: harmonic_mean(row['question_score'], row['user_score']), axis=1)
    test_df['answered_correctly'] = test_df['prediction']
    # display(test_df)
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])