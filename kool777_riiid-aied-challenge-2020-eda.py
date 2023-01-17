!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl > /dev/null
# import packages

import os, gc

import warnings

import numpy as np

import pandas as pd

import datatable as dt



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# riiideducation module

import riiideducation



%matplotlib inline

warnings.filterwarnings('ignore')



# directory

print('Competition Data/Files')

os.listdir('../input/riiid-test-answer-prediction')
# root directory

ROOT = '../input/riiid-test-answer-prediction/'



# files

train = dt.fread("../input/riiid-test-answer-prediction/train.csv").to_pandas()



train = train.astype({

    'row_id': 'int32',

    'timestamp': 'int64',

    'user_id': 'int64',

    'content_id': 'int16',

    'content_type_id': 'int8',

    'task_container_id': 'int16',

    'user_answer': 'int8',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float32',

    'prior_question_had_explanation': 'boolean'

})



questions = pd.read_csv(f'{ROOT}questions.csv')

lectures = pd.read_csv(f'{ROOT}lectures.csv')

example_test = pd.read_csv(f'{ROOT}example_test.csv')

example_sample_submission = pd.read_csv(f'{ROOT}example_sample_submission.csv')
train.head()
print(f'We have {train.shape[0]} rows and {train.shape[1]} features in train.csv.')
train.info()
print(f'Missing values in train.csv in each columns:\n{train.isnull().sum()}')
print(f'We have total of {train.isnull().values.sum()} missing values in train data.')
print('Unique Values in each column of train.csv')

print('##########################################')

for col in train:

    print(f'{col}: {train[col].nunique()}')
questions.head()
print(f'We have {questions.shape[0]} rows and {questions.shape[1]} features in questions.csv.')
print(f'Missing values in questions.csv in each columns:\n{questions.isnull().sum()}')
print(f'We have total of {questions.isnull().values.sum()} missing values in train data.')
print('Unique Values in each column of questions.csv')

print('##########################################')

for col in questions:

    print(f'{col}: {questions[col].nunique()}')
lectures.head()
print(f'We have {lectures.shape[0]} rows and {lectures.shape[1]} features in lectures.csv.')
print(f'Missing values in lectures.csv in each columns:\n{lectures.isnull().sum()}')
print(f'We have total of {lectures.isnull().values.sum()} missing values in lectures data.')
print('Unique Values in each column of lectures.csv')

print('##########################################')

for col in lectures:

    print(f'{col}: {lectures[col].nunique()}')
# You can only call make_env() once, so don't lose it!

env = riiideducation.make_env()



# You can only iterate through a result from `env.iter_test()` once

# so be careful not to lose it once you start iterating.

iter_test = env.iter_test()
count = 0

for (test_df, sample_prediction_df) in iter_test:

    test_df['answered_correctly'] = 0.5

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

    count += len(test_df)
print(f'We have {count} observations in total and {test_df.shape[1]} features in test.csv.')
test_df.head()
test_df.info()
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    train['timestamp'].hist(bins = 50,color='orange')

    plt.title("Timestamp Distribution")



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    train['user_id'].hist(bins = 50,color='red')

    plt.title("User Id Distribution")
mean = train['content_id'].mean()

median = train['content_id'].median()

mode = train['content_id'].mode()[0]



mean_2 = train['task_container_id'].mean()

median_2 = train['task_container_id'].median()

mode_2 = train['task_container_id'].mode()[0]



print(f'Content Id (Mean): {mean}')

print(f'Content Id (Median): {median}')

print(f'Content Id (Mode): {mode}\n')

print('######################################\n')

print(f'Task Container Id (Mean): {mean_2}')

print(f'Task Container Id (Median): {median_2}')

print(f'Task Container Id (Mode): {mode_2}')
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    sns.distplot(train['content_id'], color='green')

    ax.axvline(int(mean), color='r', linestyle='--')

    ax.axvline(int(median), color='y', linestyle='-')

    ax.axvline(mode, color='b', linestyle='-')

    plt.legend({'Mean':mean,'Median':median,'Mode':mode})

    plt.title("Content Id Distribution")

    

with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    sns.distplot(train['task_container_id'], color='yellow')

    ax.axvline(int(mean_2), color='r', linestyle='--')

    ax.axvline(int(median_2), color='g', linestyle='-')

    ax.axvline(mode_2, color='b', linestyle='-')

    plt.legend({'Mean':mean_2,'Median':median_2,'Mode':mode_2})

    plt.title("Task Container Id Distribution")
mean_3= train['prior_question_elapsed_time'].mean()

median_3 = train['prior_question_elapsed_time'].median()

mode_3 = train['prior_question_elapsed_time'].mode()[0]
f = plt.figure(figsize=(16, 8))



with sns.axes_style("whitegrid"):

    sns.distplot(train['prior_question_elapsed_time'], color='olive')

    plt.axvline(int(mean_3), color='c', linestyle='--')

    plt.axvline(int(median_3), color='m', linestyle='-')

    plt.axvline(mode_3, color='k', linestyle='-')

    plt.legend({'Mean':mean_3,'Median':median_3,'Mode':mode_3})

    plt.title("Prior Question Elapsed Time Distribution")

    

print(f'Prior Question Elapsed Time (Mean): {mean_3}')

print(f'Prior Question Elapsed Time (Median): {median_3}')

print(f'Prior Question Elapsed Time (Mode): {mode_3}')
mean_4 = lectures['tag'].mean()

median_4 = lectures['tag'].median()

mode_4 = lectures['tag'].mode()[0]
f = plt.figure(figsize=(16, 8))



with sns.axes_style("whitegrid"):

    sns.distplot(lectures['tag'], color='coral', bins=20)

    plt.axvline(int(mean_4), color='r', linestyle='--')

    plt.axvline(int(median_4), color='g', linestyle='-')

    plt.axvline(mode_4, color='b', linestyle='-')

    plt.legend({'Mean':mean_4,'Median':median_4,'Mode':mode_4})

    plt.title("Lecture Tag Distribution")

    

print(f'Tag (Mean): {mean_4}')

print(f'Tag (Median): {median_4}')

print(f'Tag (Mode): {mode_4}')
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    test_df['timestamp'].hist(bins = 50,color='maroon')

    plt.title("Timestamp Distribution in Test Data")



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    test_df['user_id'].hist(bins = 50,color='gold')

    plt.title("User Id Distribution in Test Data")
mean_5 = test_df['content_id'].mean()

median_5 = test_df['content_id'].median()

mode_5 = test_df['content_id'].mode()[0]



mean_6 = test_df['task_container_id'].mean()

median_6 = test_df['task_container_id'].median()

mode_6 = test_df['task_container_id'].mode()[0]



print(f'Content Id Test(Mean): {mean_5}')

print(f'Content Id Test(Median): {median_5}')

print(f'Content Id Test(Mode): {mode_5}\n')

print('######################################\n')

print(f'Task Container Id Test(Mean): {mean_6}')

print(f'Task Container Id Test(Median): {median_6}')

print(f'Task Container Id Test(Mode): {mode_6}')
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    sns.distplot(test_df['content_id'], color='cyan')

    ax.axvline(int(mean_5), color='r', linestyle='--')

    ax.axvline(int(median_5), color='y', linestyle='-')

    ax.axvline(mode_5, color='b', linestyle='-')

    plt.legend({'Mean':mean_5,'Median':median_5,'Mode':mode_5})

    plt.title("Content Id Distribution in Test Data")



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    sns.distplot(test_df['task_container_id'], color='purple')

    ax.axvline(int(mean_6), color='r', linestyle='--')

    ax.axvline(int(median_6), color='y', linestyle='-')

    ax.axvline(mode_6, color='b', linestyle='-')

    plt.legend({'Mean':mean,'Median':median,'Mode':mode})

    plt.title("Task Container Id Distribution in Test Data")
mean_7 = test_df['prior_question_elapsed_time'].mean()

median_7 = test_df['prior_question_elapsed_time'].median()

mode_7 = test_df['prior_question_elapsed_time'].mode()[0]
f = plt.figure(figsize=(16, 8))



with sns.axes_style("whitegrid"):

    sns.distplot(test_df['prior_question_elapsed_time'], color='darkgoldenrod')

    plt.axvline(int(mean_7), color='r', linestyle='--')

    plt.axvline(int(median_7), color='g', linestyle='-')

    plt.axvline(mode_7, color='b', linestyle='-')

    plt.legend({'Mean':mean_7,'Median':median_7,'Mode':mode_7})

    plt.title("Prior Question Elapsed Time Distribution")

    

print(f'Content Id Test(Mean): {mean_7}')

print(f'Content Id Test(Median): {median_7}')

print(f'Content Id Test(Mode): {mode_7}\n')
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 3)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    ay = sns.countplot(y = train['user_id'], order=train.user_id.value_counts().index[:10], palette="ocean_r")

    plt.title("Top 10 Active Users")



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    aa = sns.countplot(y = train['content_id'], order=train.content_id.value_counts().index[:10], palette="terrain")

    plt.title("Top 10 Popular Contents Ids")

    

with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 2])

    aa = sns.countplot(y = train['task_container_id'], order=train.task_container_id.value_counts().index[:10], palette="OrRd_r")

    plt.title("Top 10 Tasks")
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    ay = sns.countplot(train['user_answer'], palette="Set3")

    for p in ax.patches:

        height = p.get_height()

        ay.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/101230332*100),

                ha="center", fontsize=12)

    plt.xlabel('user answer',fontsize=12)

    plt.ylabel('count',fontsize=12)

    plt.title("User's Answer To Questions")



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    aa = sns.countplot(train['answered_correctly'], palette="pastel")

    for p in ax.patches:

        height = p.get_height()

        aa.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/101230332*100),

                ha="center", fontsize=14)

    plt.xlabel('answered correctly',fontsize=12)

    plt.ylabel('count',fontsize=12)

    plt.title("Correct Answers")
f = plt.figure(figsize=(16, 10))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    ay = sns.countplot(train['prior_question_had_explanation'].dropna(), palette="Pastel1")

    for p in ax.patches:

        height = p.get_height()

        ay.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/101230332*100),

                ha="center", fontsize=12)

    plt.xlabel('Prior question had explanation',fontsize=12)

    plt.ylabel('count',fontsize=12)

    plt.title("Users Saw Explanation", fontsize=14)

    

with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    aa = sns.countplot(train['content_type_id'], palette="twilight_r")

    for p in ax.patches:

        height = p.get_height()

        aa.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/101230332*100),

                ha="center", fontsize=12)

    plt.xlabel('Content Type Id',fontsize=12)

    plt.ylabel('count',fontsize=12)

    plt.title("Posed Question/Watching Lecture", fontsize=14)
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    ay = sns.countplot(questions['correct_answer'], palette="hls")

    for p in ax.patches:

        height = p.get_height()

        ay.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/13523*100),

                ha="center", fontsize=14)

    plt.title("User's Answer To Questions")



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    aa = sns.countplot(questions['part'], palette="deep")

    for p in ax.patches:

        height = p.get_height()

        aa.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/13523*100),

                ha="center", fontsize=12)

    plt.title("TOIEC English-language Assessment Section Number")
questions['tag'] = questions['tags'].str.split(' ')

questions['tag_length'] = questions['tag'].str.len()

tag_len = questions['tag_length'].dropna()

tag_len = tag_len.astype({'tag_length': 'int8'})



top_tags = questions.tag.explode('tags').reset_index()
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    aa = sns.countplot(tag_len, palette="coolwarm")

    for p in aa.patches:

        height = p.get_height()

        aa.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/13522*100),

                ha="center", fontsize=12)

    plt.xlabel('number of tags',fontsize=14)

    plt.ylabel('count',fontsize=14)

    plt.title("Number Of Tags Per Questions", fontsize=14)

    

with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    sns.countplot(y = top_tags['tag'], order = top_tags.tag.value_counts().index[:10], palette="ocean_r")

    plt.xlabel('count',fontsize=14)

    plt.ylabel('tag',fontsize=14)

    plt.title("Top 10 Tags",fontsize=14)
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    ay = sns.countplot(lectures['part'], palette='BuPu_r')

    for p in ax.patches:

        height = p.get_height()

        ay.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/418*100),

                ha="center", fontsize=14)

    plt.title("Category code for lecture")



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    aa = sns.countplot(lectures['type_of'], palette="gist_stern_r")

    for p in ax.patches:

        height = p.get_height()

        aa.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/418*100),

                ha="center", fontsize=12)

    plt.title("Lecture Description")
f = plt.figure(figsize=(16, 8))

gs = f.add_gridspec(1, 2)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    ay = sns.countplot(train['user_answer'], hue = train['prior_question_had_explanation'], palette="vlag")

    for p in ax.patches:

        height = p.get_height()

        ay.text(p.get_x()+p.get_width()/2.,

                height + 2,

                '{:1.2f}%'.format(height/101230332*100),

                ha="center", fontsize=12)

    plt.xlabel('User Answer',fontsize=14)

    plt.ylabel('count',fontsize=14)

    plt.title("User's Answer With And Without Explanation", fontsize=16)

        

with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 1])

    aa = sns.countplot(train['answered_correctly'], hue= train['prior_question_had_explanation'], palette="deep")

    for p in ax.patches:

        height = p.get_height()

        aa.text(p.get_x()+p.get_width()/2.,

                height + 2,

                '{:1.2f}%'.format(height/101230332*100),

                ha="center", fontsize=12)

    plt.legend(loc='center upper')

    plt.xlabel('Answered Correctly',fontsize=15)

    plt.ylabel('count',fontsize=14)

    plt.title("User's Saw explanation and Correct Answers", fontsize=15)
f = plt.figure(figsize=(16, 8))



with sns.axes_style("whitegrid"):

    sns.countplot(train['user_answer'], hue = train['answered_correctly'], palette="husl")

    plt.title("User's Answer vs Answered Correctly")
f = plt.figure(figsize=(16, 8))



with sns.axes_style("whitegrid"):

    sns.countplot(questions['correct_answer'], hue = questions['part'], palette="Spectral")

    plt.title("User's Answer vs Answered Correctly")
f = plt.figure(figsize=(16, 8))



with sns.axes_style("white"):

    sns.catplot(x="part", y="tag_length", kind="box",

                col="correct_answer", aspect=.7, data=questions)
with sns.axes_style("white"):

    sns.pairplot(questions, hue="correct_answer", palette="gnuplot_r", diag_kind="kde",

                 height=3, corner=True, plot_kws=dict(linewidth=1, alpha=1))
f = plt.figure(figsize=(16, 8))



with sns.axes_style("whitegrid"):

    sns.countplot(lectures['part'], hue = lectures['type_of'], palette="Spectral")

    plt.title("Categories in Parts")
f = plt.figure(figsize=(16, 8))



with sns.axes_style("white"):

    sns.catplot(x="part", y="tag", kind="box",

                col="type_of", aspect=.7, data=lectures)
with sns.axes_style("white"):

    sns.pairplot(lectures, hue="type_of", palette="copper", height=3,

                 corner=True, plot_kws=dict(linewidth=1, alpha=0.6))
gc.collect()
f = plt.figure(figsize=(16, 10))



mask = np.triu(np.ones_like(train.corr(), dtype=bool))



with sns.axes_style("white"):

    sns.heatmap(train.corr(), mask=mask, square=True, cmap = 'YlGnBu', annot=True);

    plt.title("Train Data Feature Correlation")
f = plt.figure(figsize=(16, 10))



mask = np.triu(np.ones_like(questions.corr(), dtype=bool))



with sns.axes_style("white"):

    sns.heatmap(questions.corr(), mask=mask, square=True, cmap = 'YlOrBr', annot=True);

    plt.title("Questions Data Feature Correlation")
f = plt.figure(figsize=(16, 10))



mask = np.triu(np.ones_like(lectures.corr(), dtype=bool))



with sns.axes_style("white"):

    sns.heatmap(lectures.corr(), mask=mask, square=True, cmap = 'icefire', annot=True);

    plt.title("Lectures Data Feature Correlation")
gc.collect()