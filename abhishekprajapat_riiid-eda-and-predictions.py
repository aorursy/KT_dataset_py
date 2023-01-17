import os

import gc

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from collections import Counter

from wordcloud import WordCloud

from tqdm.notebook import tqdm
os.listdir('../input/riiid-test-answer-prediction')
train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', nrows=1000000)

lectures = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')

questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')

example_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')

example_sample_submission = pd.read_csv('../input/riiid-test-answer-prediction/example_sample_submission.csv')
example_sample_submission.head(2)
example_test.head(2)
train_df.head()
if len(train_df) == len(train_df.row_id.unique()):

    print('row_id column is the key')
print('total train samples ', len(train_df))
print("No of students = ", len(train_df['user_id'].unique()))
# idstribution of class labels

sns.set()

plt.figure(figsize=(10,6))

sns.countplot(data=train_df, x='answered_correctly', hue='prior_question_had_explanation')

plt.title('label distribution')
# distribution of number of samples per student

sns.set()

fig = plt.figure(figsize=(15,6))

fig = sns.kdeplot(train_df.groupby(by='user_id').count()['row_id'], shade=True, gridsize=50, color='g', legend=False)

fig.figure.suptitle("User_id distribution", fontsize = 20)

plt.xlabel('User_id counts', fontsize=16)

plt.ylabel('Probability', fontsize=16);
# How many question does each student attempt

df = train_df[train_df['content_type_id'] == 0]



df = df.groupby(by='user_id').count()



fig = plt.figure(figsize=(15,6))

fig = sns.kdeplot(df['row_id'], shade=True, gridsize=50, color='r', legend=False)

fig.figure.suptitle("User attempted questions distribution", fontsize = 20)

plt.xlabel('Questions counts', fontsize=16)

plt.ylabel('Probability', fontsize=16)

plt.legend(['Questions Attempted','Questions Correctly answered'])
# distribution of correct and incorrect and no answers

df = train_df[train_df['content_type_id'] == 0]



df2 = df[df['answered_correctly'] == 1]

df3 = df[df['answered_correctly'] == 0]



df2 = df2.groupby(by='user_id').count()

df3 = df3.groupby(by='user_id').count()



fig = plt.figure(figsize=(15,6))

fig = sns.kdeplot(df2['row_id'], shade=True, gridsize=50, color='b', legend=False)

fig = sns.kdeplot(df3['row_id'], shade=True, gridsize=50, color='r', legend=False)



fig.figure.suptitle("User attempted questions distribution", fontsize = 20)

plt.xlabel('Questions counts', fontsize=16)

plt.ylabel('Probability', fontsize=16)

plt.legend(['Correctly answered','Incorrectly answered'])
# What is the distribution of students correctly answering a question

values = []



df = train_df[train_df['content_type_id'] == 0]



for group, frame in df.groupby(by='user_id'):

    

    value = len(frame[frame['answered_correctly'] == 1]) / len(frame)

    values.append(value)
fig = plt.figure(figsize=(15,6))

fig = sns.kdeplot(values, shade=True, gridsize=50, color='c', legend=False)

fig.figure.suptitle("User correctly answering distribution", fontsize = 20)

plt.xlabel('Percent Correct', fontsize=16)



print('MEAN: ', np.mean(values))

print("MAX: ", np.max(values))

print('MIN: ', np.min(values))
# What precent of students see explanations



values = []



df = train_df[train_df['content_type_id'] == 0]



for group, frame in df.groupby(by='user_id'):

    

    value = len(frame[frame['prior_question_had_explanation'] == True]) / len(frame)

    values.append(value)
plt.figure(figsize=(10,6))

sns.distplot(values, kde=False)

plt.title('Distribution if students who see x percent of explanations')

plt.xlabel('Percent explanation seen out of attempted questions')

plt.ylabel('Counts')
print("Total task container id's having questions: ", len((train_df['task_container_id'][train_df['content_type_id'] == 0]).unique()))
questions.head()
questions.info()
print('Total number of questions: ', len(questions['question_id'].unique()))

print("Total number of unique bundles: ", len(questions['bundle_id'].unique()))
fig = plt.figure(figsize=(10,6))

fig = sns.countplot(questions.groupby(by='bundle_id').count()['question_id'])

plt.xlabel('bundle_id')

plt.title('Question in bundles');
fig = plt.figure(figsize=(10,6))

fig = sns.countplot(questions['correct_answer'])

plt.xlabel('Answers')

plt.title('Correct Answers distribution')
# Distribution of number of tags per question

# I think of tags as subject includings like (maths, algebra, numbers. history etc. in encoded form)



no_of_tags = []

for i in questions['tags']:

    value = len(str(i).strip().split(' '))

    no_of_tags.append(value)
plt.figure(figsize=(10,6))

sns.countplot(no_of_tags)

plt.xlabel('No of tags')

plt.title('No of tags per question')
# distribution of tags



total = []



for i in questions['tags']:

    for j in str(i).strip().split(' '):

        total.append(j)

        

keys = set(total)

final = {}

for i in keys:

    final[i] = total.count(i)

    

values = sorted(final.items(), key=lambda x: x[1], reverse=True)

d = []

for i in values:

    d.append(i[1])
plt.figure(figsize=(10,6))

px.line(d, title='Tags distribution')
# Most commmon tags

tags = WordCloud().generate_from_frequencies(final)

px.imshow(tags, title='Most frequent Tags')
lectures.head()
lectures.info()
print('Total no. of lectures: ', len(lectures['lecture_id'].unique()))

print('Only one tag per row: ', )

print('Total no. of tags in lecture: ', len(lectures['tag'].unique()))
# distribution of lecture tags



total = []



for i in lectures['tag']:

    for j in str(i).strip().split(' '):

        total.append(j)

        

keys = set(total)

final = {}

for i in keys:

    final[i] = total.count(i)

    

values = sorted(final.items(), key=lambda x: x[1], reverse=True)

d = []

for i in values:

    d.append(i[1])
plt.figure(figsize=(10,6))

px.line(d, title='Tags distribution')
# Most common tags

tags = WordCloud().generate_from_frequencies(final)

px.imshow(tags, title='Most frequent lecture Tags')
# Looking at parts

print('Total type of parts: ', len(lectures.part.unique()))

print('Values of parts: ', lectures.part.unique())
# Counts of parts

plt.figure(figsize=(10,6))

sns.countplot(lectures['part'])

plt.title('Counts of Parts in Lectures');
# how many different unique tags does each part have or do they common tags as well ?



no_unique_tags_l = []

unique_tags_l = {}

groups = []



for group, frame in lectures.sort_values(by='part').groupby(by='part'):

    

    unique_tags = frame['tag'].unique()

    no_unique_tags = len(unique_tags)

    

    unique_tags_l[group] = unique_tags

    no_unique_tags_l.append(no_unique_tags)

    groups.append(group)

    

no_unique_tags_l = pd.DataFrame(no_unique_tags_l, columns=['count'])

no_unique_tags_l['group'] = groups
# Number of unique tags in each part ( here unique means internally part wise)

plt.figure(figsize=(10,6))

sns.barplot(x=no_unique_tags_l['group'], y=no_unique_tags_l['count'])

plt.title('No. of unique tags in each part')
final_unqiue = []

parts = []





for part, array in unique_tags_l.items():

    

    unique_tags = []

        

    other_parts = list(unique_tags_l.keys())

    final = set(other_parts)

    final.remove(part)

    

    for j in final:

        

        for k in array:

            

            if k not in unique_tags_l[j]:

                

                unique_tags.append(k)

    

    final_unqiue.append(len(unique_tags))

    parts.append(part)

    

final_unqiue = pd.DataFrame(final_unqiue, columns=['tags'])

final_unqiue['part'] = parts
# let's see how many tags are there in each part which are not in any other part



plt.figure(figsize=(10,6))

sns.barplot(x=no_unique_tags_l['group'], y=no_unique_tags_l['count'])

plt.title('No. of unique tags in each part not in any other');
# Finally let's look at type_of lectures



px.bar(lectures, x='type_of', color=lectures['type_of'], labels={'value':'type_of'}, title='Type of lectures distribution Overall')
px.bar(lectures, x='type_of', color=lectures['type_of'], labels={'value':'type_of'}, title='Type of lectures distribution based on each part', facet_col='part')
train_df.head()
# we will see first 8 students for trends

no_students = 8

scores = []

user_ids = []

question_attempted_l = []

correctly_answered_l = []

prior_questions_explanations = []



for count, (group, frame) in enumerate(train_df.groupby(by='user_id')):

    

    if count == no_students:

        break

    

    frame = frame.sort_values(by='timestamp')

    

    percentage = []

    question_attempted = []

    correctly_answered = []

    explanations = []

    attempted = 0

    correct_answers = 0

    explanation = 0

    

    df = frame[frame['content_type_id'] == 0]

    

    for answered_correctly, had_explanation in zip(df['answered_correctly'], df['prior_question_had_explanation']):

        

        attempted += 1

        question_attempted.append(attempted)

        

        if answered_correctly == 1:

            correct_answers += 1

            

        if had_explanation:

            explanation += 1

            

        correctly_answered.append(correct_answers)

            

        percent = correct_answers / attempted * 100

        percentage.append(percent)

        explanations.append(explanation)

        

    

    scores.append(percentage)

    user_ids.append(group)

    question_attempted_l.append(question_attempted)

    correctly_answered_l.append(correctly_answered)

    prior_questions_explanations.append(explanations)
# Trend in attempted question and correctly answering



plt.figure(figsize=(15,20))



for i in range(1,9):

    plt.subplot(4,2,i)

    plt.plot(question_attempted_l[i-1], question_attempted_l[i-1], label='Questions attempted')

    plt.plot(question_attempted_l[i-1], correctly_answered_l[i-1], label='Questions correctly answered')

    plt.plot(question_attempted_l[i-1], scores[i-1], label='Percentage correctly answered')

    plt.plot(question_attempted_l[i-1], prior_questions_explanations[i-1], label='Prior_questions_explanations')

    plt.legend()

    plt.ylim(0,100)

    plt.xlim(0,50)

    plt.tight_layout(pad = 2)

    plt.title(f'user_id: {user_ids[i-1]}')
# Does students time spend on answering prior questions



no_students = 8

time_spend_l = []



for count, (group, frame) in enumerate(train_df.groupby(by='user_id')):

    

    if count == no_students:

        break

    

    frame = frame.sort_values(by='timestamp')

    total_time_spends = []

    time_spends = 0

    

    for time_spend in frame['prior_question_elapsed_time'][frame['content_type_id'] == 0]:

        

        if time_spend > 0:

            time_spends += time_spend

            total_time_spends.append(time_spends)

        

    

    time_spend_l.append(total_time_spends)
time_spend_l = np.array(time_spend_l)

for index, value in enumerate(time_spend_l):

    time_spend_l[index] = np.array(time_spend_l[index]) / 10000
# Trend in time spend with percentage



plt.figure(figsize=(15,20))



for i in range(1,9):

    plt.subplot(4,2,i)

    plt.plot(question_attempted_l[i-1], correctly_answered_l[i-1], label='Questions correctly answered')

    plt.plot(question_attempted_l[i-1][1:], time_spend_l[i-1], label='time spend in 10000')

    plt.plot(question_attempted_l[i-1], scores[i-1], label='Percentage correctly answered')

    plt.legend()

    plt.ylim(0,100)

    plt.xlim(0,50)

    plt.tight_layout(pad = 2)

    plt.title(f'user_id: {user_ids[i-1]}')
print('Correlation among variables of train file')

train_df.corr().style.background_gradient(cmap='Blues')
for element in dir():

    if element[0:2] != "__":

        del globals()[element]
import os

import gc

import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import lightgbm as lgb

from multiprocessing import Pool
lectures = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')

questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
train_df = pd.read_csv("../input/riiid-test-answer-prediction/train.csv",

                        chunksize=500000)
print("Making Mapping dictionaries: \n")



conversion1 = {}

conversion2 = {}

conversion3 = {}

conversion4 = {}



for i in tqdm(lectures['lecture_id']):

    conversion1[i] = lectures['tag'][lectures['lecture_id'] == i].values[0]

    conversion2[i] = lectures['part'][lectures['lecture_id'] == i].values[0]

    conversion3[i] = lectures['type_of'][lectures['lecture_id'] == i].values[0]

    conversion4[i] = -1



for i in tqdm(questions['question_id']):

    conversion1[i] = questions['tags'][questions['question_id'] == i].values[0]

    conversion2[i] = questions['part'][questions['question_id'] == i].values[0]

    conversion3[i] = 'question'

    conversion4[i] = questions['correct_answer'][questions['question_id'] == i].values[0]
def preprocess_train(df, count):

    

    df['tags'] = df['content_id'].map(conversion1)

    df['part'] = df['content_id'].map(conversion2)

    

    df = (df.assign(tags = df['tags'].str.strip().str.split(' '))

         .explode('tags')

         .reset_index(drop=True))

    

    df.fillna(value=-1, inplace=True)

    

    df['prior_thing'] = 0

    

    for group, frame in df.groupby(by='user_id'):



        frame = frame.sort_values(by='timestamp')

        frame['prior_thing'] = frame['content_type_id'].shift(1)



    df['prior_type'] = 'question'

    

    df['prior_type'] = df['content_id'].map(conversion3)

    

    for group, frame in df.groupby(by='user_id'):



        frame = frame.sort_values(by='timestamp')

        frame['prior_type'] = frame['prior_type'].shift(1)

    

    # Now after extracting information we will drop all columns other than those with questions

    df = df[df['content_type_id'] == 0]

    

    df['correct_answer'] = df['content_id'].map(conversion4)



    df.drop(columns = ['row_id', 'content_type_id', 'user_answer', 'content_id'], inplace=True)

    

    df.fillna(value=-1, inplace=True)

    

    df['tags'] = df['tags'].astype(np.float32)



    

    le1 = preprocessing.LabelEncoder()

    le2 = preprocessing.LabelEncoder()



    df['prior_question_had_explanation'] = le1.fit_transform(df['prior_question_had_explanation'])

    df['prior_type'] = le2.fit_transform(df.loc[:, 'prior_type'].values)

    

    return df, le1, le2
categorical_features = ['user_id', 'task_container_id', 'prior_question_had_explanation', 'tags', 'part',

                       'prior_thing', 'correct_answer']



model = None

count = 0



params = {

    'keep_training_booster' : True,

    'objective': 'binary',

    'verbose':100,

    'learning_rate': 0.1,

}



# I we will run the model for 2 rounds, you could run it for more. (It takes time)



for df in train_df:

    

    if count == 2:

        break

        



    

    df, le1, le2 = preprocess_train(df, count)

    

    count += 1

    

  

    xtrain, xvalid, ytrain, yvalid = train_test_split(df.drop(columns='answered_correctly'),

                                                     df['answered_correctly'], test_size=0.2, random_state=1)

    

    lgb_train = lgb.Dataset(xtrain, ytrain, categorical_feature=categorical_features)

    lgb_valid = lgb.Dataset(xvalid, yvalid, categorical_feature=categorical_features)

    

    model = lgb.train(params,

            init_model=model,

            train_set=lgb_train,

            valid_sets=lgb_valid,

            verbose_eval=10,

            num_boost_round=100)

    

    print("ROC SCORE: ", roc_auc_score(yvalid, model.predict(xvalid)))

    

    del df, xtrain, ytrain, xvalid, yvalid, lgb_train, lgb_valid

    gc.collect()