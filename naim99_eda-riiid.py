import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from collections import Counter

from wordcloud import WordCloud
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
train_df.info()
train_df.describe()
print("No of students = ", len(train_df['user_id'].unique()))
print("distribution of number of samples per student")
sns.set()

fig = plt.figure(figsize=(15,6))

fig = sns.kdeplot(train_df.groupby(by='user_id').count()['row_id'], shade=True, gridsize=50, color='g', legend=False)

fig.figure.suptitle("User_id distribution", fontsize = 20)

plt.xlabel('User_id counts', fontsize=16)

plt.ylabel('Probability', fontsize=16);
print("How many question does each student attempt")
df = train_df[train_df['content_type_id'] == 0]



df = df.groupby(by='user_id').count()



fig = plt.figure(figsize=(15,6))

fig = sns.kdeplot(df['row_id'], shade=True, gridsize=50, color='r', legend=False)

fig.figure.suptitle("User attempted questions distribution", fontsize = 20)

plt.xlabel('Questions counts', fontsize=16)

plt.ylabel('Probability', fontsize=16)

plt.legend(['Questions Attempted','Questions Correctly answered'])
print("distribution of correct and incorrect and no answers")
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
print("What is the distribution of students correctly answering a question ?")
values = []



df = train_df[train_df['content_type_id'] == 0]



for group, frame in df.groupby(by='user_id'):

    

    value = len(frame[frame['answered_correctly'] == 1]) / len(frame)

    values.append(value)
fig = plt.figure(figsize=(15,6))

fig = sns.kdeplot(values, shade=True, gridsize=50, color='b', legend=False)

fig.figure.suptitle("User correctly answering distribution", fontsize = 20)

plt.xlabel('Percent Correct', fontsize=16)



print('MEAN: ', np.mean(values))

print("MAX: ", np.max(values))

print('MIN: ', np.min(values))
print("What precent of students see explanations ?")
values = []



df = train_df[train_df['content_type_id'] == 0]



for group, frame in df.groupby(by='user_id'):

    

    value = len(frame[frame['prior_question_had_explanation'] == True]) / len(frame)

    values.append(value)
px.histogram(values)
print("Total task container id's having questions: ", len((train_df['task_container_id'][train_df['content_type_id'] == 0]).unique()))
questions.head()
questions.info()
questions.describe()
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
print("Distribution of number of tags per question")

print("I think of tags as subject includings like (maths, algebra, numbers. history etc. in encoded form)")
no_of_tags = []

for i in questions['tags']:

    value = len(str(i).strip().split(' '))

    no_of_tags.append(value)
plt.figure(figsize=(10,6))

sns.countplot(no_of_tags)

plt.xlabel('No of tags')

plt.title('No of tags per question')
print("distribution of tags")
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
tags = WordCloud().generate_from_frequencies(final)

px.imshow(tags, title='Most frequent Tags')
lectures.head()
lectures.info()
lectures.describe()
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

px.line(d, title='Tags distribution in lectures')
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
print("Finally let's look at type_of lecture")
px.bar(lectures, x='type_of', color=lectures['type_of'], labels={'value':'type_of'}, title='Type of lectures distribution Overall')
px.bar(lectures, x='type_of', color=lectures['type_of'], labels={'value':'type_of'}, title='Type of lectures distribution based on each part', facet_col='part')
train_df.head()
print("we will see first 8 students for trends")
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