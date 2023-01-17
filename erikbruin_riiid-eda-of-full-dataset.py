



import numpy as np

import pandas as pd

import riiideducation

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.style as style

style.use('fivethirtyeight')

import seaborn as sns

import os



import os

for dirname, _, filenames in os.walk('/kaggle/input/riiid-test-answer-prediction'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%%time



train = pd.read_pickle("../input/riiid-train-data-multiple-formats/riiid_train.pkl.gzip")



print("Train size:", train.shape)
train.memory_usage(deep=True)
train.info()
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].astype('bool')



train.memory_usage(deep=True)
%%time



questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')

lectures = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')

example_test = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv')

example_sample_submission = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_sample_submission.csv')
train.sample(10)
print(f'We have {train.user_id.nunique()} unique users in our train set')
train.content_type_id.value_counts()
print(f'We have {train.content_id.nunique()} content ids in our train set, of which {train[train.content_type_id == False].content_id.nunique()} are questions.')
cids = train.content_id.value_counts()[:30]



fig = plt.figure(figsize=(12,6))

cids.plot.bar()

plt.title("Thirty most used content id's")

plt.xticks(rotation=90)

plt.show()
print(f'We have {train.task_container_id.nunique()} unique Batches of questions or lectures.')
train.user_answer.value_counts()
#1 year = 31536000000 ms

ts = train['timestamp']/(31536000000/12)

fig = plt.figure(figsize=(12,6))

ts.plot.hist(bins=100)

plt.title("Histogram of timestamp")

plt.xticks(rotation=0)

plt.xlabel("Months between this user interaction and the first event completion from that user")

plt.show()
correct = train[train.answered_correctly != -1].answered_correctly.value_counts()



fig = plt.figure(figsize=(12,4))

correct.plot.barh()

plt.title("Questions answered correctly")

plt.xticks(rotation=0)

plt.show()
bin_labels_5 = ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5']

train['ts_bin'] = pd.qcut(train['timestamp'], q=5, labels=bin_labels_5)



#make function that can also be used for other fields

def correct(field):

    correct = train[train.answered_correctly != -1].groupby([field, 'answered_correctly'], as_index=False).size()

    correct = correct.pivot(index= field, columns='answered_correctly', values='size')

    correct['Percent_correct'] = round(correct.iloc[:,1]/(correct.iloc[:,0] + correct.iloc[:,1]),2)

    correct = correct.sort_values(by = "Percent_correct", ascending = False)

    correct = correct.iloc[:,2]

    return(correct)



bins_correct = correct("ts_bin")

bins_correct = bins_correct.sort_index()



fig = plt.figure(figsize=(12,6))

bins_correct.plot.bar()

plt.title("Percent answered_correctly for 5 bins of timestamp")

plt.xticks(rotation=0)

plt.show()
task_id_correct = correct("task_container_id")



fig = plt.figure(figsize=(12,6))

task_id_correct.plot.hist(bins=40)

plt.title("Histogram of percent_correct grouped by Batch")

plt.xticks(rotation=0)

plt.show()
user_percent = train[train.answered_correctly != -1].groupby('user_id')['answered_correctly'].agg(Mean='mean', Answers='count')

print(f'the highest number of questions answered by a user is {user_percent.Answers.max()}')

user_percent = user_percent.query('Answers <= 1000').sample(n=200, random_state=1)



fig = plt.figure(figsize=(12,6))

x = user_percent.Answers

y = user_percent.Mean

plt.scatter(x, y, marker='o')

plt.title("Percent answered correctly versus number of questions answered")

plt.xticks(rotation=0)

plt.xlabel("Number of questions answered")

plt.ylabel("Percent answered correctly")

z = np.polyfit(x, y, 1)

p = np.poly1d(z)

plt.plot(x,p(x),"r--")



plt.show()

train[train.answered_correctly != -1].prior_question_had_explanation.value_counts()
pq = train[train.answered_correctly != -1].groupby(['prior_question_had_explanation']).agg({'answered_correctly': ['mean']})

fig = plt.figure(figsize=(12,4))

pq.plot.barh(legend=None)

plt.title("Answered_correctly versus Prior Question had explanation")

plt.xlabel("Percent answered correctly")

plt.ylabel("Prior question had explanation")

plt.xticks(rotation=0)

plt.show()
pq = train[train.answered_correctly != -1]

pq = pq[['prior_question_elapsed_time', 'answered_correctly']]

pq = pq.groupby(['answered_correctly']).agg({'answered_correctly': ['count'], 'prior_question_elapsed_time': ['mean']})

pq
questions.head()
questions.shape
questions[questions.tags.isna()]
train.query('content_id == "10033" and answered_correctly != -1')
questions['tags'] = questions['tags'].astype(str)



tags = [x.split() for x in questions[questions.tags != "nan"].tags.values]

tags = [item for elem in tags for item in elem]

tags = set(tags)

print(f'There are {len(tags)} different tags')
correct = train[train.answered_correctly != -1].groupby(["content_id", 'answered_correctly'], as_index=False).size()

correct = correct.pivot(index= "content_id", columns='answered_correctly', values='size')

correct.columns = ['Wrong', 'Right']

correct = correct.fillna(0)

correct[['Wrong', 'Right']] = correct[['Wrong', 'Right']].astype(int)

questions = questions.merge(correct, left_on = "question_id", right_on = "content_id", how = "left")

questions.head()
tags_df = pd.DataFrame()

tags = list(tags)



for i in range(len(tags)):

    df = questions[questions.tags.str.contains(tags[i])].agg({'Wrong': ['sum'], 'Right': ['sum']})

    df['tag'] = tags[i]

    df = df.set_index('tag')

    tags_df = tags_df.append(df)



tags_df['Percent_correct'] = tags_df.Right/(tags_df.Right + tags_df.Wrong)

tags_df = tags_df.sort_values(by = "Percent_correct")



tags_df.head()
select_rows = list(range(0,10)) + list(range(178, tags_df.shape[0]))

tags_select = tags_df.iloc[select_rows,2]



fig = plt.figure(figsize=(12,6))

x = tags_select.index

y = tags_select.values

clrs = ['red' if y < 0.6 else 'green' for y in tags_select.values]

tags_select.plot.bar(x, y, color=clrs)

plt.title("Ten hardest and ten easiest tags")

plt.xlabel("Tag")

plt.ylabel("Percent answers correct of questions with the tag")

plt.xticks(rotation=90)

plt.show()
questions.part.value_counts()
part = questions.groupby('part').agg({'Wrong': ['sum'], 'Right': ['sum']})

part['Percent_correct'] = part.Right/(part.Right + part.Wrong)

part = part.iloc[:,2]



fig = plt.figure(figsize=(12,6))

part.plot.bar()

plt.title("Percent_correct by part")

plt.xlabel("Part")

plt.ylabel("Percent answers correct")

plt.xticks(rotation=0)

plt.show()
lectures.head()
example_test.shape
example_test.head()
batches_test = set(list(example_test.task_container_id.unique()))

batches_train = set(list(train.task_container_id.unique()))

print(f'All batches in example_test are also in train is {batches_test.issubset(batches_train)}.')
user_test = set(list(example_test.user_id.unique()))

user_train = set(list(train.user_id.unique()))



print(f'User_ids in example_test but not in train: {user_test - user_train}.')
example_sample_submission.shape
example_sample_submission.head()