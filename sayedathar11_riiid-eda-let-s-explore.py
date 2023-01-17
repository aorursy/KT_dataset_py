## Youtube Video To be added ! 

import warnings

from IPython.display import YouTubeVideo

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from IPython.display import HTML

from IPython.display import YouTubeVideo



YouTubeVideo('cjCwpP3cUpg', width=800, height=300)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

from IPython.display import display

import random

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import time

import os

import matplotlib.pyplot as plt

import matplotlib_venn as venn

sns.set()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("Loading Train Data ....")

df_train = pd.read_csv("../input/riiid-test-answer-prediction/train.csv" , nrows = 10**6)

print("Train Data Info")

print("***************************")

print(df_train.info())
print("Number of Null Objects in Elapsed Time Column are : ",df_train.prior_question_elapsed_time.isna().sum())

print("Number of Null Objects in Had Explanation Column are : ",df_train.prior_question_had_explanation.isna().sum())
df_train.head()
unique_users_list = df_train.user_id.unique().tolist()

print("Number of Unique Users are :" , len(unique_users_list))

print("{:>20}: {:>8}".format('Percentage of Unique Users are',(len(unique_users_list)/(df_train.shape[0])) *100))
df_users_content_group = df_train.groupby(['content_type_id'])['user_id'].count().reset_index()

df_users_content_group.columns = ['content_type_id' , 'count']

fig = px.bar(

    df_users_content_group, 

    x='content_type_id', 

    y="count", 

    color = "content_type_id",

    orientation='v', 

    title='User Counts based on Content_type_id', 

    width=500,

    height=500

)



fig.show()
unique_content_list = df_train.content_id.unique().tolist()

print("Number of Unique Conents are :" , len(unique_content_list))

#print(%.3f"Percentage of Unique Questions are : " , (len(unique_questions_list))/(df_train.shape[0]))

# Let's check our memory usage

print("{:>20}: {:>8}".format('Percentage of Unique Content are',len(unique_content_list)/(df_train.shape[0]) *100))
## Lets Now plot top content like top 10 content and their distribution based on occurence

df_top_content = df_train.groupby('content_id')['user_id'].count().reset_index().sort_values(by ='user_id', ascending=False)[:5]

df_top_content['content_id'] = df_top_content['content_id'].astype('category')

df_top_content.columns = ['content_id' , 'count']

fig = px.bar(

    df_top_content, 

    x='content_id', 

    y="count", 

    orientation='v', 

    title='Top 5 Content Ids Having most user', 

    width=800,

    height=800

)



fig.show()
df_ques_only = df_train.loc[df_train['content_type_id'] == 0]

df_top_ques = df_ques_only.groupby('content_id')['user_id'].count().reset_index().sort_values(by ='user_id', ascending=False)[:5]

df_top_ques['content_id'] = df_top_ques['content_id'].astype('category')

df_top_ques.columns = ['content_id' , 'count']

fig = px.bar(

    df_top_ques, 

    x='content_id', 

    y="count", 

    orientation='v', 

    title='Top 5 Question Ids Having most user', 

    width=800,

    height=800

)



fig.show()
correct_answer_stu = df_train.loc[df_train['answered_correctly'] == 1 ].shape[0]

print("Number of Correct Answer Students  are :" ,correct_answer_stu)

print("{:>20}: {:>8}".format('Percentage of Correct Answers',(correct_answer_stu)/(df_train.shape[0]) *100))
df_ques_only = df_train.loc[df_train['content_type_id'] == 0]

df_correct_ans = df_ques_only.groupby('answered_correctly')['user_id'].count().reset_index().sort_values(by ='user_id', ascending=False)

df_correct_ans

df_correct_ans.columns = ['answered_correctly' , 'count']

fig = px.bar(

    df_correct_ans, 

    x='answered_correctly', 

    y="count", 

    orientation='v', 

    title='User Counts Based on Correctly answer question', 

    width=500,

    height=500

)



fig.show()
df_questions_user = df_train.loc[df_train['content_type_id'] == 0 ]

df_lectures_user = df_train.loc[df_train['content_type_id'] == 1 ]

df_questions_user_id_set = set(df_questions_user['user_id'])

df_lectures_user_id_set = set(df_lectures_user['user_id'])

venn.venn2([df_questions_user_id_set,df_lectures_user_id_set],set_labels=('Question','Lectures'))
df_sample_stu = df_train.loc[df_train['user_id'] == 115]

unique_questions_list = df_sample_stu.content_id.unique().tolist()

print("Number of Unique Questions by Student 115 are :" , len(unique_questions_list))

#print(%.3f"Percentage of Unique Questions are : " , (len(unique_questions_list))/(df_train.shape[0]))

# Let's check our memory usage

print("{:>20}: {:>8}".format('Percentage of Unique Questions Seen by 115 are',len(unique_questions_list)/(df_sample_stu.shape[0]) *100))

correct_answer_stu = df_sample_stu.loc[df_sample_stu['answered_correctly'] == 1 ].shape[0]

print("Number of Correct Answer by 115   are :" ,correct_answer_stu)

print("{:>20}: {:>8}".format('Percentage of Correct Answers by 115 are',(correct_answer_stu)/(df_sample_stu.shape[0]) *100))
df_ques = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')

df_ques.head()
print("Number of Unique Questions are " , df_ques.shape[0])

train_ques_set = set(df_train['content_id'])

all_ques_set = set(df_ques['question_id'])



not_in_train_ques = list(all_ques_set.difference(train_ques_set))

print('{}:{}'.format("The Number of Questions Not in Train Set Are" , len(not_in_train_ques)))
df_ques.head()
df_ques_bundle = df_ques.groupby('part')['question_id'].count().reset_index().sort_values(by ='question_id', ascending=False)

df_ques_bundle.columns = ['part' ,'count']

df_ques_bundle.plot.bar(x = 'part' , y = 'count', title = "Number of Questions by Parts")
# Making List of Unique Tags !

tag_list = []

for i in df_ques.index:

    temp = str(df_ques['tags'][i]).split(" ")

    for tag in temp:

        if tag not in tag_list:

            tag_list.append(tag)

## Peaking Over Few Tags 

tag_list[:10]
tag_dict = {key:0 for key in tag_list}

for i in df_ques.index:

    temp = str(df_ques['tags'][i]).split(" ")

    for tag in temp:

        tag_dict[tag] +=1

# sorting dict 

sorted_tag_dict = {k: v for k, v in sorted(tag_dict.items(), key=lambda item: item[1] , reverse=True)}

## Lets Now Get Minimum and Maximum appearance of Tags



print("Maximum Appearance of a Tag is :" , max(sorted_tag_dict.values()))

print("Minimum Appearance of a Tag is :" , min(sorted_tag_dict.values()))

print("Tag Having Maximum Appearance is :" , list(sorted_tag_dict.keys())[0] )

print("Tag Having Minimum Appearance is :" , list(sorted_tag_dict.keys())[-1] )
# Lets Take Top 10 Tags and plot their occurences

import itertools

top_10_tags = dict(itertools.islice(sorted_tag_dict.items(), 10))

df_top_10 = pd.DataFrame(list(top_10_tags.items()) , columns=['Tag' , 'appearances']) 

import seaborn as sns

ax = sns.barplot(x="Tag", y="appearances", data=df_top_10)

#sorted_tag_dict.keys()[0]

list(sorted_tag_dict.keys())[0] 
df_train = df_train.rename(columns = {'content_id':'question_id'})

df_comb = df_train.merge(df_ques , on = 'question_id' , how = 'left')

df_comb.head()
df_comb = df_comb.loc[df_comb['content_type_id'] == 0 ] # Take only questions Data 

df_correct_answer_data = df_comb.loc[df_comb['answered_correctly'] == 1]

df_incorrect_answer_data = df_comb.loc[df_comb['answered_correctly'] == 0]
# Making List of Unique Tags !

correct_answer_tag_list = []

for i in df_correct_answer_data.index:

    temp = str(df_correct_answer_data['tags'][i]).split(" ")

    for tag in temp:

        if tag not in correct_answer_tag_list:

            correct_answer_tag_list.append(tag)

# Making List of Unique Tags !

incorrect_answer_tag_list = []

for i in df_incorrect_answer_data.index:

    temp = str(df_incorrect_answer_data['tags'][i]).split(" ")

    for tag in temp:

        if tag not in incorrect_answer_tag_list:

            incorrect_answer_tag_list.append(tag)

print("Is Two Tag Lists of Correct and Incorrect answers Equal(Having Same Elements ! ) ?" )

print("Answer is :",set(correct_answer_tag_list) == set(incorrect_answer_tag_list))


venn.venn2([set(correct_answer_tag_list),set(incorrect_answer_tag_list)],set_labels=('Correct Answer Tags','Incorrect Answer Tags'))
df_lectures = df_comb.loc[df_comb['content_type_id'] == 1] 

df_questions = df_comb.loc[df_comb['content_type_id'] == 0] # Take only questions Data 



# Making List of Unique Questions Tags !

questions_tag_list = []

for i in df_questions.index:

    temp = str(df_questions['tags'][i]).split(" ")

    for tag in temp:

        if tag not in questions_tag_list:

            questions_tag_list.append(tag)

# Making List of Unique Lectures Tag

lectures_tag_list = []

for i in df_lectures.index:

    temp = str(df_lectures['tags'][i]).split(" ")

    for tag in temp:

        if tag not in lectures_tag_list:

            lectures_tag_list.append(tag)

venn.venn2([set(questions_tag_list),set(lectures_tag_list)],set_labels=('Answer Tags','Lectures Tags'))
test_example = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')
test_example.head()
test_example_user_id = set(test_example['user_id'])

train_user_id = set(df_train['user_id'])

print(test_example_user_id.difference(train_user_id))