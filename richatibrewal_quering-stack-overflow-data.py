import bq_helper

from bq_helper import BigQueryHelper

import pandas as pd

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="stackoverflow")
# query_get_users = """ Select posts.id,posts.tags, posts.owner_user_id, posts.score, users.reputation

# from `bigquery-public-data.stackoverflow.posts_questions` as posts JOIN `bigquery-public-data.stackoverflow.users` as users

# on posts.owner_user_id = users.id

# where posts.owner_user_id Is NOT NULL and posts.tags is not null and Extract(Year from posts.creation_date) > 2018

# order by posts.id DESC

# limit 1000;

# """



# response_get_users = stackOverflow.query_to_pandas_safe(query_get_users,2)
# response_get_users.to_csv('question_users.csv',index=False)

# response_get_users
query_get_answers = """ Select answers.id as answer_id ,answers.owner_user_id as answer_owner_user_id, 

users_answer.display_name as answer_user_name, answers.score as answer_score , users_answer.reputation as answer_users_reputation,

questions.id as question_id,  questions.owner_user_id as question_owner_user_id, questions.score as question_score, 

users_question.display_name as question_user_name,

users_question.reputation as question_users_reputation, questions.tags

from `bigquery-public-data.stackoverflow.posts_questions` as questions

JOIN 

`bigquery-public-data.stackoverflow.posts_answers` as answers

on questions.id = answers.parent_id

JOIN

`bigquery-public-data.stackoverflow.users` as users_answer

on answers.owner_user_id = users_answer.id

JOIN 

`bigquery-public-data.stackoverflow.users` as users_question

on questions.owner_user_id = users_question.id

where answers.owner_user_id Is NOT NULL and questions.owner_user_id is not NULL and questions.tags is NOT NULL 

and Extract(Year from questions.creation_date)>2018

limit 10000

;



"""



response_get_answers = stackOverflow.query_to_pandas_safe(query_get_answers,30)

# for i in range(0, len(response_get_answers)):

#     response_get_answers.answers_text[i] = response_get_answers.answers_text[i].replace("\"", "''")

#     response_get_answers.question_text[i] = response_get_answers.question_text[i].replace("\"", "''")

    


response_get_answers.to_csv('answers_question_users_10000.csv', index=False)

len(response_get_answers)
query_get_tags = """select tag_name, id from `bigquery-public-data.stackoverflow.tags`;"""

response_tags = stackOverflow.query_to_pandas_safe(query_get_tags, 1)

response_tags

response_tags.to_csv('tags_name.csv', index=False)



response_tags['length'] = response_tags['tag_name'].str.len()

response_tags.sort_values('length', ascending=False, inplace=True)



tags = response_tags.tag_name.tolist()

tag_ids = response_tags.id.tolist()

def convertTagsToList(tagList, questionId):

    res = []

    for i in range(0, 55665):

       

        index = tagList.find(tags[i])

        if(index != -1):

            res.append([tags[i], tag_ids[i], questionId])

        if (len(res) >=5):

            break

            

    

    

    return res

        

tag_question = []

tagQuestion = []



for i in range(0, len(response_get_answers)):

    tagQuestion.append((response_get_answers.tags[i],response_get_answers.question_id[i]))

    

#     all_tags = convertTagsToList(tagList, questionId)

#     print (all_tags)

#     tag_question.extend(all_tags)

    
tagQuestionSet = set(tagQuestion)

tagQuestion = list(tagQuestionSet)

tagQuestion[0][1]
tagList = []



for i in range(0, len(tagQuestion)):

    tag = tagQuestion[i][0]

    questionId = tagQuestion[i][1]

    

    all_tags = convertTagsToList(tag, questionId)

#     print (i)

    tag_question.extend(all_tags)
tag_dataFrame = pd.DataFrame(tag_question, columns=['tag_name', 'tag_id', 'question_id'])

# tag_question
tag_dataFrame
tag_dataFrame.to_csv('question_tags_10000.csv',index=False)
# query_get_users_answers = """ Select answers.id, answers.owner_user_id, answers.score, users.reputation, questions.id

# from `bigquery-public-data.stackoverflow.posts_answers` as answers 

# JOIN 

# `bigquery-public-data.stackoverflow.users` as users

# JOIN

# `bigquery-public-data.stackoverflow.posts_questions` as questions

# on answers.owner_user_id = users.id and questions.id = answers.parent_id 

# where answers.owner_user_id Is NOT NULL and questions.owner_user_id IS NOT NULL

# and questions.tags is not null and Extract(Year from questions.creation_date) > 2018

# limit 1000;

# """



# response_get_answers = stackOverflow.query_to_pandas_safe(query_get_users,2)
# query_get_users_answers = """ Select answers.id as answers_id, answers.owner_user_id, answers.score, users.reputation, questions.id as question_id

# from `bigquery-public-data.stackoverflow.posts_answers` as answers 

# INNER JOIN 

# `bigquery-public-data.stackoverflow.users` as users

# on answers.owner_user_id = users.id

# INNER JOIN

# `bigquery-public-data.stackoverflow.posts_questions` as questions

# on questions.id = answers.parent_id 

# where answers.owner_user_id Is NOT NULL and questions.owner_user_id IS NOT NULL

# and questions.tags is not null and Extract(Year from questions.creation_date) > 2018

# limit 1000;

# """



# response_get_answers = stackOverflow.query_to_pandas_safe(query_get_users_answers,2)
# response_get_answers
# response_get_answers.to_csv('answers_question_users.csv',index=False)
# tag = 'c'



# query_get_questions = """ Select questions.owner_user_id, users.reputation as score

# from `bigquery-public-data.stackoverflow.posts_questions` as questions 

# JOIN 

# `bigquery-public-data.stackoverflow.users` as users

# on questions.owner_user_id = users.id

# where questions.owner_user_id Is NOT NULL

# and questions.tags like '%c%' and Extract(Year from questions.creation_date) > 2018 



# UNION ALL

# Select answers.owner_user_id, users.reputation as score

# from `bigquery-public-data.stackoverflow.posts_questions` as questions

# JOIN

# `bigquery-public-data.stackoverflow.posts_answers` as answers

# on questions.id = answers.parent_id

# JOIN

# `bigquery-public-data.stackoverflow.users` as users

# on answers.owner_user_id = users.id

# where answers.owner_user_id Is NOT NULL and questions.owner_user_id IS NOT NULL

# and questions.tags like '%c%' and Extract(Year from answers.creation_date) > 2018 

# order by score DESC

# limit 1000;

# """



# get_questions= stackOverflow.query_to_pandas_safe(query_get_questions,2)

# get_questions