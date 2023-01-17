import bq_helper

import matplotlib.pyplot as plt

import pandas as pd



stack_overflow_helper = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="stackoverflow")
python_trends_by_year_query = '''

    select 

        extract(year from questions.creation_date) as year,

        count(distinct questions.id) as python_questions_count,

        count(*) as python_answers_count

        FROM `bigquery-public-data.stackoverflow.posts_questions` questions

        left join `bigquery-public-data.stackoverflow.posts_answers` answers

        on questions.id = answers.parent_id

        where questions.tags like "%python%"

        group by year

        order by year desc

        

'''

stack_overflow_helper.estimate_query_size(python_trends_by_year_query)

python_trends_year = stack_overflow_helper.query_to_pandas_safe(python_trends_by_year_query)

python_trends_year.head()

python_trends_year['question_to_answer_ratio'] = python_trends_year.apply(lambda row: row.python_questions_count / row.python_answers_count, axis=1)

python_trends_year.head()
plt.plot('year', 'python_questions_count', data=python_trends_year)

plt.plot('year', 'python_answers_count', data=python_trends_year)

plt.plot('year', 'question_to_answer_ratio', data=python_trends_year)

plt.legend()
# The next thing we want to get is the most popular tags in 2018

python_tags_2018_query = '''

    select 

        questions.tags, questions.view_count, users.location

        from `bigquery-public-data.stackoverflow.posts_questions` questions

        inner join `bigquery-public-data.stackoverflow.users` users

        on users.id = questions.owner_user_id 

        where questions.tags like "%python%"

        and extract(year from questions.creation_date) = 2018

        order by questions.view_count desc

'''

python_tags_2018 = stack_overflow_helper.query_to_pandas_safe(python_tags_2018_query)

def split_location(row):

    if row.location == 'None' or row.location == '':

        return ""

    location = row.location.split(', ')

    if len(location) == 1:

        return location[0]

    return location[-1]



# Also, we need to transform location to country

python_tags_2018['country'] = python_tags_2018.apply(split_location, axis=1)

python_tags_2018 = python_tags_2018.drop(['location'], axis=1)

# Now, we need to count the number of tags per row

python_tags_2018['tag_count'] = python_tags_2018.apply(lambda row: len(row.tags.split('|')), axis=1)

python_tags_2018.head(10)

python_tag_group = python_tags_2018.groupby('tag_count')['view_count'].mean()



ax = python_tag_group.plot.barh(x='tag_count', y='view_count')

ax.set_xlabel("Mean view count")

ax.set_ylabel("Number of tags used")

ax.set_title("Question view counts by the number of tags")

python_tag_group = python_tags_2018.groupby('country').size().reset_index(name='count')

python_tag_group.sort_values(by='count', ascending=False).head(11)

# Time to find the most popular tags for python

# What we need to do now is to create a new dataframe

col_names = ['tag', 'country']

tag_list = []

for row in python_tags_2018.itertuples():

    for tag in row[1].split('|'):

        if tag != "python":

            tag_list.append({'tag': tag, 'country': row[3]})



tag_df = pd.DataFrame(tag_list, columns=col_names)

tag_df.head(5)
# Group by tags to view how in how many questions were the tags used.

tag_count_df = tag_df.groupby('tag').size().reset_index(name='count')

tag_count_df.sort_values(by=['count'], ascending=False).head(10)

india_tag_df = tag_df[tag_df['country'] == "India"]

india_tag_df = india_tag_df.groupby('tag').size().reset_index(name='count')

india_tag_df = india_tag_df.sort_values(by=['count'], ascending=False)

ax = india_tag_df.head(10)[::-1].plot.barh(x='tag', y='count')

ax.set_xlabel("Number of questions tagged")

ax.set_ylabel("Tags")

ax.set_title("Most used tags by developers in India when asking questions")

usa_tag_df = tag_df[(tag_df['country'] == "USA") | (tag_df['country'] == "United States")].groupby('tag').size().reset_index(name='count')

usa_tag_df = usa_tag_df.sort_values(by=['count'], ascending=False)

ax = usa_tag_df.head(10)[::-1].plot.barh(x='tag', y='count')

ax.set_xlabel("Number of questions tagged")

ax.set_ylabel("Tags")

ax.set_title("Most used tags by developers in the USA when asking questions")
germany_tag_df = tag_df[(tag_df['country'] == "Germany")].groupby('tag').size().reset_index(name='count')

germany_tag_df = germany_tag_df.sort_values(by=['count'], ascending=False)

ax = germany_tag_df.head(10)[::-1].plot.barh(x='tag', y='count')

ax.set_xlabel("Number of questions tagged")

ax.set_ylabel("Tags")

ax.set_title("Most used tags by developers in Germany when asking questions")