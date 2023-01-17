from google.cloud import bigquery

client=bigquery.Client()

dataset_ref=client.dataset("stackoverflow", project="bigquery-public-data")

dataset=client.get_dataset(dataset_ref)

tables=[table.table_id for table in client.list_tables(dataset)]

tables
table_ref=dataset_ref.table("posts_answers")

answers=client.get_table(table_ref)

answers.schema
table_ref=dataset_ref.table("users")

answers=client.get_table(table_ref)

answers.schema
tags_query="""

                SELECT tags, COUNT(1) as num_tags

                FROM `bigquery-public-data.stackoverflow.posts_questions`

                GROUP BY tags

                ORDER BY num_tags DESC

            """

query_job = client.query(tags_query)

iterator=query_job.result(timeout=30)

tags_df=query_job.to_dataframe()

tags_df.head()
import wordcloud

import matplotlib.pyplot as plt
words=' '.join(tags_df.tags).lower()
cloud = wordcloud.WordCloud(background_color='black',

                            max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=300,

                            relative_scaling=.5).generate(words)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('tags_stackoverflow.png')

plt.imshow(cloud);
table_ref=dataset_ref.table("posts_questions")

questions=client.get_table(table_ref)

questions.schema
num_answers_query="""SELECT q.tags,COUNT(1) as number_answers, 

                    FROM `bigquery-public-data.stackoverflow.posts_answers` a

                    INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` q

                    ON a.parent_id=q.id

                    GROUP BY q.tags

                    ORDER BY number_answers DESC"""

query_job=client.query(num_answers_query)

iterator=query_job.result(timeout=30)

num_answers_df=query_job.to_dataframe()
num_answers_df.head()
answer_per_questions_query="""

                            WITH num_tags AS(SELECT tags, COUNT(1) as num_tags

                                            FROM `bigquery-public-data.stackoverflow.posts_questions`

                                            GROUP BY tags

                            ), num_ans AS(SELECT q.tags,COUNT(1) as num_answers, 

                    FROM `bigquery-public-data.stackoverflow.posts_answers` a

                    INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` q

                    ON a.parent_id=q.id

                    GROUP BY q.tags)

                            SELECT t.tags, round(num_answers/num_tags,2) AS pct_ans

                            FROM num_tags t

                            INNER JOIN num_ans a

                            ON t.tags=a.tags

                            ORDER BY pct_ans DESC"""

answer_per_questions_df=client.query(answer_per_questions_query).result().to_dataframe()
answer_per_questions_df.head()
do_nothingers="""SELECT COUNT(1) AS nothinger_count

            FROM `bigquery-public-data.stackoverflow.users` u

            LEFT JOIN (

            SELECT DISTINCT owner_user_id

            FROM `bigquery-public-data.stackoverflow.posts_answers`

            union all

            SELECT DISTINCT owner_user_id

            FROM `bigquery-public-data.stackoverflow.posts_questions`)b

            ON u.id=b.owner_user_id

            WHERE b.owner_user_id is null

            """
questioners="""SELECT COUNT(DISTINCT q.owner_user_id) AS questioner_count

             FROM `bigquery-public-data.stackoverflow.posts_questions` q

             LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` a

             ON q.owner_user_id=a.owner_user_id

             WHERE a.owner_user_id is NULL"""
answerers="""SELECT COUNT(DISTINCT a.owner_user_id) AS questioner_count

             FROM `bigquery-public-data.stackoverflow.posts_answers` a

             LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` q

             ON q.owner_user_id=a.owner_user_id

             WHERE q.owner_user_id is NULL"""
answerers_questioners="""SELECT COUNT(DISTINCT q.owner_user_id) AS questioner_count

             FROM `bigquery-public-data.stackoverflow.posts_answers` a

             INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` q

             ON q.owner_user_id=a.owner_user_id

                """
total_users="""SELECT COUNT(1) as total

            FROM `bigquery-public-data.stackoverflow.users`"""

total=client.query(total_users).result().to_dataframe().iat[0,0]

print("Total number of users:{}".format(total))
import pandas as pd

user_type_df=pd.DataFrame(columns=["type","counts","percent"])
count=[]

percent=[]
#EXECUTE THE QUERIES

types={"do_nothingers":do_nothingers, "questioners":questioners, "answerers":answerers,"answerers_questioners":answerers_questioners}

for value in list(types.values()):

    query_job=client.query(value)

    num_count = query_job.result(timeout=45).to_dataframe().iat[0,0]

    count.append(num_count)

    percent.append(round(num_count/total*100,2))



user_type_df["type"]=list(types.keys())

user_type_df["counts"]=count

user_type_df["percent"]=percent

user_type_df
#Credit: https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_features.html

import matplotlib.pyplot as plt



explode = (0, 0.1, 0, 0) 



fig1, ax1 = plt.subplots()

ax1.pie(user_type_df["percent"], explode=explode, labels=user_type_df["type"], autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  



plt.show()
