# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # attractive vizualization of data representation

import wordcloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import bq_helper

stackoverflow = bq_helper.BigQueryHelper("bigquery-public-data", "stackoverflow")



#answers = pd.read_csv('../input/AnswersScores.csv', header=0)



# Any results you write to the current directory are saved as output.
stackoverflow.list_tables()

#stackoverflow.head("posts_questions",num_rows=5)
#Registered users yearly on Stack Overflow

percentageUsers = """select EXTRACT(YEAR FROM creation_date) Years, count(*) usersYearly,

ROUND(100*count(*)/9804894,1) as Register_User_Percentage

FROM `bigquery-public-data.stackoverflow.users`

where EXTRACT(YEAR FROM creation_date) <= 2018

group by EXTRACT(YEAR FROM creation_date)"""



peryearlyusers = stackoverflow.query_to_pandas(percentageUsers)

dfUsers  = pd.DataFrame(peryearlyusers)

#dfUsers

#dfUsers.plot.bar()

ax = sns.barplot(x = "Years",y = "Register_User_Percentage",

data = peryearlyusers,palette="coolwarm").set_title("Yearly percentage(%) of registered users of total users on Stack Overflow")

sns.set(rc={'figure.figsize':(11.7,8.27)})
#Active users yearly on Stack Overflow who have posted 5 or more questions or answers on SO

activeUsers = """WITH RunTotalTestData AS (

  SELECT * FROM UNNEST([STRUCT(1 AS Year, 1 AS runningTotal),

  (2008,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2008)),

  (2009,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2009)),

  (2010,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2010)),

  (2011,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2011)),

  (2012,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2012)),

  (2013,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2013)),

  (2014,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2014)),

  (2015,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2015)),

  (2016,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2016)),

  (2017,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2017)),

  (2018,(select count(*) from `bigquery-public-data.stackoverflow.users` where EXTRACT(YEAR FROM creation_date) = 2018))

  ]) 

)

select A.Years, A.runningTotalUsers UsersRunningTotalYearly,A.ActiveusersYearly ActiveUsersYearly, 

ROUND(100*A.ActiveusersYearly/A.runningTotalUsers,1) as ActiveUsersPercentageYearly from (

select resulttable.Years Years, count(resulttable.OwnerUserId) ActiveusersYearly,

(select B.runningTotal from RunTotalTestData B where B.Year = resulttable.Years) runningTotalUsers

    from (

    select EXTRACT(YEAR FROM posts.creation_date) Years, posts.owner_user_id OwnerUserId,  count(*) AskedQuestion5  

    from `bigquery-public-data.stackoverflow.posts_questions` AS posts 

    where posts.owner_user_id is not null 

    and EXTRACT(YEAR FROM posts.creation_date) <= 2018

    group by EXTRACT(YEAR FROM posts.creation_date), posts.Owner_User_Id

    union all

    select EXTRACT(YEAR FROM posts.creation_date) Years, posts.owner_user_id OwnerUserId,  count(*) AskedQuestion5  

    from `bigquery-public-data.stackoverflow.posts_answers` AS posts 

    where posts.owner_user_id is not null 

    and EXTRACT(YEAR FROM posts.creation_date) <= 2018

    group by EXTRACT(YEAR FROM posts.creation_date), posts.Owner_User_Id

    

        ) resulttable 

where resulttable.AskedQuestion5 > 4

group by resulttable.Years

) A

order by A.Years"""

yearlyactiveUsers = stackoverflow.query_to_pandas(activeUsers)

ax = sns.barplot(x = "Years",y = "ActiveUsersPercentageYearly",

data = yearlyactiveUsers,palette="coolwarm").set_title("Active users yearly on Stack Overflow who posted 5 or more questions/answers yearly")

sns.set(rc={'figure.figsize':(12,5)})

ax1= yearlyactiveUsers.plot(x="Years",y=["UsersRunningTotalYearly","ActiveUsersYearly"], 

kind="bar",figsize=(12,5),title='Total number of Register Users Vs Active Users yearly on SO')

#ax1.set(ylim=(0, 9806941))

yearlyactiveUsers.head(11)
#yearly completely un-answered and non-accepted answers questions (Merged) on stack overflow

unansweredQ = """select EXTRACT(YEAR FROM creation_date) Years, count(*) UnAnsweredYearly,

ROUND(100*count(*)/16867398,1) as percentage from `bigquery-public-data.stackoverflow.posts_questions` 

where id not in (SELECT distinct id  FROM `bigquery-public-data.stackoverflow.posts_questions`

where post_type_id = 1 and Accepted_Answer_Id is not null and EXTRACT(YEAR FROM creation_date) <= 2018 ) 

and post_type_id = 1  and EXTRACT(YEAR FROM creation_date) <= 2018

group by EXTRACT(YEAR FROM creation_date)"""



yearlyunansweredQ = stackoverflow.query_to_pandas(unansweredQ)

dfUsers  = pd.DataFrame(yearlyunansweredQ)

#dfUsers

#dfUsers.plot.bar()

ax = sns.barplot(x = "Years",y = "percentage",

data = yearlyunansweredQ,palette="coolwarm").set_title("Yearly completely un-answered and non-accepted answers questions on stack overflow in %")

sns.set(rc={'figure.figsize':(12,5)})

#total number of quesitons asked Vs remain un-answered questions throughout the years 

#from 2008 to 2018 as 2019 is not finished yet to analyse

#If a questions did not get accepted answer we should also considered it as an unasnwered questoins on Stack Overflow



questions = """SELECT

  EXTRACT(YEAR FROM creation_date) AS Year,

  COUNT(*) AS Total_Posted_Questions,

  SUM(IF(answer_count > 0 and Accepted_Answer_Id is not null,1,0)) AS Total_Answered_Questions,

  SUM(IF(answer_count <= 0 or Accepted_Answer_Id is null ,1,0)) AS Total_UnAnswered_Questions

FROM `bigquery-public-data.stackoverflow.posts_questions`

where EXTRACT(YEAR FROM creation_date) <= 2018

GROUP BY Year

ORDER BY Year"""



yearlyallquestions = stackoverflow.query_to_pandas(questions)

dfUsers  = pd.DataFrame(yearlyallquestions)

#dfUsers

#dfUsers.plot.bar()

#ax = sns.barplot(x = "Years",y = "percentage",

#data = yearlyunansweredQ,palette="coolwarm").set_title("Yearly completely un-answered and non-accepted answers questions on stack overflow in %")



yearlyallquestions.plot(x="Year",y=["Total_Posted_Questions","Total_Answered_Questions","Total_UnAnswered_Questions"], 

kind="bar",figsize=(12,5),title='Total number of posted questions Vs answered questions Vs unanswered questions yearly on SO')





dfScoreofQuestions = pd.read_csv("../input/extracted-so-dataset/Questions Scores.csv")

sns.set(style="darkgrid")

sns.lineplot(x="Years", y="Questions Rate In Percentage", hue="Questions Score",data=dfScoreofQuestions) 

# we could also use directly *** allScores variable

sns.set(rc={'figure.figsize':(12,5)})

dfScoreofQuestions.head()
dfScoreofAnswers = pd.read_csv("../input/extracted-so-dataset/AnswersScores.csv")

sns.set(style="darkgrid")

sns.lineplot(x="Years", y="Answers Rate In Percentage", hue="Answers Score",data=dfScoreofAnswers) 

# we could also use directly *** allScores variable

sns.set(rc={'figure.figsize':(12,5)})

dfScoreofAnswers.head()
dfCommentsRatioYearly = pd.read_csv("../input/extracted-so-dataset/Comments Ratio Yearly.csv")

ax = sns.barplot(x = "Years",y = "Yearly Comments Ratio in Percentage",

data = dfCommentsRatioYearly,palette="coolwarm").set_title("Yearly comments ratio of questions and answers")

sns.set(rc={'figure.figsize':(12,5)})

dfCommentsRatioYearly.head()
import matplotlib.pyplot as plt



dfTopTagsOfQuestions = pd.read_csv("../input/extracted-so-dataset/Yearly discussion of top 20 tags questions.csv")

sns.set(style="darkgrid")

sns.lineplot(x="Years", y="Yearly percentage of Questions", hue="TagName",data=dfTopTagsOfQuestions

            )

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.004, 1), loc=2, borderaxespad=0.)

sns.set(rc={'figure.figsize':(8,6)})

dfTopTagsOfQuestions.head()
dfTopTagsOfAnswers = pd.read_csv("../input/extracted-so-dataset/Yearly discussion of top 20 tags answers.csv")

sns.set(style="darkgrid")

sns.lineplot(x="Years", y="Yearly Percentage of Answers", hue="TagName",data=dfTopTagsOfAnswers

            )

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.004, 1), loc=2, borderaxespad=0.)

sns.set(rc={'figure.figsize':(8,6)})

dfTopTagsOfAnswers.head()
dfReviewYearly = pd.read_csv("../input/extracted-so-dataset/Review Yearly.csv")

ax = sns.barplot(x = "Years",y = "Yearly Percentage of Review",

data = dfReviewYearly,palette="coolwarm").set_title("Yearly review of questions and answers")

sns.set(rc={'figure.figsize':(12,5)})

dfReviewYearly.head()