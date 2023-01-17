# Load some modules...

import pandas as pd

import numpy as np

import calendar 

import time 
# Load the data in dataframes...

df_professionals = pd.read_csv("../input/professionals.csv")

df_answers = pd.read_csv("../input/answers.csv")

df_questions = pd.read_csv("../input/questions.csv")
df_questions_answers = df_questions.merge(right=df_answers, how='inner', 

                                         left_on='questions_id', 

                                         right_on='answers_question_id')



df_questions_answers["question_count"] = 1

df_questions_count = df_questions_answers.groupby("answers_author_id").sum()

print(df_questions_count.head())
df_professionals.head()


def calc_days_since_joined(x):

    registration_timestamp = x.split(" ", 1)[0]

    registration_timestamp = calendar.timegm(time.strptime(registration_timestamp, '%Y-%m-%d'))

    timestamp = time.time()    

    

    result = int((timestamp - registration_timestamp)/60/60/24)

    

    return result





# We use copy of the professionals dataframe, so we can drop some columns without losing them

df_professionals_custom = df_professionals.copy()



# Here is the real action in....

df_professionals_custom["days_since_join"] = df_professionals["professionals_date_joined"].apply(calc_days_since_joined)



# We drop some columns, we don't need them in this dataframe

df_professionals_custom.drop(["professionals_location", "professionals_industry", "professionals_headline", "professionals_date_joined"], axis=1, inplace=True)





df_professionals_custom.head()
df_professionals_custom = df_professionals_custom.merge(right=df_questions_count, how='inner', 

                                                 left_on='professionals_id', 

                                                 right_on='answers_author_id')



print(df_professionals_custom.head())
# 

def answered_questions_quote(x):

    result = x["question_count"] / (x["days_since_join"]/10)

    return result

    

    

df_professionals_custom["answered_questions_per_10days_quote"] = df_professionals_custom.apply(answered_questions_quote, axis=1)







df_professionals_custom.head()