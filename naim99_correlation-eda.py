import pandas as pd 
train_df = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv', 

    low_memory=False, 

    nrows=10**7, 

    dtype={

        'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

        'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

        'prior_question_had_explanation': 'boolean'

    }

)

train_df
train_df.head()

questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')

questions
questions['correct_answer'].hist() 
questions['part'].hist() 
df1 = questions.groupby("tags",as_index=False)["question_id"].count() 

df1 = df1.sort_values(by='question_id', ascending=False)

df1.head(15)
df2 = questions.groupby("part",as_index=False)["question_id"].count() 

df2 = df2.sort_values(by='question_id', ascending=False)

df2.head(15)
df3 = questions.groupby("correct_answer",as_index=False)["tags"].count() 

df3 = df3.sort_values(by='tags', ascending=False)

df3.head(15)
df4 = questions.groupby("correct_answer",as_index=False)["question_id"].count() 

df4 = df4.sort_values(by='question_id', ascending=False)

df4.head(15)
lectures = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')

lectures
df5 = lectures.groupby("part",as_index=False)["lecture_id"].count() 

df5 = df5.sort_values(by='lecture_id', ascending=False)

df5.head(15)
df6 = lectures.groupby("tag",as_index=False)["lecture_id"].count() 

df6 = df6.sort_values(by="lecture_id", ascending=False)

df6.head(15)
train_df.describe(include="all")
for col in train_df:

    print(col,len(train_df[col].unique()))
train_df.corr().style.background_gradient(cmap='coolwarm')
df_row = pd.concat([questions, lectures])

df_row
df_row['correct_answer'].value_counts() 
df_row.corr().style.background_gradient(cmap='coolwarm')