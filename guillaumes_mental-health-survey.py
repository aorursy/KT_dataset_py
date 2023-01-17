import pandas as pd

import sqlite3



def create_connection(db_file):

    """ create a database connection to the SQLite database

        specified by the db_file

    :param db_file: database file

    :return: Connection object or None

    """

    conn = None

    try:

        conn = sqlite3.connect(db_file)

    except Exception as e:

        print(e)



    return conn
conn = create_connection('/kaggle/input/mental-health-in-the-tech-industry/mental_health.sqlite')
sql = """

SELECT name FROM  sqlite_master 

WHERE type ='table';

"""

df = pd.read_sql(sql=sql, con=conn)

df.head()
sql = """

SELECT *

FROM Survey;

"""

survey = pd.read_sql(sql=sql, con=conn)

survey.head()
survey.shape
sql = """

SELECT *

FROM Question;

"""

question = pd.read_sql(sql=sql, con=conn)

question.head()
question.shape
sql = """

SELECT *

FROM Answer;

"""

answer = pd.read_sql(sql=sql, con=conn)

answer.head(20)
answer.shape
answer[answer.QuestionID == 1].AnswerText.astype(int).describe()
answer[answer.QuestionID == 1].AnswerText.astype(int).hist()
answer.UserID.nunique()
answer.UserID.value_counts()
df = pd.DataFrame({'UserID': answer.UserID.unique()})

df.head()
answer_age = answer[answer['QuestionID']==1]

list_age = []

for user in df.UserID:

    age = int(answer_age[answer_age['UserID']==user]['AnswerText'].values[0])

    if age > 15 and age < 90:

        list_age.append(age)

    else:

        list_age.append(None)

df['age'] = list_age

df.head()
df.age.hist()
answer[answer['QuestionID']==2].AnswerText.value_counts()
answer_gender = answer[answer['QuestionID']==2]

list_gender = []

for user in df.UserID:

    gender = answer_gender[answer_gender['UserID']==user]['AnswerText'].values[0]

    if 'female' in gender.lower():

        list_gender.append('female')

    elif 'male' in gender.lower():

        list_gender.append('male')

    else:

        list_gender.append(None)
df['gender'] = list_gender

df.head()
df.gender.hist()
answer[answer['QuestionID']==3].AnswerText.value_counts()
answer_country = answer[answer['QuestionID']==3]

list_country = []

for user in df.UserID:

    country = answer_country[answer_country['UserID']==user]['AnswerText'].values[0]

    if 'united state' in country.lower():

        list_country.append('USA')

    elif 'united kingdom' in country.lower():

        list_country.append('UK')

    else:

        list_country.append(country)

df['country'] = list_country

df.head()
df.country.value_counts().index
df[df['country'].isin(df.country.value_counts().index.tolist()[:7])].country.hist()
question