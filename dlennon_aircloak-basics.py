!pip install psycopg2
import pandas as pds

import plotly_express as px



pds.set_option('max_rows', 12)
AIRCLOAK_PG_HOST = "covid-db.aircloak.com"

AIRCLOAK_PG_PORT = 9432

AIRCLOAK_PG_USER = "covid-19-5BCFDEEB3CDD876492CD"

AIRCLOAK_PG_PASSWORD = "RjV+coInOrmahmEUDorvLL9XPNLEDgdsU4Zl1wr3cMpt04ojx5bH/1bnFLw4/WMf/yHpSXFIKkdMiMl2D4KrGQ=="

COVID_DATASET = "cov_clear"
import psycopg2



conn = psycopg2.connect(

    user=AIRCLOAK_PG_USER, 

    host=AIRCLOAK_PG_HOST, 

    port=AIRCLOAK_PG_PORT, 

    dbname=COVID_DATASET,

    password=AIRCLOAK_PG_PASSWORD)
def query(statement):

    return pds.read_sql_query(statement, conn)
def get_tables():

    return query("SHOW TABLES")



def get_table_columns(table):

    return query(f'SHOW COLUMNS FROM {table}')
get_tables()
get_table_columns("questions")
questions_df = query("SELECT * FROM questions")

questions_df[:10]
# survey_df = query("SELECT * FROM survey")

# survey_df[:10]
get_table_columns("survey")
# A simple query to extract counts of each distinct value in the tables

def count_distinct_values(table, column, order_by='count'):

    return query(f'''

        SELECT {column}, count(*) as count

        FROM {table} 

        GROUP BY {column}

        ORDER BY {order_by} DESC''')



count_distinct_values("survey", "feeling_now")
get_table_columns("symptoms")
symptoms_count = count_distinct_values("symptoms", "symptom")



px.bar(symptoms_count, 

        x='count', 

        y='symptom', 

        orientation='h', 

        height=650)
anxiety_symptoms = query('''

        SELECT symptom, avg(how_anxious) as avg_anxiety, count(*) as num_respondents

        FROM symptoms, survey

        WHERE symptoms.uid = survey.uid

        GROUP BY symptom

        ORDER BY avg(how_anxious) ASC''')



anxiety_symptoms
px.bar(anxiety_symptoms, 

       x='num_respondents', 

       y='symptom', 

       orientation = 'h', 

       color='avg_anxiety', 

       height=650)