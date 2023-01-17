# Project Origin: https://www.kaggle.com/c/nips-2015-papers
# Neural Information Processing Systems (NIPS)
# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')

# identify official names/casing of EventType Field
paper_types = pd.read_sql_query("""
SELECT distinct EventType
FROM Papers
""", con)
print(paper_types)
# Poster, Oral, Spotlight

# Words on Diverse Topics
gender_words = ['gender','sex','male','female','man','woman','girl','boy','MTF','FTM','intersex','transgender','transsexual','genderfluid','genderfucker','non-binary']
so_words = ['sexual orientation','sexual preference','gay','lesbian','bisexual','GLB','LGB','pansexual']
poc_words = ['people of color','caucasian','black','African-American','Asian American','AAPI','API','person of color','European American','Latino','Latina','Native American','Indigenous']
religious_words = ['Islam','Muslim','Judiasm','Jewish','Jew','Catholic','Protestant','Presbyterian','Christian','Buddhist','Buddhism']
diverse_words = ['queer','LGBT','LGBTQ']

def lowercase_the_words(array_name):
  lowered_array = []
  for a in array_name:
      lowered_array.append(a.lower)
  return lowered_array

def run_the_words(array_name):
    words = lowercase_the_words(array_name)
    print(words)
    word_result = ["Word","Paper Type", "Used In Count"]
    for word in words:
        word_search_query = """
            SELECT EventType, Count(Distinct ID) as count
            FROM Papers
            WHERE PaperText like '%""" + str(word) + """%'
            GROUP BY 1
            """
        word_info = pd.read_sql_query(word_search_query, con)
        #print(word_info['EventType'],word_info['count'])
        #if word_info:
        word_result.append(word_info)
    print(word_result)

run_the_words(diverse_words)

