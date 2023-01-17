import pandas as pd

import numpy as np

import os

import sys

from gensim.summarization import summarize 

from gensim.summarization import keywords

import re
print(os.listdir("../input"))

PATH = "../input/"
answers = pd.read_csv(PATH + 'answers.csv')

comments = pd.read_csv(PATH + 'comments.csv')

emails = pd.read_csv(PATH + 'emails.csv')

group_memberships = pd.read_csv(PATH + 'group_memberships.csv')

groups = pd.read_csv(PATH + 'groups.csv')

matches = pd.read_csv(PATH + 'matches.csv')

professionals = pd.read_csv(PATH + 'professionals.csv')

questions = pd.read_csv(PATH + 'questions.csv')

school_memberships = pd.read_csv(PATH + 'school_memberships.csv')

students = pd.read_csv(PATH + 'students.csv')

tag_questions = pd.read_csv(PATH + 'tag_questions.csv')

tag_users = pd.read_csv(PATH + 'tag_users.csv')

tags = pd.read_csv(PATH + 'tags.csv')
answers.head()
answers['answers_body'].head()
Professionals_ID = pd.merge(professionals[['professionals_id','professionals_industry','professionals_headline']], answers[['answers_author_id','answers_id','answers_body']], left_on='professionals_id', right_on='answers_author_id', how='inner')

Professionals_ID.sort_values('professionals_id')



Professionals_ID.head()
len(Professionals_ID.answers_body)
Professionals_ID["answers_keywords"] = ""



Professionals_ID.head()
Professionals_ID['answers_body'].fillna("No Answer", inplace = True)

for num in range(len(Professionals_ID.answers_body)):

    Answers_Text = Professionals_ID['answers_body'][num]

    Answers_Text = re.sub('<.+?>', '', Answers_Text, 0, re.I|re.S)

    Answers_Text = re.split('\n', Answers_Text)

    for n in Answers_Text:

        if (n.startswith('http')):

            del Answers_Text[Answers_Text.index(n)]

    

    result = ""

    for n in Answers_Text:

        result += n + ". "

    

    if (num%1000 == 0 ):

        try:

            print("iteration : " + str(num))

            print("1. Summarizing :", '\n', summarize(str(result)), '\n')

            print("######################################################################################################", '\n')

            print("2. Keywords :", '\n', keywords(str(result)))

        except ValueError:

            pass  # do nothing!

    

    keywords1 = re.split('\n', keywords(str(result)))

    

    Professionals_ID["answers_keywords"][num] = keywords1
Professionals_ID[['professionals_id','answers_body','answers_keywords']].tail(10)
Professionals_ID.to_csv("Professinals_kewords", sep='\t', encoding='utf-8')