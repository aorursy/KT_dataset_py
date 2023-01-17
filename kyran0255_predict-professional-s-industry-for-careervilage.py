import pandas as pd

import numpy as np

import os

from os import chdir

import sys

import re 



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns 
answers = pd.read_csv('../input/answers.csv')

comments = pd.read_csv('../input/comments.csv')

emails = pd.read_csv("../input/emails.csv")

group_memberships = pd.read_csv('../input/group_memberships.csv')

groups = pd.read_csv('../input/groups.csv')

matches = pd.read_csv('../input/matches.csv')

professionals = pd.read_csv("../input/professionals.csv")

questions = pd.read_csv('../input/questions.csv')

school_memberships = pd.read_csv('../input/school_memberships.csv')

students = pd.read_csv('../input/students.csv')

tag_questions = pd.read_csv("../input/tag_questions.csv")

tag_users = pd.read_csv('../input/tag_users.csv')

tags = pd.read_csv('../input/tags.csv')
answers.head()
answers.answers_body[1] 
answers.describe()
comments.head()
comments.comments_body[1]
comments.describe()
emails.head()
emails.describe() 
group_memberships.head()
group_memberships.describe()
groups.head()
groups.groups_group_type.unique() 
groups.describe()
sorted_groups = groups['groups_group_type'].value_counts()

plt.figure(figsize=(20,10))

sns.barplot(sorted_groups.values,sorted_groups.index)

plt.xlabel("Count", fontsize=20)

plt.ylabel("Group Type", fontsize=20)

plt.show()
matches.head()
matches.describe()
professionals.head()
print('location:', professionals.professionals_location.unique())



print('Industry:', professionals.professionals_industry.unique())
professionals.describe()
professionals_locations = professionals['professionals_location'].value_counts().head(30)

plt.figure(figsize=(20,10))

sns.barplot(professionals_locations.values, professionals_locations.index)

plt.xlabel("Count", fontsize=20)

plt.ylabel("Location", fontsize=20)

plt.show()
professionals_industries = professionals['professionals_industry'].value_counts().head(30)

plt.figure(figsize=(20,10))

sns.barplot(professionals_industries.values, professionals_industries.index)

plt.xlabel("Count", fontsize=20)

plt.ylabel("Industry", fontsize=20)

plt.show()
professionals_headlines = professionals['professionals_headline'].value_counts().head(30)

plt.figure(figsize=(20,10))

sns.barplot(professionals_headlines.values, professionals_headlines.index)

plt.xlabel("Count", fontsize=20)

plt.ylabel("Headlines", fontsize=20)

plt.show()
questions.head()
questions.describe()
school_memberships.head()
school_memberships.describe()
students.head()
students.describe()
students_locations = students['students_location'].value_counts().head(30)

plt.figure(figsize=(20,10))

sns.barplot(students_locations.values, students_locations.index)

plt.xlabel("Count", fontsize=20)

plt.ylabel("Location", fontsize=20)

plt.show()
tag_questions.head()
tag_questions.describe()
tag_users.head()
tag_users.describe()
tags.head()
tags.describe()
tag_n_user = pd.merge(tags, tag_users, left_on='tags_tag_id', right_on='tag_users_tag_id', how='outer')

tag_n_user.sort_values('tag_users_tag_id')[:10]
tag_n_user.count()
tag_n_questions = pd.merge(tags, tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id', how='outer')

tag_n_questions.sort_values('tags_tag_id')[:10]
tag_n_questions.count()
tag_n_questions[['tags_tag_name','tag_questions_question_id']]
tags_top20 = tag_n_questions['tags_tag_name'].value_counts().head(20)

plt.figure(figsize=(20,10))

sns.barplot(tags_top20.values, tags_top20.index)

plt.xlabel("Count", fontsize=20)

plt.ylabel("Tags_top20", fontsize=20)

plt.show()
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from nltk.corpus import words



vectorizer = CountVectorizer(analyzer = 'word', 

                             lowercase = True,

                             tokenizer = None,

                             preprocessor = None,

                             stop_words = 'english',

                             min_df = 2, # 토큰이 나타날 최소 문서 개수로 오타나 자주 나오지 않는 특수한 전문용어 제거에 좋다. 

                             ngram_range=(1, 3),

                             vocabulary = set(words.words()), # nltk의 words를 사용하거나 문서 자체의 사전을 만들거나 선택한다. 

                             max_features = 90000

                            )

vectorizer
pipeline = Pipeline([

    ('vect', vectorizer),

    ('tfidf', TfidfTransformer(smooth_idf = False)),

])  

pipeline
# See answers

%time answer_train_tfidf_vector = pipeline.fit_transform(answers['answers_body'].values.astype('U'))  
vocab = vectorizer.get_feature_names()

print(len(vocab))

vocab[:10]
import numpy as np

dist = np.sum(answer_train_tfidf_vector, axis=0)



for tag, count in zip(vocab, dist):

    print(count, tag)



pd.DataFrame(dist, columns=vocab)
# questions.questions_id = answers.answers_question_id



Q_n_A = pd.merge(questions[['questions_id','questions_title','questions_body']], answers[['answers_question_id','answers_id','answers_body']], left_on='questions_id', right_on='answers_question_id', how='outer')

Q_n_A.sort_values('questions_id')



# inner(Except NaN) : questions_id = (51123) 23931, answers_id = 51123

# outer(Include NaN) : questions_id = (51944) 23931, answers_id = 51123
Q_n_A.describe()
# professionals.professionals_id = answers.answers_author_id



Professionals_ID = pd.merge(professionals[['professionals_id','professionals_industry','professionals_headline']], answers[['answers_author_id','answers_id','answers_body']], left_on='professionals_id', right_on='answers_author_id', how='inner')

Professionals_ID.sort_values('professionals_id')



Professionals_ID



# inner(Except NaN) : professionals_id = (50106) 10067, answers_author_id = (50106) 10067

# outer(Include NaN) : professionals_id = (68191) 28152, answers_author_id = (51123) 10169



### No answers professionals delete SO inner join is our choice
Professionals_ID.describe()
Professionals_ID.answers_body[:]
Answers_Text = re.sub('<.+?>', '', Professionals_ID.answers_body[0], 0, re.I|re.S)

Answers_Text = re.sub('\n','. ', Answers_Text)

Answers_Text
Answers_Text = []

Answers_Text = re.sub('<.+?>', '', Professionals_ID.answers_body[0], 0, re.I|re.S)

A_list = []

A_list.append(Answers_Text)

A_list

#Answers_Text.append(re.sub('<.+?>', '', Professionals_ID.answers_body[1], 0, re.I|re.S))

Professionals_ID.answers_body[0]
str(Answers_Text)
from gensim.summarization import summarize 

from gensim.summarization import keywords 



print("1. Summarizing :", '\n', summarize(str(Answers_Text)), '\n')

print("######################################################################################################", '\n')

print("2. Keywords :", '\n', keywords(str(Answers_Text)))
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

    

    if (num%500 == 0 ):

        try:

            print("iteration : " + str(num))

            print("1. Summarizing :", '\n', summarize(str(result)), '\n')

            print("######################################################################################################", '\n')

            print("2. Keywords :", '\n', keywords(str(result)))

        except ValueError:

            pass  # do nothing!

    

    keywords1 = re.split('\n', keywords(str(result)))

    

    Professionals_ID["answers_keywords"][num] = keywords1
Professionals_ID[['professionals_id','answers_body','answers_keywords']].head(10)