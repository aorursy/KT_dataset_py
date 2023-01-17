import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt



%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)

from wordcloud import WordCloud,STOPWORDS

import warnings

warnings.filterwarnings('ignore')

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
answers = pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')

comments = pd.read_csv('../input/data-science-for-good-careervillage/comments.csv')

emails = pd.read_csv("../input/data-science-for-good-careervillage/emails.csv")

group_memberships = pd.read_csv('../input/data-science-for-good-careervillage/group_memberships.csv')

groups = pd.read_csv('../input/data-science-for-good-careervillage/groups.csv')

matches = pd.read_csv('../input/data-science-for-good-careervillage/matches.csv')

professionals = pd.read_csv("../input/data-science-for-good-careervillage/professionals.csv")

questions = pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')

school_memberships = pd.read_csv('../input/data-science-for-good-careervillage/school_memberships.csv')

students = pd.read_csv('../input/data-science-for-good-careervillage/students.csv')

tag_questions = pd.read_csv("../input/data-science-for-good-careervillage/tag_questions.csv")

tag_users = pd.read_csv('../input/data-science-for-good-careervillage/tag_users.csv')

tags = pd.read_csv('../input/data-science-for-good-careervillage/tags.csv')
df_list = [answers,comments,emails,group_memberships,groups,matches,professionals,questions,school_memberships,students,tag_questions,tag_users,tags]
# sanity check there is 13 csv file.

len(df_list)
for table in df_list:

    print(table.columns)
answers.head()
comments.head()
emails.head() #to find out the type of the emails_frequency_level -> varchar
group_memberships.head()
groups.head() #to find out the type of the groups_group_type -> varchar
matches.head()
professionals.dropna().head() #to find out the type of the cols and ignore the nan's -> varchar*3
questions.head() #same here
school_memberships.head()
students.dropna().head()
tag_questions.head()
tag_users.head()
tags.head()
matches.head()
from IPython.display import Image

Image(filename="../input/cv-graph-schema2/graph_schema_kaggle.png")