from IPython.display import Image

Image("../input/careervillage-image/CareerVillage_logo_wide_FullColor_785x100.png")
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
answers.head()
answers.describe()
words = answers['answers_body'][~pd.isnull(answers["answers_body"])]

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(words))

plt.figure(figsize=(16,12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Answers")

plt.axis("off")
comments.head()
comments.describe()
words = comments['comments_body'][~pd.isnull(comments["comments_body"])]

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(words))

plt.figure(figsize=(16,12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Comments")

plt.axis("off")
emails.head()
emails.describe()
emails.info()
plt.figure(figsize=(10,6))

sns.barplot(emails['emails_frequency_level'].values,emails['emails_frequency_level'].index)

plt.xlabel("emails_frequency_level", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.show()
group_memberships.head()
group_memberships.describe()
groups.head()
groups.describe()
sorted_groups = groups['groups_group_type'].value_counts()

plt.figure(figsize=(12,8))

sns.barplot(sorted_groups.values,sorted_groups.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Group Type", fontsize=15)

plt.show()
matches.head()
matches.describe()
professionals.head()
professionals.describe()
professionals.isnull().sum()
locations = professionals['professionals_location'].value_counts().head(10)

plt.figure(figsize=(12,8))

sns.barplot(locations.values, locations.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Location", fontsize=15)

plt.show()
industries = professionals['professionals_industry'].value_counts().head(10)

plt.figure(figsize=(12,8))

sns.barplot(industries.values, industries.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Industry", fontsize=15)

plt.show()
headlines = professionals['professionals_headline'].value_counts().head(10)

plt.figure(figsize=(12,8))

sns.barplot(headlines.values, headlines.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Headlines", fontsize=15)

plt.show()
words = professionals['professionals_headline'][~pd.isnull(professionals['professionals_headline'])]

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(words))

plt.figure(figsize=(16,12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Headlines")

plt.axis("off")
questions.head()
questions.describe()
words = questions['questions_title'][~pd.isnull(questions['questions_title'])]

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(words))

plt.figure(figsize=(16,12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Questions")

plt.axis("off")
words = questions['questions_body'][~pd.isnull(questions['questions_body'])]

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(words))

plt.figure(figsize=(16,12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Questions body")

plt.axis("off")
school_memberships.head()
school_memberships.describe()
students.head()
students.describe()
students.isnull().sum()
locations = students['students_location'].value_counts().head(10)

plt.figure(figsize=(12,8))

sns.barplot(locations.values, locations.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Location", fontsize=15)

plt.show()
tag_questions.head()
tag_questions.describe()
tag_users.head()
tag_users.describe()
tags.head()
tags.describe()
tags_name = tags['tags_tag_name'].value_counts().head(10)

plt.figure(figsize=(12,8))

sns.barplot(tags_name.values, tags_name.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Tags", fontsize=15)

plt.show()