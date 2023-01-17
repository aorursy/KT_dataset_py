# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import init_notebook_mode, iplot

from bokeh.plotting import figure, show, output_file, output_notebook

from bokeh.palettes import Spectral11, colorblind, Inferno, BuGn, brewer

import plotly.figure_factory as ff

from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource,LinearColorMapper,BasicTicker, PrintfTickFormatter, ColorBar

output_notebook()
## Load dataset
answers=pd.read_csv('../input/answers.csv')

comments = pd.read_csv('../input/comments.csv',parse_dates = ['comments_date_added'])

emails = pd.read_csv("../input/emails.csv",parse_dates = ['emails_date_sent'])

group_memberships = pd.read_csv('../input/group_memberships.csv')

groups = pd.read_csv('../input/groups.csv')

matches = pd.read_csv('../input/matches.csv')

professionals = pd.read_csv("../input/professionals.csv" ,parse_dates = ['professionals_date_joined'])

questions = pd.read_csv('../input/questions.csv')

school_memberships = pd.read_csv('../input/school_memberships.csv')

students = pd.read_csv('../input/students.csv',parse_dates = ['students_date_joined'])

tag_questions = pd.read_csv("../input/tag_questions.csv")

tag_users = pd.read_csv('../input/tag_users.csv')

tags = pd.read_csv('../input/tags.csv')
answers.head(2)
comments.head(2)
emails.head(2)
group_memberships.head(2)
questions.head(2)
school_memberships.head(2)
students.head(2)
tag_questions.head(2)
tag_users.head(2)
tags.head(2)
print('First Student joined at ',students.students_date_joined.min())

print('Latest Student joined from this Dataset',students.students_date_joined.max())
def extract_date(df, column):

    

    df['year'] = df[column].apply(lambda x: x.year)

    df['month'] = df[column].apply(lambda x: x.month)

    df['day'] = df[column].apply(lambda x: x.day)
extract_date(professionals, 'professionals_date_joined')
professionals.head(3)
year  = professionals.year.unique()

type(year)
rev = -np.sort(-year)

rev
TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'

p = figure(title="Year-wise total number of professionals Count", y_axis_type="linear", plot_height = 400,

           tools = TOOLS, plot_width = 800)

p.xaxis.axis_label = 'Year'

p.yaxis.axis_label = 'Total Count'

p.line(rev, professionals.year.value_counts() ,line_color="purple", line_width = 3)

p.select_one(HoverTool).tooltips = [

    ('year', '@x'),

    ('Number of Count', '@y'),

]



output_file("line_chart.html", title="Line Chart")

show(p)
professionals = professionals.dropna()
top_location = professionals.professionals_location.value_counts().head(12)
top_location.plot.barh(figsize=(20,30),legend=True,fontsize='16', color=['#b2ade6'])

plt.title('Top Locations of professionals\n', fontsize='16')

plt.ylabel('location', fontsize='30')

plt.xlabel('Records', fontsize='30')
industries = professionals['professionals_industry'].value_counts().head(12)
industries.plot.barh(figsize=(20,30),legend=True,fontsize='16', color=['#b2ade6'])

plt.title('Top Industries\n', fontsize='16')

plt.ylabel('Industries', fontsize='30')

plt.xlabel('Records', fontsize='30')
headlines = professionals['professionals_headline'].value_counts().head(12)
headlines.plot.barh(figsize=(20,30),legend=True,fontsize='16', color=['#b2ade6'])

plt.title('Top Headlines\n', fontsize='16')

plt.ylabel('Headlines', fontsize='30')

plt.xlabel('Records', fontsize='30')
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

words = ' '.join(professionals['professionals_headline'])

wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,

                      background_color='white',min_font_size=6,

                      width=2000,collocations=False,

                      height=1500

                     ).generate(words)

plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
extract_date(emails, 'emails_date_sent')
fig = plt.figure(figsize=(12, 7))

sns.set_style("whitegrid", {'axes.grid' : False})

sns.countplot(emails['emails_frequency_level'],palette="Set3")

plt.xlabel('Email Level')

plt.ylabel('Count')
emailyear  = emails.year.unique()

emailrev = -np.sort(-emailyear)

emailrev
TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'

p = figure(title="Year-wise total number of Email Count", y_axis_type="linear", plot_height = 400,

           tools = TOOLS, plot_width = 800)

p.xaxis.axis_label = 'Year'

p.yaxis.axis_label = 'Total Count'

p.line(emailrev, emails.year.value_counts() ,line_color="purple", line_width = 3)

p.select_one(HoverTool).tooltips = [

    ('year', '@x'),

    ('Number of Email Count', '@y'),

]



output_file("line_chart.html", title="Line Chart")

show(p)
extract_date(students, 'students_date_joined')  


students.isnull().sum()

students  = students.dropna()

students.isnull().sum()
## Top Student location
top_Student_location = students.students_location.value_counts().head(12)

top_Student_location.plot.barh(figsize=(20,30),legend=True,fontsize='16', color=['#b2ade6'])

plt.title('Top Locations of Students\n', fontsize='16')

plt.ylabel('location', fontsize='30')

plt.xlabel('Records', fontsize='30')
studentyear  =  students.year.unique()

studentrev = -np.sort(-studentyear)

studentrev
TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'

p = figure(title="Year-wise total number of Student Count", y_axis_type="linear", plot_height = 400,

           tools = TOOLS, plot_width = 800)

p.xaxis.axis_label = 'Year'

p.yaxis.axis_label = 'Total Count'

p.line(studentrev, students.year.value_counts() ,line_color="purple", line_width = 3)

p.select_one(HoverTool).tooltips = [

    ('year', '@x'),

    ('Number of Student Count', '@y'),

]



output_file("line_chart.html", title="Line Chart")

show(p)
top_tag = tags.tags_tag_name.value_counts().head(12)

top_tag.plot.barh(figsize=(20,30),legend=True,fontsize='16', color=['#b2ade6'])

plt.title('Top Tags\n', fontsize='16')

plt.ylabel('Tags', fontsize='30')
values = ','.join(str(v) for v in tags['tags_tag_name'])

wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,

                      background_color='white',min_font_size=6,

                      width=2000,collocations=False,

                      height=1500

                     ).generate(values)

plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
extract_date(comments, 'comments_date_added')
commentyear  =  comments.year.unique()

commentrev = -np.sort(-commentyear)

commentrev



TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'

p = figure(title="Year-wise total number of Comment Count", y_axis_type="linear", plot_height = 400,

           tools = TOOLS, plot_width = 800)

p.xaxis.axis_label = 'Year'

p.yaxis.axis_label = 'Total Count'

p.line(commentrev, comments.year.value_counts() ,line_color="purple", line_width = 3)

p.select_one(HoverTool).tooltips = [

    ('year', '@x'),

    ('Number of Comment Count', '@y'),

]



output_file("line_chart.html", title="Line Chart")

show(p)
values = ','.join(str(v) for v in comments['comments_body'])

wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,

                      background_color='white',min_font_size=6,

                      width=2000,collocations=False,

                      height=1500

                     ).generate(values)

plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
questions.head()
from sklearn.model_selection import train_test_split

X = emails[['emails_id']]

y = emails.year

emails.columns
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#from sklearn.svm import SVC

#svclassifier = SVC(kernel='linear')

#svclassifier.fit(X_train, y_train)



from sklearn.svm import SVC # "Support vector classifier"

model = SVC(kernel='linear', C=1E10)

model.fit(X, y)
y_pred = model.predict(X)
y_pred
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y, y_pred))

print(classification_report(y, y_pred))