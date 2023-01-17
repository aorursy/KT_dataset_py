# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # graphing capabilities

from beautifultext import BeautifulText as bt # utility script

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import seaborn as sns# for data viz.

import plotly.express as px

import plotly.graph_objects as go





pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
g1=bt(font_family='Comic Sans MS',color='Dark Black',font_size=19)

g1.printbeautiful('Reading Files')
multiple_choice_responses = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv",low_memory=False)# this warning shows when pandas finds 

# difficult to guess datatype for each column in large dataset

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv",low_memory=False)

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv",low_memory=False)

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv",low_memory=False)
g1=bt(font_family='Comic Sans MS',color='bLUE',font_size=19)

g1.printbeautiful('MULTIPLE CHOICE QUESTION OVERVIEW')
multiple_choice_responses.head(10)
g1=bt(font_family='Comic Sans MS',color='bLUE',font_size=19)

g1.printbeautiful('MUTIPLE CHOICE RESPONSES MISSING VALUES')
total = multiple_choice_responses.isnull().sum().sort_values(ascending=False)

percent_1 = multiple_choice_responses.isnull().sum()/multiple_choice_responses.isnull().count()*100

percent_1 = (round(percent_1, 1)).sort_values(ascending=False)

missing_multiple_choice_responses = pd.concat([total, percent_1], axis=1, keys=["Total", "%"], sort=False)



g1=bt(font_family='Comic Sans MS',color='bLUE',font_size=19)

g1.printbeautiful('PERCENTAGE OF MISSING VALUES')

missing_multiple_choice_responses.head(10)
g1=bt(font_family='Comic Sans MS',color='GREEN',font_size=19)

g1.printbeautiful('AGE OVERVIEW')
age_groups=multiple_choice_responses.groupby('Q1').count().Q2

age_groups.drop(age_groups.tail(1).index,inplace=True)



plt.figure(figsize=(15,7))



sns.barplot(x=age_groups.index, y=age_groups.values,palette='winter')



plt.xlabel('AGE GROUPS',fontsize=20)



plt.ylabel('TOTAL NUMBER OF PEOPLE IN AN AGE GROUP',fontsize=15)
g1=bt(font_family='Comic Sans MS',color='bLUE',font_size=19)

g1.printbeautiful('''This shows 25-29 is age group where People are incilned towards Data science 

                  And also upto the age of 29 this Number of Participants were increasing then it starts to decrease''')
gender_dist=multiple_choice_responses.Q2.iloc[1:].value_counts()

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=gender_dist.index[:2], values=gender_dist.values[:2], hole=.5)])

fig.show()
country=multiple_choice_responses.Q3.value_counts()





fig = go.Figure(go.Treemap(

    labels = country.index,

    parents=['World']*len(country),

    values = country

))



fig.update_layout(title = 'Country of Survey Participants')

fig.show()



## credits https://www.kaggle.com/subinium/the-hitchhiker-s-guide-to-the-kaggle thanks for this wonderful plot type