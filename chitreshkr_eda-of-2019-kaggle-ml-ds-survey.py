# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
question = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
m_responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv',low_memory=False)
other = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv',low_memory=False)
survey = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', -1)
m_responses.head()
m_responses.rename(columns =  {'Time from Start to Finish (seconds)':'Time',

                               'Q1':'Age',

                              'Q2':'Gender','Q3':'Country','Q4':'Education','Q5':'Title',

                              'Q6':'Company_size','Q7':'Team_size','Q8':'ML_incorporate',

                              'Q10':'Salary','Q11':'Money_Spent_on_Ml_and_Cloud','Q14':'Primary_tool',

                              'Q15':'Coding_experience','Q19':'First_language_recommend','Q23':'ML_exp'},inplace=True)

m_responses.head()
my_response_1 = m_responses.copy()

my_response_1 = my_response_1.drop([0])

my_response_1.head()
india = my_response_1[my_response_1['Country'] == 'India']

india.head()
m_responses['Gender'].value_counts()
#m_responses = m_responses.drop("In which country do you currently reside?", axis=0)

plt.figure(figsize=(20,12))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Country", data=my_response_1)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Age", data=my_response_1)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Education", data=my_response_1)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 85)

sns.set(style="darkgrid")

ax = sns.countplot(x="Education", data=india)
usa = my_response_1[my_response_1['Country'] == 'United States of America']

#usa.head()
plt.figure(figsize=(20,10))

plt.xticks(rotation= 45)

sns.set(style="darkgrid")

ax = sns.countplot(x="Education", data=usa)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Gender", data=my_response_1)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Age", data=usa)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Age", data=india)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Title", data=usa)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Title", data=india)
russia = my_response_1[my_response_1['Country'] == 'Russia']

russia.head()
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Title", data=russia)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Age", data=russia)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 90)

sns.set(style="darkgrid")

ax = sns.countplot(x="Education", data=russia)
plt.figure(figsize=(20,10))

plt.xticks(rotation= 45)

sns.set(style="darkgrid")

ax = sns.countplot(x="Salary", data=my_response_1)