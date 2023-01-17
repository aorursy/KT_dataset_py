# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import numpy as np 

import pandas as pd

pd.set_option('display.max_columns', 5000)

pd.set_option('max_colwidth', -1)

import seaborn as sns

import cufflinks as cf

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from IPython.display import Markdown, display



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib inline



import warnings 

warnings.filterwarnings("ignore")

def read_csv(file_name):

    df = pd.read_csv(file_name)

    return df



mcq_responses_df = read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

other_txt_df = read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

questions_df = read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

survey_schema_df = read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')
mcq_responses_df.drop(mcq_responses_df.index[0], inplace=True)
mcq_responses_df.head()
questions_df
display(Markdown('**{}**'.format(questions_df['Q3'][0])))

mcq_responses_df['Q3'].value_counts()[0:30].plot.barh(figsize=(5, 8))

plt.title('Top Countries with more Respondents')

plt.show()
india_df = mcq_responses_df[mcq_responses_df['Q3']=='India']

usa_df = mcq_responses_df[mcq_responses_df['Q3']=='United States of America']
def plot_grouped_graph(col_name, india_df=india_df, usa_df=usa_df):

    display(Markdown('**{}**'.format(questions_df[col_name][0])))

    

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))

    sns.countplot(y=col_name, data=india_df, ax=ax[0], order = india_df[col_name].value_counts().index)

    sns.countplot(y=col_name, data=usa_df, ax=ax[1], order = usa_df[col_name].value_counts().index)

    

    ax[0].tick_params(labelsize=10)

    ax[0].set_ylabel('')

    ax[0].set_xlabel('')

    ax[1].set_ylabel('')

    ax[1].set_xlabel('')

    ax[1].tick_params(labelsize=10)

    

    ax[0].set_title('India', fontsize=20)

    ax[1].set_title('USA', fontsize=20)

    

    plt.show()

    return None
plot_grouped_graph('Q1')
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 10))





sns.countplot(x='Q1', hue='Q2', data=india_df, 

              order = india_df['Q1'].value_counts().sort_index().index, 

              ax=ax[0])



ax[0].set_title('Age & Gender Distribution of India', size=15)





sns.countplot(x='Q1', hue='Q2', data=usa_df, 

              order = usa_df['Q1'].value_counts().sort_index().index, 

              ax=ax[1])



ax[1].set_title('Age & Gender Distribution of USA', size=15)

ax[0].set_ylabel('')

ax[0].set_xlabel('')

ax[1].set_ylabel('')

ax[1].set_xlabel('')

plt.show()

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 12))





sns.countplot(x='Q4', hue='Q2', data=india_df, 

              order = india_df['Q4'].value_counts().sort_index().index, 

              ax=ax[0])



ax[0].set_title('India', size=15)





sns.countplot(x='Q4', hue='Q2', data=usa_df, 

              order = usa_df['Q4'].value_counts().sort_index().index, 

              ax=ax[1])



ax[1].set_title('USA', size=15)

ax[0].set_ylabel('')

ax[0].legend(loc=1)

ax[0].set_xlabel('')

ax[1].set_ylabel('')

ax[1].set_xlabel('')

ax[1].legend(loc=1)

plt.xticks(rotation=45, size=12)

plt.show()
plot_grouped_graph('Q23')
plot_grouped_graph('Q10')
def multiple_responses_question_to_df(col_name, df):

    option_df = df.loc[:, india_df.columns.str.startswith(col_name)]



    temp_df = {}

    for col in option_df.columns:

        frame = option_df[col].value_counts().to_frame()

        name = frame.index.tolist()[0]



        if isinstance(name, int):

            continue

        else:

            temp_df[name.split('(')[0]] = frame[col][name]

    return pd.DataFrame(temp_df, index=[0]).transpose()

 







def plot_mcq_df(india_mcq_df, usa_mcq_df):

    

    fig, ax = plt.subplots(1, 2, figsize=(30, 12))

    india_mcq_df.sort_values(by=0).plot.barh(legend=False, ax=ax[0])

    usa_mcq_df.sort_values(by=0).plot.barh(legend=False,ax=ax[1])

    

    ax[0].tick_params(labelsize=14)

    ax[0].set_ylabel('')

    ax[0].set_xlabel('')

    ax[1].set_ylabel('')

    ax[1].set_xlabel('')

    ax[1].tick_params(labelsize=14)

    

    ax[0].set_title('India', fontsize=20)

    ax[1].set_title('USA', fontsize=20)

    

    plt.show()

    return None
india_mcq_df = multiple_responses_question_to_df('Q16', india_df)

usa_mcq_df = multiple_responses_question_to_df('Q16', usa_df)



plot_mcq_df(india_mcq_df, usa_mcq_df)
india_mcq_df = multiple_responses_question_to_df('Q18', india_df)

usa_mcq_df = multiple_responses_question_to_df('Q18', usa_df)



plot_mcq_df(india_mcq_df, usa_mcq_df)
india_mcq_df = multiple_responses_question_to_df('Q13', india_df)

usa_mcq_df = multiple_responses_question_to_df('Q13', usa_df)



plot_mcq_df(india_mcq_df, usa_mcq_df)
india_mcq_df = multiple_responses_question_to_df('Q12', india_df)

usa_mcq_df = multiple_responses_question_to_df('Q12', usa_df)



plot_mcq_df(india_mcq_df, usa_mcq_df)
india_mcq_df = multiple_responses_question_to_df('Q21', india_df)

usa_mcq_df = multiple_responses_question_to_df('Q21', usa_df)



plot_mcq_df(india_mcq_df, usa_mcq_df)
plot_grouped_graph('Q22')