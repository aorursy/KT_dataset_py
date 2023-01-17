from IPython.display import Image

import os

Image("../input/photos/data_plot.jpg")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="ticks", color_codes=True)

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)



from IPython import display

%matplotlib inline



%config IPCompleter.greedy=True



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dirname = '/kaggle/input/kaggle-survey-2019'

dirname
#Read the csv files with Pandas



questions = pd.read_csv(dirname + '/questions_only.csv')

survey_res = pd.read_csv(dirname + '/multiple_choice_responses.csv')

survey_schema = pd.read_csv(dirname + '/survey_schema.csv')

text_res = pd.read_csv(dirname + '/other_text_responses.csv')

print(f'Reading the shape of the data')

print(f"Questions: {questions.shape}")

print(f"Survey Responses: {survey_res.shape}")

print(f"Survey Schema: {survey_schema.shape}")

print(f"Text responses: {text_res.shape}")
questions.head(10)
survey_res.head()
survey_res.describe()
Image("../input/photos/canada.jpg")
survey_res.rename(columns={'Q1' : 'Age',

                           'Q2' : 'Gender',

                           'Q3' : 'Country',

                           'Q4' : 'Education Level',

                           'Q5' : 'Job Title',

                           'Q6' : 'Company Size',

                           'Q10' : 'Salary'},

                  inplace=True)

print(survey_res.columns)
# Keep relevant columns

col_to_keep = ['Age',

               'Gender',

               'Country',

               'Education Level',

               'Job Title',

               'Company Size',

               'Salary',]

col_to_keep

#use only the kept columns as the new survey result data

survey_res = survey_res[col_to_keep]

survey_res.head()
Canada = survey_res[survey_res['Country'].str.contains('Canada')]

print('{} people from Canada participated in the survey'.format(Canada.shape[0]))
Canada.describe()
#Check current data type:

Canada.dtypes
Canada.isnull().sum()
#Comment out the code in order not to refill multiple times



# Canada['Salary'] = Canada['Salary'].fillna(method='ffill', limit=1, inplace=True).fillna(method='bfill', limit=1, inplace=True)

# Canada
Canada.describe()
age = Canada.Age.value_counts()

age
percent_age = (age/(Canada.shape[0]))*100

percent_age
import seaborn as sns





stan_color = sns.color_palette()[0]

sal_order = age.index

sns.countplot(data=Canada, y="Age", color=stan_color, order=sal_order)



gender = Canada.Gender.value_counts()

gender
Canada_wom = Canada[Canada['Gender']=='Female']

Canada_wom.describe()
percent_women = (Canada_wom.count()/Canada.shape[0])*100

percent_women

Canada_men = Canada[Canada['Gender']=='Male']

Canada_men.describe()
Canada_Ed = Canada['Education Level'].value_counts(sort=True)

Canada_Ed
labels_edu = Canada_Ed.index

values_edu = Canada_Ed.values



edu_colors= ['#A7FFEB','#43A047', '#1B5E20', '#76FF03', '#C6FF00', '#DCEDC8', '#E8F5E9'] 



pie = go.Pie(labels=labels_edu, values=values_edu, marker=dict(colors=edu_colors,line=dict(color='#000000', width=1)))

layout = go.Layout(title='Education Level')



fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
Canada_Salary = Canada.Salary.value_counts()

Canada_Salary
import seaborn as sns



stan_color = sns.color_palette()[0]

#Order the plot by count values

sal_order = Canada_Salary.index

sns.countplot(data=Canada, y='Salary', color=stan_color, order=sal_order)
Canada_Job = Canada['Job Title'].value_counts()

Canada_Job
def compute_percentage(df,col):

    """

    The compute_percentage object computs the percentage of the value counts for the column.

    

    Args:

        df (dataframe): The dataset for the analysis

        col: The specific column (feature) of interest within the dataframe

        

    Returns:

            percentage of the frequency of the feature

    """

    return df[col].value_counts(normalize=True) * 100



def bi_variant_chart(col1,col2,x_title,y_title):

    """

    Args:

        col1 (str): col1 is the first feature for plotting the bar chart

        col2 (str): col2 is the second feature for plotting the bar chart

        x_title (str): Title for the x-axis

        y_title: Title for the y-axis

    

    """

    

    index = Canada[col1].dropna().unique()

    vals = Canada[col2].unique()

    layout = go.Layout()

    trace = []

    for j,y_axis in enumerate(vals):

        trace.append(go.Bar(x = Canada[Canada[col2] == y_axis][col1].value_counts().index,

                            y = Canada[Canada[col2] == y_axis][col1].sort_values().value_counts().values,

                opacity = 0.6, name = vals[j]))

    fig = go.Figure(data = trace, layout = layout)

    fig.update_layout(

        title = x_title,

        yaxis = dict(title = y_title),

        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),

        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")

    fig.show()
bi_variant_chart("Salary","Education Level","Salary VS Education Level","Count")
bi_variant_chart("Salary","Job Title","Salary VS Job Title","Count")
bi_variant_chart("Company Size","Salary","Company Size VS Salary","Count")
bi_variant_chart("Gender","Salary","Gender VS Salary","Count")