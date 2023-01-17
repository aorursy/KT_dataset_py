# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

from plotly import tools

from plotly.subplots import make_subplots

import plotly.express as px





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_2018 = pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')
data_2019 = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
data_2019.head()
def return_percentage(data,question_part):

    """Calculates percent of each value in a given column"""

    total = data[question_part].count()

    counts_df= data[question_part].value_counts().to_frame()

    percentage_df = (counts_df*100)/total

    return percentage_df
x = list(data_2019.columns)

list(zip(x,data_2019.iloc[0,:]))
x = list(data_2018.columns)

list(zip(x,data_2018.iloc[0,:]))
# list(data_2019.columns)
data_less = data_2019[['Q1', 'Q2', 'Q3','Q4', 'Q6', 'Q9_Part_1', 'Q9_Part_2', 'Q9_Part_3', 'Q9_Part_4', 'Q9_Part_5', 

                 'Q9_Part_6', 'Q10', 'Q11', 'Q15']]

data_most_ds = data_2019[['Q12_Part_1', 'Q12_Part_2', 'Q12_Part_3', 'Q12_Part_4', 'Q12_Part_5', 'Q12_Part_6', 'Q12_Part_7',

                        'Q12_Part_8', 'Q12_Part_9', 'Q12_Part_10', 'Q12_Part_11', 'Q12_Part_12']]
data_less.head()
import matplotlib.pyplot as plt

import seaborn as sns
data_2018.head()
list(data_2018.columns)
data_less_2018 = data_2018[['Q1', 'Q3', 'Q2','Q11_Part_1','Q11_Part_2',

                            'Q11_Part_3','Q11_Part_4','Q11_Part_5','Q11_Part_6','Q11_Part_7', 'Q24']]

data_less_2018
chart = sns.countplot(data_less['Q2'])

# plt.xticks(rotate = 90)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
chart = sns.countplot(data_less_2018['Q1'])

# plt.xticks(rotate = 90)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
df_gender_2019 = return_percentage(data_less, 'Q2').reset_index()

df_gender_2019.columns = ['Gender', 'Percentage']

df_gender_2019

# (data_less[data_less['Q2'] == 'Female'].count() / data_less['Q2'].count())*100

# (data_less[data_less['Q2'] == 'Male'].count() / data_less['Q2'].count())*100
df_gender_2018= return_percentage(data_less_2018, 'Q1').reset_index()

df_gender_2018.columns = ['Gender', 'Percentage']

df_gender_2018
#Make pie chart for the male and female also

# fig = make_subplots(rows = 1, cols = 2)

# trace0 = px.pie(df_gender_2019, values='Percentage', names='Gender')

# trace1 = px.pie(df_gender_2018, values='Percentage', names='Gender')

# fig.add_trace(trace0, row = 1, col = 1)

# fig.add_trace(trace1, row = 1, col = 2)

# fig.update_layout(height=600, width=800, title_text="Comparision of Respondents by Country in 2018 and 2019", title_x = 0.5)

# fig.show()
# colors2 = ['dodgerblue', 'plum', '#F0A30A','#8c564b'] 

# gender_count_2019 = data_less['Q2'].value_counts(sort=True)

# gender_count_2018 = data_less_2018['Q1'].value_counts(sort=True)

# fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

# fig.add_trace(go.Pie(labels=list(data_less['Q2'].unique()), values=gender_count_2019.values, name="2019",marker=dict(colors=colors2)), 1,1)

# fig.add_trace(go.Pie(labels=list(data_less_2018['Q1'].unique()), values=gender_count_2018.values, name="2018",marker=dict(colors=colors2)), 1,2)

# fig.show()

data_less_2018.head()
data_less.head()
#Working on making Bar charts for 2018 and 2019 and compare as to where the respondents are mostly from 

# return_percentage(data_less, 'Q3').head(10)

df_countries_2018 = return_percentage(data_less_2018, 'Q3').head(10).reset_index()

df_countries_2018.columns = ['Country', 'Percentage']

df_countries_2019 = return_percentage(data_less, 'Q3').head(10).reset_index()

df_countries_2019.columns = ['Country', 'Percentage']

# print(df_countries_2019)

fig = make_subplots(rows = 1, cols = 2)

trace0 = go.Bar(x = df_countries_2018['Country'], y = df_countries_2018['Percentage'])

trace1 = go.Bar(x = df_countries_2019['Country'], y = df_countries_2019['Percentage'])

fig.add_trace(trace0, row= 1, col = 1)

fig.add_trace(trace1, row= 1, col = 2)

fig.update_layout(height=600, width=800, title_text="Comparision of Respondents by Country in 2018 and 2019", title_x = 0.5)

fig.show()
return_percentage(data_less_2018, 'Q2')

# return_percentage(data_less, 'Q1')
return_percentage(data_less, 'Q1')
# age_values_2018 = data_less_2018['Q2'].value_counts()

# age_values_2018
# age_values_2019 = data_less['Q1'].value_counts()

# age_values_2019
age_brackets = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70-79','80+']

age_values_2018 = [3037, 5141, 6159, 3776, 2253, 1360, 858, 582, 328, 273, 53, 1]

age_values_2019 = [2502, 3610, 4458, 3120, 2087, 1439, 949, 692, 422, 338, 50, 50]

fig = make_subplots(rows = 1, cols = 2)

trace0 = go.Scatter(x = age_brackets, y = age_values_2018)

trace1 = go.Scatter(x = age_brackets, y = age_values_2019)

fig.add_trace(trace0, row= 1, col = 1)

fig.add_trace(trace1, row= 1, col = 2)

fig.update_layout(height=600, width=800, title_text="Comparision of Age in 2018 and 2019", title_x = 0.5)

fig.show()
data_less
#Get all columns which have data present, get data except the NaN's and then count the number of responses of each 



type_of_work_2018 = pd.DataFrame()

type_of_work_2019 = pd.DataFrame()



for i in range(7):

    type_of_work_2018['Q11_Part_'+str(i+1)] = data_less_2018['Q11_Part_'+str(i+1)]

for i in range(6):    

    type_of_work_2019['Q9_Part_'+str(i+1)] = data_less['Q9_Part_'+str(i+1)]



# type_of_work_2019

    

    
print(type_of_work_2018['Q11_Part_7'].unique())

x_2019 = ['Analyze and understand data to influence product or business decisions', 

     'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

     'Build prototypes to explore applying machine learning to new areas',

     'Build and/or run a machine learning service that operationally improves my product or workflows',

     'Experimentation and iteration to improve existing ML models',

     'Do research that advances the state of the art of machine learning']

y_2019 = list(type_of_work_2019.count())



y_2018 = list(type_of_work_2018.count())[:-1]

x_2018 = ['Analyze and understand data to influence product or business decisions',

          'Build and/or run a machine learning service that operationally improves my product or workflows',

          'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

          'Build prototypes to explore applying machine learning to new areas',

          'Do research that advances the state of the art of machine learning',

          'None of these activities are an important part of my role at work']

print(y_2018)



fig = make_subplots(rows = 1, cols = 2)

trace0 = go.Bar(x = x_2018, y = y_2018)

trace1 = go.Bar(x = x_2019, y = y_2019)

fig.add_trace(trace0, row= 1, col = 1)

fig.add_trace(trace1, row= 1, col = 2)

fig.update_layout(height=600, width=900, title_text="Comparision of Type of work done in 2018 and 2019", title_x = 0.5, showlegend = False)

# fig.show()
data_less['Q15'].value_counts()

number_of_values = ['< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']

y_values_2019 = [3828, 4061, 3365, 1887, 1045, 576]

y_values_2018 = [4542, 5359, 4023, 2145, 1102, 500]

fig = make_subplots(rows = 1, cols = 2)

trace0 = go.Scatter(x = number_of_values, y = y_values_2018)

trace1 = go.Scatter(x = number_of_values, y = y_values_2019)

fig.add_trace(trace0, row= 1, col = 1)

fig.add_trace(trace1, row= 1, col = 2)

fig.update_layout(height=600, width=900, title_text="Comparision of Type of work done in 2018 and 2019", title_x = 0.5, showlegend = False)

# fig.show()
data_less_2018['Q24'].value_counts()
ML_algoriths_used = data_2018['Q20'].value_counts().reset_index()
algo_use = {}

for i in range(12):

    print(data_2019['Q24_Part_'+str(i+1)].value_counts())
data_2018['Q20_OTHER_TEXT']
# text = pd.DataFrame()

# text['ML_algo'] = data_2019['Q24_OTHER_TEXT'].str.lower()

# text['count'] = 1

# text.drop(0)[['ML_algo','count']].groupby('ML_algo').sum()[['count']].sort_values('count', ascending=False)



# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# plt.figure(figsize=[15,8])



# # Create and generate a word cloud image:

# ide_words = ' '.join(text['ML_algo'].drop(0).dropna().values)

# # print(ide_words)

# wordcloud = WordCloud(colormap="tab10",

#                       width=1200,

#                       height=480,

#                       normalize_plurals=False,

#                       background_color="white",

#                       random_state=5).generate(ide_words)



# # Display the generated image:

# plt.imshow(wordcloud, interpolation='bilinear')

# plt.axis("off")

# plt.show()
data_2018['Q4'].value_counts()

# pd.crosstab()
data_2019['Q4'].value_counts()
type_of_work_2019['All'] = '' 

type_of_work_2019

for i in range(6):

    if type_of_work_2019['Q9_Part_'+str(i+1)] is 'NaN':

        continue

    type_of_work_2019['All'] += type_of_work_2019['Q9_Part_'+str(i+1)]
#Type of education and work relationship
# data_2019['Q6']
type_of_work_2018
data_2018_mod = pd.DataFrame()

data_2018_mod['Q4'] = data_2018['Q4']

ar = ['Master’s degree', 'Bachelor’s degree', 'Doctoral degree']

data_2018_mod['Q4'] = data_2018_mod.loc[data_2018_mod['Q4'].isin(ar)]



work_degree = pd.concat([pd.crosstab(type_of_work_2018['Q11_Part_1'], data_2018_mod['Q4']), 

                    pd.crosstab(type_of_work_2018['Q11_Part_2'], data_2018_mod['Q4']),

                    pd.crosstab(type_of_work_2018['Q11_Part_3'], data_2018_mod['Q4']),

                    pd.crosstab(type_of_work_2018['Q11_Part_4'], data_2018_mod['Q4']),

                    pd.crosstab(type_of_work_2018['Q11_Part_5'], data_2018_mod['Q4']),

                    pd.crosstab(type_of_work_2018['Q11_Part_6'], data_2018_mod['Q4']),

                    pd.crosstab(type_of_work_2018['Q11_Part_7'], data_2018_mod['Q4'])])

# print(ml_exp[['Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data']])

print(work_degree[['Bachelor’s degree', 'Doctoral degree', 'Master’s degree']])

work_degree=work_degree.fillna(0)

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_1']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_2']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_3']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_4']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_5']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_6']))

# return_percentage(data_less_2018, 'Q3')

trace1 = go.Bar(

    x=work_degree.index,

    y=work_degree['Bachelor’s degree'].values/10138,

    name='Bachelor’s degree',

    marker=dict(

    color="#66545e")

)



trace2 = go.Bar(

    x=work_degree.index,

    y=work_degree['Master’s degree'].values/17855,

    name='Master’s degree',

    marker=dict(

    color="#4d0e20")

)



trace3 = go.Bar(

    x=work_degree.index,

    y=work_degree['Doctoral degree'].values/6231,

    name='Doctoral degree',

    marker=dict(

    color="#b35a00")

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    barmode='group',height=600,width=1200,title='Type of work done grouped by degree in 2018',title_x = 0.5,yaxis_title='Percentage of people',xaxis_title=''

)



fig = go.Figure(data=data, layout=layout)

fig.show()
data_2019_mod = pd.DataFrame()

data_2019_mod['Q4'] = data_2019['Q4']

ar = ['Master’s degree', 'Bachelor’s degree', 'Doctoral degree']

data_2019_mod['Q4'] = data_2019_mod.loc[data_2019_mod['Q4'].isin(ar)]



work_degree = pd.concat([pd.crosstab(type_of_work_2019['Q9_Part_1'], data_2019_mod['Q4']), 

                    pd.crosstab(type_of_work_2019['Q9_Part_2'], data_2019_mod['Q4']),

                    pd.crosstab(type_of_work_2019['Q9_Part_3'], data_2019_mod['Q4']),

                    pd.crosstab(type_of_work_2019['Q9_Part_4'], data_2019_mod['Q4']),

                    pd.crosstab(type_of_work_2019['Q9_Part_5'], data_2019_mod['Q4']),

                    pd.crosstab(type_of_work_2019['Q9_Part_6'], data_2019_mod['Q4'])])

# print(ml_exp[['Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data']])

# print(ml_exp[['Bachelor’s degree', 'Doctoral degree', 'Master’s degree']])

work_degree=work_degree.fillna(0)

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_1']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_2']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_3']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_4']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_5']))

# print(pd.crosstab(data_2019_mod['Q4'], type_of_work_2019['Q9_Part_6']))

# return_percentage(data_less_2018, 'Q3')

trace1 = go.Bar(

    x=work_degree.index,

    y=work_degree['Bachelor’s degree'].values/5641,

    name='Bachelor’s degree',

    marker=dict(

    color="#66545e")

)



trace2 = go.Bar(

    x=work_degree.index,

    y=work_degree['Master’s degree'].values/11879,

    name='Master’s degree',

    marker=dict(

    color="#4d0e20")

)



trace3 = go.Bar(

    x=work_degree.index,

    y=work_degree['Doctoral degree'].values/4853,

    name='Doctoral degree',

    marker=dict(

    color="#b35a00")

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    barmode='group',height=600,width=1200,title='Type of work done grouped by degree in 2019',title_x = 0.5,yaxis_title='Percentage of people',xaxis_title=''

)



fig = go.Figure(data=data, layout=layout)

fig.show()