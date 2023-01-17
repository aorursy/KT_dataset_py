

import numpy as np 

import pandas as pd 



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# palette of colors to be used for plots

colors = ["steelblue","dodgerblue","lightskyblue","powderblue","cyan","deepskyblue","cyan","darkturquoise","paleturquoise","turquoise"]





# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')
# Importing the 2019 survey dataset



#Importing the 2019 Dataset

data_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

data_2019.columns = data_2019.iloc[0]

data_2019=data_2019.drop([0])



# Helper functions



def return_count(data,question_part):

    """Counts occurences of each value in a given column"""

    counts_df = data[question_part].value_counts().to_frame()

    return counts_df



def return_percentage(data,question_part):

    """Calculates percent of each value in a given column"""

    total = data[question_part].count()

    counts_df= data[question_part].value_counts().to_frame()

    percentage_df = (counts_df*100)/total

    return percentage_df





    

def plot_graph(data,question,title,x_axis_title,y_axis_title):

    """ plots a percentage bar graph"""

    df = return_percentage(data,question)

    

    trace1 = go.Bar(

                    x = df.index,

                    y = df[question],

                    #orientation='h',

                    marker = dict(color='dodgerblue',

                                 line=dict(color='black',width=1)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title,width=800, height=500,

                       xaxis=dict(type='category',categoryorder='array',categoryarray=salary_order,title=y_axis_title),

                       yaxis= dict(title=x_axis_title))

                       

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)    
# Replace missing answer with “not answer”

df_2019 = data_2019

df_2019['What is the size of the company where you are employed?'] = df_2019['What is the size of the company where you are employed?'].fillna('not answer')



# Splitting all the datasets genderwise

male_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Male'].loc[df_2019['What is the size of the company where you are employed?'] != 'not answer']

female_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Female'].loc[df_2019['What is the size of the company where you are employed?'] != 'not answer']



# Separation of data set by the number of employees in the company

employees_2019_0 = df_2019[df_2019['What is the size of the company where you are employed?'] == '0-49 employees']

employees_2019_50 = df_2019[df_2019['What is the size of the company where you are employed?'] == '50-249 employees']

employees_2019_250 = df_2019[df_2019['What is the size of the company where you are employed?'] == '250-999 employees']

employees_2019_1000 = df_2019[df_2019['What is the size of the company where you are employed?'] == '1000-9,999 employees']

employees_2019_NotAnswer = df_2019[df_2019['What is the size of the company where you are employed?'] == 'not answer']

employees_2019_10000 = df_2019[df_2019['What is the size of the company where you are employed?'] == '> 10,000 employees']



# Distribution of respondents by the number of employees in the company in 2019

count_employees = df_2019['What is the size of the company where you are employed?'].value_counts()[:].reset_index()



pie_employees = go.Pie(labels=count_employees['index'],values=count_employees['What is the size of the company where you are employed?'],name="Employees",hole=0.5,domain={'x': [0,1]})



layout = dict(title = 'Distribution of respondents by the number of employees in the company in 2019', font=dict(size=10), legend=dict(orientation="h"),

              annotations = [dict(x=0.5, y=0.5, text='Employees', showarrow=False, font=dict(size=20))])



fig = dict(data=[pie_employees], layout=layout)

py.iplot(fig)

import textwrap

from  textwrap import fill



# Positions of respondents who did not answer about the size of the company

x_axis=range(12)

role = employees_2019_NotAnswer['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts().to_frame()

role = role/len(employees_2019_NotAnswer)*100

labels = role.index



role.plot(kind='bar',color='dodgerblue',linewidth=1,edgecolor='k',legend=None)

plt.gcf().set_size_inches(16,7)

plt.title('Positions of respondents who did not answer about the size of the company', fontsize = 16)

plt.xticks(x_axis, [textwrap.fill(label, 9) for label in labels], 

           rotation = 5, fontsize=13, horizontalalignment="center")

plt.ylabel('Percentage of Respondents',fontsize=15)

plt.xlabel('Role',fontsize=15)

plt.show()
# Column of dataset under consideration

q = 'What is the size of the company where you are employed?'



#Preprocessing of data on the ratio of companies by gender

label_employees = ['0-49 employees',

                '50-249 employees',

                '250-999 employees',

                '1000-9,999 employees',

                '> 10,000 employees']



df1 = return_count(female_2019,q)

df2 = return_count(male_2019, q)



fig = go.Figure(data=[

    go.Bar(name='Females', y=df1[q], x=df1.index,marker_color='lightcoral'),

    go.Bar(name='Males', y=df2[q], x=df2.index,marker_color='dodgerblue')

])    

fig.update_layout(barmode='group',title='Number of respondents in different companies by gender',xaxis=dict(title='Company size',categoryarray=label_employees),yaxis=dict(title='Number of working respondents'))

fig.show() 
# Column of dataset under consideration

q = 'What is your age (# years)?'

def concat_age(df):

    df[q] = np.where(df[q].isin(['18-21','22-24']), '18-24',df[q])

    df[q] = np.where(df[q].isin(['25-29','30-34']), '25-34',df[q])

    df[q] = np.where(df[q].isin(['35-39','40-44']), '35-44' ,df[q])

    df[q] = np.where(df[q].isin(['45-49','50-54']), '45-54',df[q])

    df[q] = np.where(df[q].isin(['55-59','60-69']), '55-69',df[q])

    df[q] = np.where(df[q].isin(['70+']),'70+' ,df[q])

    return df



#Preprocessing of data on the ratio of companies by age

label = ['18-24', '25-34', '35-44', '45-54', '55-69', '70+']



employees_2019_0 = concat_age(employees_2019_0)

employees_2019_50 = concat_age(employees_2019_50)

employees_2019_250 = concat_age(employees_2019_250)

employees_2019_1000 = concat_age(employees_2019_1000)

employees_2019_10000 = concat_age(employees_2019_10000)



df1 = return_count(employees_2019_0,q)

df2 = return_count(employees_2019_50, q)

df3 = return_count(employees_2019_250, q)

df4 = return_count(employees_2019_1000, q)

df5 = return_count(employees_2019_10000, q)



fig = go.Figure(data=[

    go.Bar(name='0-49 employees', y=df1[q], x=df1.index,marker=dict(color='#F08080')),

    go.Bar(name='50-249 employees', y=df2[q], x=df2.index,marker=dict(color='#FFA500')),

    go.Bar(name='250-999 employees', y=df3[q], x=df3.index,marker=dict(color='#00FA9A')),

    go.Bar(name='1000-9,999 employees', y=df4[q], x=df4.index,marker=dict(color='#1E90FF')),

    go.Bar(name='> 10,000 employees', y=df5[q], x=df5.index,marker=dict(color='#8A2BE2'))

])    

fig.update_layout(barmode='group',title='The number of respondents in different companies by age',xaxis=dict(title='Age of respondents',categoryarray=label),yaxis=dict(title='Number of working respondents'))

fig.show() 
# Column of dataset under consideration

q = 'What is the size of the company where you are employed?'



#Pre-processing of data on the ratio of companies by education

q1 = 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'

Master = df_2019[df_2019[q1] == 'Master’s degree']

Professional = df_2019[df_2019[q1] == 'Professional degree']

Bachelor = df_2019[df_2019[q1] == 'Bachelor’s degree']

study = df_2019[df_2019[q1] == 'Some college/university study without earning a bachelor’s degree']

Doctoral = df_2019[df_2019[q1] == 'Doctoral degree']

not_answer = df_2019[df_2019[q1] == 'I prefer not to answer']

no_formal_education = df_2019[df_2019[q1] == 'No formal education past high school']



label = ['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']



df1 = return_count(Master,q)

df2 = return_count(Professional, q)

df3 = return_count(Bachelor, q)

df4 = return_count(study, q)

df5 = return_count(Doctoral, q)

df6 = return_count(not_answer, q)

df7 = return_count(no_formal_education, q)



fig = go.Figure(data=[

    go.Bar(name='Master’s degree', y=df1[q], x=df1.index,marker=dict(color='#F08080')),

    go.Bar(name='Professional degree', y=df2[q], x=df2.index,marker=dict(color='#FFA500')),

    go.Bar(name='Bachelor’s degree', y=df3[q], x=df3.index,marker=dict(color='#00FA9A')),

    go.Bar(name='Some college/university study without earning a bachelor’s degree', y=df4[q], x=df4.index,marker=dict(color='#1E90FF')),

    go.Bar(name='Doctoral degree', y=df5[q], x=df5.index,marker=dict(color='#8A2BE2')),

    go.Bar(name='I prefer not to answer', y=df6[q], x=df6.index,marker=dict(color='#5A2BE2')),

    go.Bar(name='No formal education past high school', y=df7[q], x=df7.index,marker=dict(color='#8A2902'))

])    

fig.update_layout(barmode='group',title='The number of respondents in different companies by education',xaxis=dict(title='Company size',categoryarray=label),yaxis=dict(title='Number of respondents'))

fig.show() 
# Column of dataset under consideration

q = 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice'



#Preprocessing of data on the ratio of companies by role

label = ['Software Engineer', 'Other', 'Data Scientist', 'Statistician',

 'Product/Project Manager', 'Data Analyst', 'Research Scientist',

 'Business Analyst', 'Data Engineer', 'DBA/Database Engineer']





df1 = return_count(employees_2019_0,q)

df2 = return_count(employees_2019_50, q)

df3 = return_count(employees_2019_250, q)

df4 = return_count(employees_2019_1000, q)

df5 = return_count(employees_2019_10000, q)



fig = go.Figure(data=[

    go.Bar(name='0-49 employees', y=df1[q], x=df1.index,marker=dict(color='#F08080')),

    go.Bar(name='50-249 employees', y=df2[q], x=df2.index,marker=dict(color='#FFA500')),

    go.Bar(name='250-999 employees', y=df3[q], x=df3.index,marker=dict(color='#00FA9A')),

    go.Bar(name='1000-9,999 employees', y=df4[q], x=df4.index,marker=dict(color='#1E90FF')),

    go.Bar(name='> 10,000 employees', y=df5[q], x=df5.index,marker=dict(color='#8A2BE2'))

])    

fig.update_layout(barmode='group',title='The number of respondents in different companies by role',xaxis=dict(title='Role',categoryarray=label),yaxis=dict(title='Number of working respondents'))

fig.show() 
# Column of dataset under consideration

q = 'Have you ever used a TPU (tensor processing unit)?'



#Preprocessing of data on the ratio of companies by frequency of use of the tensor processor'

label = ['Never', 'Once','2-5 times', '6-24 times',  '> 25 times']



df1 = return_count(employees_2019_0,q)

df2 = return_count(employees_2019_50, q)

df3 = return_count(employees_2019_250, q)

df4 = return_count(employees_2019_1000, q)

df5 = return_count(employees_2019_10000, q)



fig = go.Figure(data=[

    go.Bar(name='0-49 employees', y=df1[q], x=df1.index,marker=dict(color='#F08080')),

    go.Bar(name='50-249 employees', y=df2[q], x=df2.index,marker=dict(color='#FFA500')),

    go.Bar(name='250-999 employees', y=df3[q], x=df3.index,marker=dict(color='#00FA9A')),

    go.Bar(name='1000-9,999 employees', y=df4[q], x=df4.index,marker=dict(color='#1E90FF')),

    go.Bar(name='> 10,000 employees', y=df5[q], x=df5.index,marker=dict(color='#8A2BE2'))

])    

fig.update_layout(barmode='group',title='The number of respondents in different companies by frequency of use of the tensor processor',xaxis=dict(title='Frequency of use of the tensor processor',categoryarray=label),yaxis=dict(title='Number of working respondents'))

fig.show() 
# Column of dataset under consideration

q = 'For how many years have you used machine learning methods?'



#Preprocessing of data on the ratio of companies by number of years using machine learning methods

label = ['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years']



df1 = return_count(employees_2019_0,q)

df2 = return_count(employees_2019_50, q)

df3 = return_count(employees_2019_250, q)

df4 = return_count(employees_2019_1000, q)

df5 = return_count(employees_2019_10000, q)



fig = go.Figure(data=[

    go.Bar(name='0-49 employees', y=df1[q], x=df1.index,marker=dict(color='#F08080')),

    go.Bar(name='50-249 employees', y=df2[q], x=df2.index,marker=dict(color='#FFA500')),

    go.Bar(name='250-999 employees', y=df3[q], x=df3.index,marker=dict(color='#00FA9A')),

    go.Bar(name='1000-9,999 employees', y=df4[q], x=df4.index,marker=dict(color='#1E90FF')),

    go.Bar(name='> 10,000 employees', y=df5[q], x=df5.index,marker=dict(color='#8A2BE2'))

])    

fig.update_layout(barmode='group',title='The number of respondents in different companies by number of years using machine learning methods',xaxis=dict(title='Number of years using machine learning methods',categoryarray=label),yaxis=dict(title='Number of working respondents'))

fig.show() 
# Column of dataset under consideration

q = 'What is your current yearly compensation (approximate $USD)?'



#Preprocessing the salary data to get standard salary categories

def salary(df):

    df['Salary Range'] = np.where(df[q].isin(['$0-999','1,000-1,999','2,000-2,999','3,000-3,999',

                     '4,000-4,999','5,000-7,499','7,500-9,999']),'0-10,000',' ')

    df['Salary Range'] = np.where(df[q].isin(['10,000-14,999','15,000-19,999',]),'10-20,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['20,000-24,999','25,000-29,999',]),'20-30,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['30,000-39,999']),'30-40,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['40,000-49,999']),'40-50,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['50,000-59,999']),'50-60,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['60,000-69,999']),'60-70,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['70,000-79,999']),'70-80,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['80,000-89,999']),'80-90,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['90,000-99,999']),'90-100,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['100,000-124,999']),'100-125,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['125,000-149,999']),'125-150,000', df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['150,000-199,999']),'150-200,000',df['Salary Range'])

    df['Salary Range'] = np.where(df[q].isin(['200,000-249,999', '250,000-299,999', '300,000-400,000', 

                                                       '400,000-500,000', '> $500,000']),'200,000+',df['Salary Range'])

    return df





salary_order2 = ['0-10,000',

                '10-20,000',

                '20-30,000',

                '30-40,000',

                '40-50,000',

                '50-60,000',

                '60-70,000',

                '70-80,000',

                '80-90,000',

                '90-100,000',

                '100-125,000',

                '125-150,000',

                '150-200,000',

                '200,000+']



df1 = salary(employees_2019_0)

df2 = salary(employees_2019_50)

df3 = salary(employees_2019_250)

df4 = salary(employees_2019_1000)

df5 = salary(employees_2019_10000)



df1 = return_count(df1,'Salary Range')

df2 = return_count(df2, 'Salary Range')

df3 = return_count(df3, 'Salary Range')

df4 = return_count(df4, 'Salary Range')

df5 = return_count(df5, 'Salary Range')



fig = go.Figure(data=[

    go.Bar(name='0-49 employees', y=df1['Salary Range'], x=df1.index,marker=dict(color='#F08080')),

    go.Bar(name='50-249 employees', y=df2['Salary Range'], x=df2.index,marker=dict(color='#FFA500')),

    go.Bar(name='250-999 employees', y=df3['Salary Range'], x=df3.index,marker=dict(color='#00FA9A')),

    go.Bar(name='1000-9,999 employees', y=df4['Salary Range'], x=df4.index,marker=dict(color='#1E90FF')),

    go.Bar(name='> 10,000 employees', y=df5['Salary Range'], x=df5.index,marker=dict(color='#8A2BE2'))

])    

fig.update_layout(barmode='group',title='The number of respondents in different companies by salaries in 2019',xaxis=dict(title='Annual Salary in USD',categoryarray=salary_order2),yaxis=dict(title='Count of respondents'))

fig.show()    