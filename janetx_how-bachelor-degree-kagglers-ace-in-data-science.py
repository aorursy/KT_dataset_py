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
import pandas as pd

multiple_choice_responses = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv")

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")
import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

import pycountry



py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.subplots import make_subplots
multiple_choice_responses
print(multiple_choice_responses[:1].values)
undergraduates = multiple_choice_responses[multiple_choice_responses['Q4'] == "Bachelor’s degree"]
undergraduates_count = multiple_choice_responses['Q4'][multiple_choice_responses['Q4'] == "Bachelor’s degree"].value_counts().values[0]

masters_count = multiple_choice_responses['Q4'][multiple_choice_responses['Q4'] == "Master’s degree"].value_counts().values[0]

doctors_count = multiple_choice_responses['Q4'][multiple_choice_responses['Q4'] == "Doctoral degree"].value_counts().values[0]
print('number of bachelor degree kagglers:', undergraduates_count)

print('number of master degree kagglers:', masters_count)

print('number of doctoral degree kagglers:', doctors_count)
# degree_count = pd.DataFrame(data = [undergraduates_count,masters_count,doctors_count],index = ['bachlors','masters','doctors'])    



degree_total = len(multiple_choice_responses)

degree_bachlor = undergraduates_count/degree_total*100

degree_master = masters_count/degree_total*100

degree_doctor = doctors_count/degree_total*100
degree_doctor
df = pd.DataFrame(data = [degree_bachlor,degree_master,degree_doctor],

                          columns = ['Number of responses'], index = ['bachelor','master','doctor'])

df.index.names = ['Highest Education of respondents']



df.plot(kind='bar',color='c',legend=False,linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(10,8)

plt.title('Percentages of respondent highest education', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.ylabel('Percentage of Respondents',fontsize=15)

plt.show()
students_count = undergraduates['Q1'][(undergraduates['Q1'] == "18-21") | (multiple_choice_responses['Q1'] == "22-24")].value_counts().values[0]

working_count = undergraduates_count-students_count
print('number of bachelor degree kagglers age<24:', students_count)

print('number of bachelor degree kagglers age>24:', working_count)
df_age = pd.DataFrame(data = [students_count,working_count],

                          columns = ['Number of respondents'], index = ['18-24','>24'])

df_age.index.names = ['Age of respondents']



df_age.plot(kind='bar',color='c',legend=False,linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(10,8)

plt.title('Age of Bachelor degree respondents', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.ylabel('Number of Respondents',fontsize=15)

plt.show()
undergraduates['Q23']=undergraduates['Q23'].fillna(value="no experience")
undergraduates['Q23'][undergraduates['Q23'] == "< 1 years"].value_counts()
no_exp = undergraduates['Q23'][undergraduates['Q23'] == "no experience"].value_counts().values[0]

less_exp =  undergraduates['Q23'][undergraduates['Q23'] == "< 1 years"].value_counts().values[0] + undergraduates['Q23'][undergraduates['Q23'] == "1-2 years"].value_counts().values[0]

some_exp = undergraduates['Q23'][undergraduates['Q23'] == "2-3 years"].value_counts().values[0] + undergraduates['Q23'][undergraduates['Q23'] == "3-4 years"].value_counts().values[0] + undergraduates['Q23'][undergraduates['Q23'] == "4-5 years"].value_counts().values[0] + undergraduates['Q23'][undergraduates['Q23'] == "5-10 years"].value_counts().values[0]

lots_exp = undergraduates['Q23'][undergraduates['Q23'] == "10-15 years"].value_counts().values[0] + undergraduates['Q23'][undergraduates['Q23'] == "20+ years"].value_counts().values[0]
print (no_exp)

print(less_exp)

print(some_exp)

print(lots_exp)
len(undergraduates['Q23'])
df_exp = pd.DataFrame(data = [no_exp,less_exp,some_exp,lots_exp],

                          columns = ['Number of respondents'], index = ['no experience','< 2 years','2-10 years','> 10 years'])

df_exp.index.names = ['Level of experience']



df_exp.plot(kind='bar',color='c',legend=False,linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(10,8)

plt.title('Experience in Machine Learning of Bachelor degree respondents', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.ylabel('Number of Respondents',fontsize=15)

plt.show()
undergraduates['Q5']
topnum = 15

count_title = undergraduates['Q5'].value_counts()[:topnum].reset_index()
count_title
pie_title = go.Pie(labels=count_title['index'],values=count_title['Q5'],name="bachelor degree respondents' job titles",hole=0.5,domain={'x': [0.1,0.66]})

layout = dict(title = 'Top-12 job titles with bachelor degree respondents', font=dict(size=12), legend=dict(orientation="h"),

              annotations = [dict(x=0.38, y=0.5, text='job titles', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[pie_title], layout=layout)

py.iplot(fig)
for n in undergraduates.columns:

    print(n)
undergraduates['Q10']
top_num = 30

count_salary = undergraduates['Q10'].value_counts()[:top_num].reset_index()
count_salary
df_comp = pd.DataFrame(data = count_salary['Q10'].values,

                          columns = ['Number of respondents'], index = count_salary['index'].values)

df_comp.index.names = ['Compensation']



df_comp.plot(kind='bar',color='c',legend=False,linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(40,20)

plt.title('Compensation of Bachlor degree respondents', fontsize = 20)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='center')

plt.ylabel('Number of Respondents',fontsize=20)

plt.show()
undergraduates['Q10']=undergraduates['Q10'].fillna(value="$0-999")
no_money = len(undergraduates['Q10'][undergraduates['Q10'] == "$0-999"])

less_money =  len(undergraduates['Q10'][undergraduates['Q10'] == "1,000-1,999"]) + len(undergraduates['Q10'][undergraduates['Q10'] == "2,000-2,999"]) + len(undergraduates['Q10'][undergraduates['Q10'] == "3,000-3,999"]) + len(undergraduates['Q10'][undergraduates['Q10'] == "4,000-4,999"]) + len(undergraduates['Q10'][undergraduates['Q10'] == "5,000-7,499"]) + len(undergraduates['Q10'][undergraduates['Q10'] == "7,500-9,999"])

some_money = len(undergraduates['Q10'][undergraduates['Q10'] == "10,000-14,999"]) + len(undergraduates['Q10'][undergraduates['Q10'] == "15,000-19,999"]) + len(undergraduates['Q23'][undergraduates['Q23'] == "15,000-19,999"]) + len(undergraduates['Q23'][undergraduates['Q23'] == "20,000-24,999"]) + len(undergraduates['Q10'][undergraduates['Q10'] == "25,000-29,999"])

more_money = len(undergraduates['Q10'][(undergraduates['Q10'] == "30,000-39,999") | (undergraduates['Q10'] == "40,000-49,999")])

much_money = len(undergraduates['Q10'][(undergraduates['Q10'] == "50,000-59,999") | (undergraduates['Q10'] == "60,000-69,999")])

lots_money = len(undergraduates['Q10'][(undergraduates['Q10'] == "70,000-79,999") | (undergraduates['Q10'] == "80,000-89,999") | (undergraduates['Q10'] == "90,000-99,999")])

high_money = len(undergraduates['Q10'][(undergraduates['Q10'] == "100,000-124,999") | (undergraduates['Q10'] == "125,000-149,999")])

tons_money = len(undergraduates['Q10'][(undergraduates['Q10'] == "150,000-199,999") | (undergraduates['Q10'] == "200,000-249,999") | (undergraduates['Q10'] == "250,000-299,999")])

insane_money = len(undergraduates['Q10'][(undergraduates['Q10'] == "300,000-500,000") | (undergraduates['Q10'] == "> $500,000")])
df_money = pd.DataFrame(data = [no_money, less_money, some_money, more_money, much_money, lots_money, high_money, tons_money, insane_money],

                          columns = ['Number of respondents'], index = ["$0-999", "$1,000-9,999", "$10,000-29,999", "$30,000-49,999", "$50,000-69,999", "$70,000-99,999", "$100,000-149,999", "$150,000-299,999", "> $300,000"])

df_money.index.names = ['Compensation']



df_money.plot(kind='bar',color='g',legend=False,linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(18,8)

plt.title('Compensation of Bachelor degree respondents', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='center')

plt.ylabel('Number of Respondents',fontsize=15)

plt.show()
rich_guys = undergraduates['Q5'][(undergraduates['Q10'] == "100,000-124,999") | (undergraduates['Q10'] == "125,000-149,999") | (undergraduates['Q10'] == "150,000-199,999") | (undergraduates['Q10'] == "200,000-249,999") | (undergraduates['Q10'] == "250,000-299,999") | (undergraduates['Q10'] == "300,000-500,000") | (undergraduates['Q10'] == "> $500,000")]
rich_title = rich_guys.value_counts()[:topnum].reset_index()
rich_title
pie_rich = go.Pie(labels=rich_title['index'],values=rich_title['Q5'],name="job titles for people making 6 figures",hole=0.5,domain={'x': [0.1,0.66]})

layout = dict(title = 'Job titles for people making 6 figures', font=dict(size=12), legend=dict(orientation="h"),

              annotations = [dict(x=0.38, y=0.5, text='job titles', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[pie_rich], layout=layout)

py.iplot(fig)
rich_ages = undergraduates['Q1'][(undergraduates['Q10'] == "100,000-124,999") | (undergraduates['Q10'] == "125,000-149,999") | (undergraduates['Q10'] == "150,000-199,999") | (undergraduates['Q10'] == "200,000-249,999") | (undergraduates['Q10'] == "250,000-299,999") | (undergraduates['Q10'] == "300,000-500,000") | (undergraduates['Q10'] == "> $500,000")]
rich_age = rich_ages.value_counts()[:topnum].reset_index()
rich_age
sci_ages = undergraduates['Q1'][((undergraduates['Q10'] == "100,000-124,999") | (undergraduates['Q10'] == "125,000-149,999") | (undergraduates['Q10'] == "150,000-199,999") | (undergraduates['Q10'] == "200,000-249,999") | (undergraduates['Q10'] == "250,000-299,999") | (undergraduates['Q10'] == "300,000-500,000") | (undergraduates['Q10'] == "> $500,000")) & (undergraduates['Q5'] == "Data Scientist")]
sci_age = sci_ages.value_counts()[:topnum].reset_index()
sci_age


fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=rich_age['index'], values=rich_age['Q1'], name="ages for people making 6 figures"),

              1, 1)

fig.add_trace(go.Pie(labels=sci_age['index'], values=sci_age['Q1'], name="ages for data scientists making 6 figures"),

              1, 2)

fig.update_traces(hole=.5, hoverinfo="label+percent+name")

fig.update_layout(

    title_text="Age Distribution for all bachelor degree respondents and data scientist making 6 figures",font=dict(size=12), legend=dict(orientation="h"),

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='all', x=0.20, y=0.5, font_size=20, showarrow=False),

                 dict(text='data scientists', x=0.88, y=0.5, font_size=20, showarrow=False)])



fig.show()
rich_countries = undergraduates['Q3'][(undergraduates['Q10'] == "100,000-124,999") | (undergraduates['Q10'] == "125,000-149,999") | (undergraduates['Q10'] == "150,000-199,999") | (undergraduates['Q10'] == "200,000-249,999") | (undergraduates['Q10'] == "250,000-299,999") | (undergraduates['Q10'] == "300,000-500,000") | (undergraduates['Q10'] == "> $500,000")]

rich_country = rich_countries.value_counts()[:topnum].reset_index()

sci_countries = undergraduates['Q3'][((undergraduates['Q10'] == "100,000-124,999") | (undergraduates['Q10'] == "125,000-149,999") | (undergraduates['Q10'] == "150,000-199,999") | (undergraduates['Q10'] == "200,000-249,999") | (undergraduates['Q10'] == "250,000-299,999") | (undergraduates['Q10'] == "300,000-500,000") | (undergraduates['Q10'] == "> $500,000")) & (undergraduates['Q5'] == "Data Scientist")]

sci_country = sci_countries.value_counts()[:topnum].reset_index()
fig_country = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig_country.add_trace(go.Pie(labels=rich_country['index'], values=rich_country['Q3'], name="countries for people making 6 figures"),

              1, 1)

fig_country.add_trace(go.Pie(labels=sci_country['index'], values=sci_country['Q3'], name="countries for data scientists making 6 figures"),

              1, 2)

fig_country.update_traces(hole=.5, hoverinfo="label+percent+name")

fig_country.update_layout(

    title_text="Country Distribution for all bachelor degree respondents and data scientist making 6 figures",font=dict(size=12), legend=dict(orientation="h"),

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='all', x=0.20, y=0.5, font_size=20, showarrow=False),

                 dict(text='data scientists', x=0.88, y=0.5, font_size=20, showarrow=False)])



fig_country.show()
twit_under= len(undergraduates[undergraduates['Q12_Part_1'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

hack_under= len(undergraduates[undergraduates['Q12_Part_2'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

reddit_under= len(undergraduates[undergraduates['Q12_Part_3'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

kaggle_under= len(undergraduates[undergraduates['Q12_Part_4'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

course_under= len(undergraduates[undergraduates['Q12_Part_5'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

youtube_under= len(undergraduates[undergraduates['Q12_Part_6'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

podcast_under= len(undergraduates[undergraduates['Q12_Part_7'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

blogs_under= len(undergraduates[undergraduates['Q12_Part_8'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

journal_under= len(undergraduates[undergraduates['Q12_Part_9'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

slack_under= len(undergraduates[undergraduates['Q12_Part_10'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

none_under= len(undergraduates[undergraduates['Q12_Part_11'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

other_under= len(undergraduates[undergraduates['Q12_Part_12'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

print(twit_under)

print(hack_under)

print(reddit_under)

print(kaggle_under)

print(course_under)

print(youtube_under)

print(podcast_under)

print(blogs_under)

print(journal_under)

print(slack_under)

print(none_under)

print(other_under)

df_media = pd.DataFrame(data = [twit_under, hack_under, reddit_under, kaggle_under, course_under, youtube_under, podcast_under, blogs_under, journal_under, slack_under, none_under, other_under],

                          columns = ['Number of responses'], index = ['Twitter','Hacker News','Reddit','Kaggle','course forum','Youtube','Podcast','Blogs','Journals','Slack','None','Other'])



df_media.plot(kind='bar',color='c',legend=False,linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(20,8)

plt.title('Percentages of Media Sources Used by Respondents', fontsize = 15)

plt.xticks(rotation=0,fontsize='14', horizontalalignment='center')

plt.xlabel('Media Sources', fontsize=15)

plt.ylabel('Percentage of Respondents',fontsize=15)

plt.show()
udacity_under= len(undergraduates[undergraduates['Q13_Part_1'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

coursera_under= len(undergraduates[undergraduates['Q13_Part_2'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

edx_under= len(undergraduates[undergraduates['Q13_Part_3'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

datacamp_under= len(undergraduates[undergraduates['Q13_Part_4'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

dataquest_under= len(undergraduates[undergraduates['Q13_Part_5'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

kaggle_under= len(undergraduates[undergraduates['Q13_Part_6'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

fastai_under= len(undergraduates[undergraduates['Q13_Part_7'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

udemy_under= len(undergraduates[undergraduates['Q13_Part_8'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

linkedin_under= len(undergraduates[undergraduates['Q13_Part_9'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

university_under= len(undergraduates[undergraduates['Q13_Part_10'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

noneC_under= len(undergraduates[undergraduates['Q13_Part_11'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

otherC_under= len(undergraduates[undergraduates['Q13_Part_12'].isnull()==False])/len(undergraduates['Q12_Part_1'])*100

print(udacity_under)

print(coursera_under)

print(edx_under)

print(datacamp_under)

print(dataquest_under)

print(kaggle_under)

print(fastai_under)

print(udemy_under)

print(linkedin_under)

print(university_under)

print(noneC_under)

print(otherC_under)

df_course = pd.DataFrame(data = [udacity_under, coursera_under, edx_under, datacamp_under, dataquest_under, kaggle_under, fastai_under, udemy_under, linkedin_under, university_under, noneC_under, otherC_under],

                          columns = ['Number of responses'], index = ['Udacity','Coursera','Edx','Datacamp','Dataquest','Kaggle','Fast.ai','Udemy','Linkedin','University','None','Other'])



df_course.plot(kind='bar',color='c',legend=False,linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(20,8)

plt.title('Percentages of Course Platform Used by Respondents', fontsize = 15)

plt.xticks(rotation=0,fontsize='14', horizontalalignment='center')

plt.xlabel('Course Platforms', fontsize=15)

plt.ylabel('Percentage of Respondents',fontsize=15)

plt.show()
colors1 = ['dodgerblue', 'plum', '#F0A30A','#8c564b'] 

counts = undergraduates['Q2'].value_counts(sort=True)

labels = counts.index

values = counts.values



pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors1))

layout = go.Layout(title='Gender Distribution for Bachelor degree kagglers')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
# undergraduates['Q3'].replace({'United States of America':'United States','Viet Nam':'Vietnam','China':"People 's Republic of China","United Kingdom of Great Britain and Northern Ireland":'United Kingdom',"Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)
topn = 10

count_country = undergraduates['Q3'].value_counts()[:topn].reset_index()
count_country
pie_country = go.Pie(labels=count_country['index'],values=count_country['Q3'],name="country of residence",hole=0.4,domain={'x': [0.2,0.66]})

layout = dict(title = 'Top-10 countries with bachelor degree respondents', font=dict(size=10), legend=dict(orientation="h"),

              annotations = [dict(x=0.43, y=0.5, text='countries', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[pie_country], layout=layout)

py.iplot(fig)