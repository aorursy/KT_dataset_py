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

data = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv")

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")

req = data[1:]
india = req[req.Q3 == 'India'].reset_index(drop=True)
#2018 survey responses

import pandas as pd

SurveySchema = pd.read_csv("../input/kaggle-survey-2018/SurveySchema.csv")

freeFormResponses = pd.read_csv("../input/kaggle-survey-2018/freeFormResponses.csv")

data2 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")

req2 = data2[1:]
#2018 survey respondants from India

india2 = req2[req2.Q3 == 'India'].reset_index(drop=True)
current_title = india.groupby('Q5').count().reset_index().loc[:,'Q5':'Time from Start to Finish (seconds)'].rename(columns={'Q5':'Current_Postion_Of_Participant','Time from Start to Finish (seconds)':'Frequency'})
t = current_title.sort_values(by=['Frequency'],ascending=False).reset_index(drop=True)
t['perc'] = (t['Frequency']*100)/t['Frequency'].sum()
import plotly.graph_objects as go



labels = t['Current_Postion_Of_Participant']

values = t['perc']



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(

    title="Proportion of Work Profession of survey paricipants in India",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)

current_age2 = india2.groupby('Q1').count().reset_index().loc[:,'Q1':'Time from Start to Finish (seconds)'].rename(columns={'Q1':'Age_Group','Time from Start to Finish (seconds)':'Frequency'})
current_age2.sort_values(by=['Frequency'],ascending=False,inplace=True)
current_age2['perc']=(current_age2['Frequency']*100)/current_age2['Frequency'].sum()
import plotly.graph_objects as go



labels = current_age2['Age_Group']

values = current_age2['perc']



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(

    title="Proportion of Age-Groups of survey paricipants in India in 2019",

    font=dict(

        family="Courier New, monospace",

        size=15,

        color="#7f7f7f"

    )

)

current_age_2 = india2.groupby('Q2').count().reset_index().loc[:,'Q2':'Time from Start to Finish (seconds)'].rename(columns={'Q2':'Age_Group','Time from Start to Finish (seconds)':'Frequency'})
current_age_2.sort_values(by=['Frequency'],ascending=False,inplace=True)
current_age_2['perc']=(current_age_2['Frequency']*100)/current_age_2['Frequency'].sum()
import plotly.graph_objects as go



labels = current_age_2['Age_Group']

values = current_age_2['perc']



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(

    title="Proportion of Age-Groups of survey paricipants in India in 2018",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)

current_degree = india.groupby('Q4').count().reset_index().loc[:,'Q4':'Time from Start to Finish (seconds)'].rename(columns={'Q4':'Educational_level','Time from Start to Finish (seconds)':'Frequency'})
current_degree.sort_values(by=['Frequency'],ascending=False,inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,10))



plt.xticks(current_degree.index, current_degree.Educational_level.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Educational_level", y="Frequency",data=current_degree)

ax.set_title('Proportion of participants from a Given Educational Background in India in 2019',fontsize=15)

ax.set(xlabel='Educational Level of participant', ylabel='Number of participants in the survey')

current_degree2 = india2.groupby('Q4').count().reset_index().loc[:,'Q4':'Time from Start to Finish (seconds)'].rename(columns={'Q4':'Educational_level','Time from Start to Finish (seconds)':'Frequency'})
current_degree2.sort_values(by=['Frequency'],ascending=False,inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,10))



plt.xticks(current_degree2.index, current_degree2.Educational_level.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Educational_level", y="Frequency",data=current_degree2)

ax.set_title('Proportion of participants from a Given Educational Background in India in 2018',fontsize=15)

ax.set(xlabel='Educational Level of participant', ylabel='Number of participants in the survey')

current_degree_job_compensation = india.groupby(['Q4','Q5','Q10']).count().reset_index().loc[:,'Q4':'Time from Start to Finish (seconds)'].rename(columns={'Q4':'Educational_level','Q5':'Work_profession','Q10':'Salary','Time from Start to Finish (seconds)':'Frequency'})
current_degree_job_compensation['perc']=(current_degree_job_compensation['Frequency']*100)/(current_degree_job_compensation['Frequency'].sum())
cur = current_degree_job_compensation[current_degree_job_compensation.Work_profession == 'Data Scientist']
c = cur[(cur.Salary == '150,000-199,999')|(cur.Salary == '20,000-24,999')|(cur.Salary == '25,000-29,999')|(cur.Salary == '125,000-149,000')|(cur.Salary == '300,000-500,000')|(cur.Salary == '30,000-39,999')|(cur.Salary == '$0-999')|(cur.Salary == '1,000-1,999')|(cur.Salary == '2,000-2,999')]
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

fig, ax = plt.subplots(1, 1, figsize=(15,10))

plt.xticks(c.index, c.Educational_level.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Educational_level", y="perc",hue='Salary', data=c)

ax.set_title('Proportion of participants in different salary brackets working as a Data-Scientist in India',fontsize=12)

ax.set(xlabel='Work Profession', ylabel='Percentage of participants in the survey')

ax.legend(bbox_to_anchor=(1.1, 1.05))



current_degree_job_compensation2 = india2.groupby(['Q4','Q6','Q9']).count().reset_index().loc[:,'Q4':'Time from Start to Finish (seconds)'].rename(columns={'Q4':'Educational_level','Q6':'Work_profession','Q9':'Salary','Time from Start to Finish (seconds)':'Frequency'})
current_degree_job_compensation2['perc']=(current_degree_job_compensation2['Frequency']*100)/(current_degree_job_compensation2['Frequency'].sum())
cur = current_degree_job_compensation2[current_degree_job_compensation2.Work_profession == 'Data Scientist']
c = cur[(cur.Salary == '150,000-200,000')|(cur.Salary == '20-30,000')|(cur.Salary == '30-40,000')|(cur.Salary == '125-150,000')|(cur.Salary == '300,000-400,000')|(cur.Salary == '30-40,000')|(cur.Salary == '0-10,000')]
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

fig, ax = plt.subplots(1, 1, figsize=(15,10))

plt.xticks(c.index, c.Educational_level.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Educational_level", y="perc",hue='Salary', data=c)

ax.set_title('Proportion of participants in different salary brackets working as a Data-Scientist in India in 2018',fontsize=12)

ax.set(xlabel='Work Profession', ylabel='Percentage of participants in the survey')

ax.legend(bbox_to_anchor=(1.1, 1.05))



students = india[india.Q5 == 'Student'].reset_index(drop=True)
#platforms_students = students.groupby('Q13').count().reset_index().loc[:,'Q13':'Time from Start to Finish (seconds)'].rename(columns={'Q13':'Platforms','Time from Start to Finish (seconds)':'Frequency'})



platforms = ['Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12']

l = list()

for i in platforms:

    df = students.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Platforms'}, inplace = True)

    l.append(df)

platforms_students = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
platforms_students.sort_values(by=['Frequency'],ascending=False,inplace=True)
platforms_students['perc'] = (platforms_students['Frequency']*100)/(platforms_students['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(platforms_students.index, platforms_students.Platforms.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Platforms", y="perc",data=platforms_students)

ax.set_title('Proportion of Platforms used to study data science courses by students in India in 2019',fontsize=15)

ax.set(xlabel='Tools Used', ylabel='Percentage of Participants in the survey')

students = india2[india2.Q6 == 'Student'].reset_index(drop=True)
#platforms_students = students.groupby('Q13').count().reset_index().loc[:,'Q13':'Time from Start to Finish (seconds)'].rename(columns={'Q13':'Platforms','Time from Start to Finish (seconds)':'Frequency'})



platforms = ['Q36_Part_1','Q36_Part_2','Q36_Part_3','Q36_Part_4','Q36_Part_5','Q36_Part_6','Q36_Part_7','Q36_Part_8','Q36_Part_9','Q36_Part_10','Q36_Part_11','Q36_Part_12','Q36_Part_13']

l = list()

for i in platforms:

    df = students.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Platforms'}, inplace = True)

    l.append(df)

platforms_students = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
platforms_students.sort_values(by=['Frequency'],ascending=False,inplace=True)
platforms_students['perc'] = (platforms_students['Frequency']*100)/(platforms_students['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(platforms_students.index, platforms_students.Platforms.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Platforms", y="perc",data=platforms_students)

ax.set_title('Proportion of Platforms used to study data science courses by students in India in 2018',fontsize=15)

ax.set(xlabel='Tools Used', ylabel='Percentage of Participants in the survey')

students = india[india.Q5 == 'Student'].reset_index(drop=True)
#platforms_students = students.groupby('Q13').count().reset_index().loc[:,'Q13':'Time from Start to Finish (seconds)'].rename(columns={'Q13':'Platforms','Time from Start to Finish (seconds)':'Frequency'})



sources = ['Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']

l = list()

for i in sources:

    df = students.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Sources'}, inplace = True)

    l.append(df)

sources_students = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
sources_students.sort_values(by=['Frequency'],ascending=False,inplace=True)
sources_students['perc']=(sources_students['Frequency']*100)/sources_students['Frequency'].sum()
sources_students.reset_index(drop=True,inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(sources_students.index, sources_students.Sources.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Sources", y="perc",data=sources_students)

ax.set_title('Proportion of Fav Sources that report on data science courses by students in India in 2019',fontsize=15)

ax.set(xlabel='Sources', ylabel='Percentage of Participants in the survey')

students2 = india2[india2.Q6 == 'Student'].reset_index(drop=True)
#platforms_students = students.groupby('Q13').count().reset_index().loc[:,'Q13':'Time from Start to Finish (seconds)'].rename(columns={'Q13':'Platforms','Time from Start to Finish (seconds)':'Frequency'})



sources = ['Q38_Part_1','Q38_Part_2','Q38_Part_3','Q38_Part_4','Q38_Part_5','Q38_Part_6','Q38_Part_7','Q38_Part_8','Q38_Part_9','Q38_Part_10','Q38_Part_11','Q38_Part_12','Q38_Part_13','Q38_Part_14','Q38_Part_15','Q38_Part_16','Q38_Part_17','Q38_Part_18','Q38_Part_19','Q38_Part_20','Q38_Part_21','Q38_Part_22']

l = list()

for i in sources:

    df = students2.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Sources'}, inplace = True)

    l.append(df)

sources_students = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
sources_students.sort_values(by=['Frequency'],ascending=False,inplace=True)
sources_students['perc']=(sources_students['Frequency']*100)/sources_students['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(sources_students.index, sources_students.Sources.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Sources", y="perc",data=sources_students)

ax.set_title('Proportion of Fav Sources that report on data science courses by students in India in 2019',fontsize=15)

ax.set(xlabel='Sources', ylabel='Percentage of Participants in the survey')

#platforms_students = students.groupby('Q13').count().reset_index().loc[:,'Q13':'Time from Start to Finish (seconds)'].rename(columns={'Q13':'Platforms','Time from Start to Finish (seconds)':'Frequency'})



tools = ['Q14']

l = list()

for i in tools:

    df = students.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Tools'}, inplace = True)

    l.append(df)

tools_students = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
tools_students.sort_values(by=['Frequency'],ascending=False,inplace=True)
tools_students['perc'] = (tools_students['Frequency']*100)/(tools_students['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(tools_students.index, tools_students.Tools.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Tools", y="perc",data=tools_students)

ax.set_title('Tools used by students to analyze data in India in 2019',fontsize=15)

ax.set(xlabel='Sources', ylabel='Percentage of Participants in the survey')

#platforms_students = students.groupby('Q13').count().reset_index().loc[:,'Q13':'Time from Start to Finish (seconds)'].rename(columns={'Q13':'Platforms','Time from Start to Finish (seconds)':'Frequency'})



tools = ['Q12_MULTIPLE_CHOICE']

l = list()

for i in tools:

    df = students2.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Tools'}, inplace = True)

    l.append(df)

tools_students = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
tools_students.sort_values(by=['Frequency'],ascending=False,inplace=True)
tools_students['perc'] = (tools_students['Frequency']*100)/(tools_students['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(tools_students.index, tools_students.Tools.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Tools", y="perc",data=tools_students)

ax.set_title('Tools used by students to analyze data in India in 2018',fontsize=15)

ax.set(xlabel='Sources', ylabel='Percentage of Participants in the survey')

indian_data_scientists = india[india.Q5 == 'Data Scientist'].reset_index(drop=True)

tools_to_analyze = indian_data_scientists.groupby('Q14').count().reset_index().loc[:,'Q14':'Time from Start to Finish (seconds)'].rename(columns={'Q14':'Tools_Used_To_Analyze_Data_At_Workplace','Time from Start to Finish (seconds)':'Frequency'})
tools_to_analyze = tools_to_analyze[tools_to_analyze.Tools_Used_To_Analyze_Data_At_Workplace != 0]
tools_to_analyze.sort_values(by=['Frequency'],inplace=True, ascending=False)
tools_to_analyze['perc']=(tools_to_analyze['Frequency']*100)/tools_to_analyze['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(tools_to_analyze.index, tools_to_analyze.Tools_Used_To_Analyze_Data_At_Workplace.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Tools_Used_To_Analyze_Data_At_Workplace", y="perc",data=tools_to_analyze)

ax.set_title('Proportion of tools used to analyze data by Indian Data Scientists in 2019',fontsize=15)

ax.set(xlabel='Tools Used', ylabel='Percentage of Participants in the survey')

usa = req[req.Q3 == 'United States of America'].reset_index(drop=True)
us_data_scientists = usa[usa.Q5 == 'Data Scientist'].reset_index(drop=True)

tools_to_analyze = us_data_scientists.groupby('Q14').count().reset_index().loc[:,'Q14':'Time from Start to Finish (seconds)'].rename(columns={'Q14':'Tools_Used_To_Analyze_Data_At_Workplace','Time from Start to Finish (seconds)':'Frequency'})
tools_to_analyze = tools_to_analyze[tools_to_analyze.Tools_Used_To_Analyze_Data_At_Workplace != 0]
tools_to_analyze.sort_values(by=['Frequency'],inplace=True, ascending=False)
tools_to_analyze['perc'] = (tools_to_analyze['Frequency']*100)/tools_to_analyze['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(tools_to_analyze.index, tools_to_analyze.Tools_Used_To_Analyze_Data_At_Workplace.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Tools_Used_To_Analyze_Data_At_Workplace", y="perc",data=tools_to_analyze)

ax.set_title('Proportion of tools used to analyze data by US data scientists in 2019',fontsize=15)

ax.set(xlabel='Tools Used', ylabel='Percentage of Participants in the survey')

#Comparison of cloud based API's usage among participants from 2018- 2019

india_ds_2018 = india2[india2.Q6 == 'Data Scientist'].reset_index(drop=True)

tools_to_analyze = india_ds_2018.groupby('Q12_MULTIPLE_CHOICE').count().reset_index().loc[:,'Q12_MULTIPLE_CHOICE':'Time from Start to Finish (seconds)'].rename(columns={'Q12_MULTIPLE_CHOICE':'Tools_Used_To_Analyze_Data_At_Workplace','Time from Start to Finish (seconds)':'Frequency'})
tools_to_analyze = tools_to_analyze[tools_to_analyze.Tools_Used_To_Analyze_Data_At_Workplace != 0]
tools_to_analyze.sort_values(by=['Frequency'],inplace=True, ascending=False)
tools_to_analyze['perc'] = (tools_to_analyze['Frequency']*100)/tools_to_analyze['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(tools_to_analyze.index, tools_to_analyze.Tools_Used_To_Analyze_Data_At_Workplace.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Tools_Used_To_Analyze_Data_At_Workplace", y="perc",data=tools_to_analyze)

ax.set_title('Proportion of tools used to analyze data by India data scientists in 2018',fontsize=15)

ax.set(xlabel='Tools Used', ylabel='Percentage of Participants in the survey')

ide_used = ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']

l = list()

for i in ide_used:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'ide_used'}, inplace = True)

    l.append(df)

ide_used = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
ide_used = ide_used[ide_used.ide_used!=0].reset_index(drop=True)
ide_used.sort_values(by=['Frequency'],ascending=False,inplace=True)
ide_used['perc'] = (ide_used['Frequency']*100)/ide_used['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(ide_used.index, ide_used.ide_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="ide_used", y="perc",data=ide_used)

ax.set_title('Proportion of IDEs used to analyze data by Indian data scientists in 2019',fontsize=15)

ax.set(xlabel='IDEs Used', ylabel='Percentage of Participants in the survey')

ide_used = ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']

l = list()

for i in ide_used:

    df = us_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'ide_used'}, inplace = True)

    l.append(df)

ide_used = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
ide_used = ide_used[ide_used.ide_used!=0].reset_index(drop=True)
ide_used.sort_values(by=['Frequency'],ascending=False,inplace=True)
ide_used['perc'] = (ide_used['Frequency']*100)/ide_used['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(ide_used.index, ide_used.ide_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="ide_used", y="perc",data=ide_used)

ax.set_title('Proportion of IDEs used to analyze data by US data scientists in 2019',fontsize=15)

ax.set(xlabel='IDEs Used', ylabel='Percentage of Participants in the survey')

languages_used = ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']

l = list()

for i in languages_used:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'languages_used'}, inplace = True)

    l.append(df)

languages_used = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
languages_used = languages_used[languages_used.languages_used!=0].reset_index(drop=True)
languages_used.sort_values(by=['Frequency'],ascending=False,inplace=True)
languages_used['perc'] = (languages_used['Frequency']*100)/languages_used['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(languages_used.index, languages_used.languages_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="languages_used", y="perc",data=languages_used)

ax.set_title('Proportion of Programming Languages used to analyze data by Indian data scientists in 2019',fontsize=15)

ax.set(xlabel='Programming Languages Used', ylabel='Percentage of Participants in the survey')

languages_used = ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']

l = list()

for i in languages_used:

    df = us_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'languages_used'}, inplace = True)

    l.append(df)

languages_used = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
languages_used = languages_used[languages_used.languages_used!=0].reset_index(drop=True)
languages_used.sort_values(by=['Frequency'],ascending=False,inplace=True)
languages_used['perc'] = (languages_used['Frequency']*100)/languages_used['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(languages_used.index, languages_used.languages_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="languages_used", y="perc",data=languages_used)

ax.set_title('Proportion of Programming Languages used to analyze data by US data scientists in 2019',fontsize=15)

ax.set(xlabel='Programming Languages Used', ylabel='Percentage of Participants in the survey')

data_vis_used = ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']

l = list()

for i in data_vis_used:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'data_vis_used'}, inplace = True)

    l.append(df)

data_vis_used = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
data_vis_used.sort_values(by=['Frequency'],ascending=False,inplace=True)
data_vis_used['perc']= (data_vis_used['Frequency']*100)/data_vis_used['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(data_vis_used.index, data_vis_used.data_vis_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="data_vis_used", y="perc",data=data_vis_used)

ax.set_title('Proportion of Data Visualization Tools used to analyze data by Indian data scientists in 2019',fontsize=15)

ax.set(xlabel='Data Visualization Used', ylabel='Percentage of Participants in the survey')

data_vis_used = ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']

l = list()

for i in data_vis_used:

    df = us_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'data_vis_used'}, inplace = True)

    l.append(df)

data_vis_used = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
data_vis_used = data_vis_used[data_vis_used.data_vis_used!=0].reset_index(drop=True)
data_vis_used.sort_values(by=['Frequency'],ascending=False,inplace=True)
data_vis_used['perc']=(data_vis_used['Frequency']*100)/data_vis_used['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(data_vis_used.index, data_vis_used.data_vis_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="data_vis_used", y="perc",data=data_vis_used)

ax.set_title('Proportion of Data Visualization Tools used to analyze data by US data scientists in 2019',fontsize=15)

ax.set(xlabel='Data Visualization Used', ylabel='Percentage of Participants in the survey')

languages_used = ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']

l = list()

for i in languages_used:

    df = india.groupby(['Q5',i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'languages_used','Q5':'Profession'}, inplace = True)

    l.append(df)

languages_used = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
languages_used = languages_used[languages_used.languages_used!=0].reset_index(drop=True)
languages_used.sort_values(by=['Frequency'],ascending=False,inplace=True)
languages_used['perc']=(languages_used['Frequency']*100)/(languages_used['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(languages_used.index, languages_used.languages_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="languages_used", y="perc",hue='Profession',data=languages_used)

ax.set_title('Proportion of Data Visualization Tools used to analyze data by Indian Survey Participants of different Work profession in 2019',fontsize=15)

ax.set(xlabel='Data Visualization Used', ylabel='Percentage of Participants in the survey')

tpu_usage = indian_data_scientists.groupby('Q22').count().reset_index().loc[:,:'Time from Start to Finish (seconds)']
tpu_usage.rename(columns = {'Q22':'tpu_usage','Time from Start to Finish (seconds)':'Frequency'},inplace=True)
tpu_usage.sort_values(by=['Frequency'],ascending=False,inplace=True)
tpu_usage['perc']=(tpu_usage['Frequency']*100)/tpu_usage['Frequency'].sum()
#import seaborn as sns

#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1, 1, figsize=(15,5))



#plt.xticks(hardware_used.index, hardware_used.hardware_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

#sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

#ax = sns.barplot(x="hardware_used", y="perc",data=hardware_used)

#ax.set_title('Proportion of Specialized Hardware used by India data scientists in 2019',fontsize=15)

#ax.set(xlabel='Specialized Hardware Used', ylabel='Percentage of Participants in the survey')



#import pandas.plotting 

#from pandas.plotting import table

#plt.figure(figsize=(16,8))

# plot chart

#ax1 = plt.subplot(121, aspect='equal')

#tpu_usage.plot(kind='pie', y = 'perc', ax=ax1, autopct='%1.1f%%', 

 #startangle=50, shadow=False, labels=tpu_usage['tpu_usage'], legend = False, fontsize=7)

#ax1.set_title('Proportion of TPU Usage By Data Scientists taking the survey in India')



#ax2 = plt.subplot(122)

#plt.axis('off')

##tbl = table(ax2, tpu_usage, loc='center')

#tbl.auto_set_font_size(False)

#tbl.set_fontsize(10)

#plt.show()



labels = tpu_usage['tpu_usage']

values = tpu_usage['perc']



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(

    title="Proportion of TPU Usage By Data Scientists taking the survey in India",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)

tpu_usage = us_data_scientists.groupby('Q22').count().reset_index().loc[:,:'Time from Start to Finish (seconds)']
tpu_usage.rename(columns = {'Q22':'Number_of_times_TPU_is_used','Time from Start to Finish (seconds)':'Frequency'},inplace=True)
tpu_usage.sort_values(by=['Frequency'],ascending=False,inplace=True)
tpu_usage['perc']=(tpu_usage['Frequency']*100)/tpu_usage['Frequency'].sum()
#import seaborn as sns

#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1, 1, figsize=(15,5))



#plt.xticks(hardware_used.index, hardware_used.hardware_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

#sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

#ax = sns.barplot(x="hardware_used", y="perc",data=hardware_used)

#ax.set_title('Proportion of Specialized Hardware used by India data scientists in 2019',fontsize=15)

#ax.set(xlabel='Specialized Hardware Used', ylabel='Percentage of Participants in the survey')



#import pandas.plotting 

#from pandas.plotting import table

#plt.figure(figsize=(16,8))

# plot chart

#ax1 = plt.subplot(121, aspect='equal')

#tpu_usage.plot(kind='pie', y = 'perc', ax=ax1, autopct='%1.1f%%', 

 #startangle=50, shadow=False, labels=tpu_usage['Number_of_times_TPU_is_used'], legend = False, fontsize=7)

#ax1.set_title('Proportion of TPU Usage By Data Scientists taking the survey in US')



#ax2 = plt.subplot(122)

#plt.axis('off')

#tbl = table(ax2, tpu_usage, loc='center')

#tbl.auto_set_font_size(False)

#tbl.set_fontsize(10)

#plt.show()

#import plotly.graph_objects as go



labels = tpu_usage['Number_of_times_TPU_is_used']

values = tpu_usage['perc']



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(

    title="Proportion of TPU Usage By Data Scientists taking the survey in US",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)

alg_ml = ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']

l = list()

for i in alg_ml:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'alg_ml'}, inplace = True)

    l.append(df)

alg_ml = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
alg_ml.sort_values(by=['Frequency'],ascending=False,inplace=True)
alg_ml['perc']=(alg_ml['Frequency']*100)/alg_ml['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(alg_ml.index, alg_ml.alg_ml.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="alg_ml", y="perc",data=alg_ml)

ax.set_title('Proportion of ML algorithms used by India data scientists in 2019',fontsize=15)

ax.set(xlabel='ML algorithms', ylabel='Percentage of Participants in the survey')

alg_ml = ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']

l = list()

for i in alg_ml:

    df = us_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'alg_ml'}, inplace = True)

    l.append(df)

alg_ml = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
alg_ml.sort_values(by=['Frequency'],ascending=False,inplace=True)
alg_ml['perc']=(alg_ml['Frequency']*100)/alg_ml['Frequency'].sum()
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(alg_ml.index, alg_ml.alg_ml.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="alg_ml", y="perc",data=alg_ml)

ax.set_title('Proportion of ML algorithms used by US data scientists in 2019',fontsize=15)

ax.set(xlabel='ML algorithms', ylabel='Percentage of Participants in the survey')

nlp = ['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']

l = list()

for i in nlp:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'nlp'}, inplace = True)

    l.append(df)

nlp = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
nlp.sort_values(by=['Frequency'],ascending=False,inplace=True)


nlp['perc']=(nlp['Frequency']*100)/nlp['Frequency'].sum()
#import seaborn as sns

#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1, 1, figsize=(15,5))



#plt.xticks(hardware_used.index, hardware_used.hardware_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

#sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

#ax = sns.barplot(x="hardware_used", y="perc",data=hardware_used)

#ax.set_title('Proportion of Specialized Hardware used by India data scientists in 2019',fontsize=15)

#ax.set(xlabel='Specialized Hardware Used', ylabel='Percentage of Participants in the survey')



import pandas.plotting 

from pandas.plotting import table

plt.figure(figsize=(16,8))

# plot chart

ax1 = plt.subplot(121, aspect='equal')

nlp.plot(kind='pie', y = 'perc', ax=ax1, autopct='%1.1f%%', 

 startangle=50, shadow=False, labels=nlp['nlp'], legend = False, fontsize=10)

ax1.set_title('Proportion of NLP techniques used By Data Scientists taking the survey in India')





nlp = ['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']

l = list()

for i in nlp:

    df = us_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'NLP_Techniques'}, inplace = True)

    l.append(df)

nlp = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
nlp.sort_values(by=['Frequency'],ascending=False,inplace=True)


nlp['perc']=(nlp['Frequency']*100)/nlp['Frequency'].sum()
#import seaborn as sns

#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1, 1, figsize=(15,5))



#plt.xticks(hardware_used.index, hardware_used.hardware_used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

#sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

#ax = sns.barplot(x="hardware_used", y="perc",data=hardware_used)

#ax.set_title('Proportion of Specialized Hardware used by India data scientists in 2019',fontsize=15)

#ax.set(xlabel='Specialized Hardware Used', ylabel='Percentage of Participants in the survey')



import pandas.plotting 

from pandas.plotting import table

plt.figure(figsize=(16,8))

# plot chart

ax1 = plt.subplot(121, aspect='equal')

nlp.plot(kind='pie', y = 'perc', ax=ax1, autopct='%1.1f%%', 

 startangle=50, shadow=False, labels=nlp['NLP_Techniques'], legend = False, fontsize=10)

ax1.set_title('Proportion of NLP techniques used By Data Scientists taking the survey in US')





framework = ['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']

l = list()

for i in framework:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'ML_framework_tools'}, inplace = True)

    l.append(df)

framework = pd.concat(l).reset_index(drop=True)



#rawn = role_at_work_num.reset_index(drop=True)
framework.sort_values(by=['Frequency'],inplace=True,ascending=False)
framework['perc'] = ((framework['Frequency']*100)/framework['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(framework.index, framework.ML_framework_tools.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="ML_framework_tools", y="perc",data=framework)

ax.set_title('Proportion of Machine Learning Framework tools used by India data scientists in 2019',fontsize=15)

ax.set(xlabel='Machine Learning Tools ', ylabel='perc')
framework = ['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']

l = list()

for i in framework:

    df = us_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'ML_framework_tools'}, inplace = True)

    l.append(df)

framework = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
framework.sort_values(by=['Frequency'],inplace=True,ascending=False)
framework['perc'] = ((framework['Frequency']*100)/framework['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(framework.index, framework.ML_framework_tools.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="ML_framework_tools", y="perc",data=framework)

ax.set_title('Proportion of Machine Learning Framework tools used by US data scientists in 2019',fontsize=15)

ax.set(xlabel='Machine Learning Tools ', ylabel='perc')
cloud_platforms = ['Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8','Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12']

l = list()

for i in cloud_platforms:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Cloud_Platforms_Used'}, inplace = True)

    l.append(df)

cloud = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
cloud.sort_values(by=['Frequency'],inplace=True,ascending=False)
cloud['perc'] = ((cloud['Frequency']*100)/cloud['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(cloud.index, cloud.Cloud_Platforms_Used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Cloud_Platforms_Used", y="perc",data=cloud)

ax.set_title('Proportion of Cloud Platforms used by Indian data scientists in 2019',fontsize=15)

ax.set(xlabel='Cloud Platforms', ylabel='perc')
cloud_platforms = ['Q15_Part_1','Q15_Part_2','Q15_Part_3','Q15_Part_4','Q15_Part_5','Q15_Part_6','Q15_Part_7']

l = list()

for i in cloud_platforms:

    df = india_ds_2018.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Cloud_Platforms_Used'}, inplace = True)

    l.append(df)

cloud = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
cloud.sort_values(by=['Frequency'],inplace=True,ascending=False)
cloud['perc'] = ((cloud['Frequency']*100)/cloud['Frequency'].sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(15,5))



plt.xticks(cloud.index, cloud.Cloud_Platforms_Used.str.upper(), rotation=60, horizontalalignment='right', fontsize=10)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Cloud_Platforms_Used", y="perc",data=cloud)

ax.set_title('Proportion of Cloud Platforms used by Indian data scientists in 2018',fontsize=15)

ax.set(xlabel='Cloud Platforms', ylabel='perc')
big_data = ['Q31_Part_1','Q31_Part_2','Q31_Part_3','Q31_Part_4','Q31_Part_5','Q31_Part_6','Q31_Part_7','Q31_Part_8','Q31_Part_9','Q31_Part_10','Q31_Part_11','Q31_Part_12']

l = list()

for i in big_data:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Big_Data_Products_Used_in_2019'}, inplace = True)

    l.append(df)

big_data = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
big_data.sort_values(by=['Frequency'],inplace=True,ascending=False)
big_data['perc'] = ((big_data['Frequency']*100)/big_data['Frequency'].sum())
import matplotlib.pyplot as plot

ax = big_data.plot.barh(x='Big_Data_Products_Used_in_2019', y='perc', title="Usage of big data products by data scientists in India in 2019");





# Despine

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_visible(False)



# Switch off ticks

ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")



# Draw vertical axis lines

vals = ax.get_xticks()

for tick in vals:

    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, zorder=1)



# Set x-axis label

ax.set_xlabel("Big Data products used", labelpad=20, weight='bold', size=12)



# Set y-axis label

ax.set_ylabel("Proportion of people", labelpad=20, weight='bold', size=12)



# Format y-axis label

#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

plot.show(block=True)
big_data2 = ['Q30_Part_1','Q30_Part_2','Q30_Part_3','Q30_Part_4','Q30_Part_5','Q30_Part_6','Q30_Part_7','Q30_Part_8','Q30_Part_9','Q30_Part_10','Q30_Part_11','Q30_Part_12','Q30_Part_13','Q30_Part_14','Q30_Part_15','Q30_Part_16','Q30_Part_17','Q30_Part_18','Q30_Part_19','Q30_Part_20','Q30_Part_21','Q30_Part_22','Q30_Part_23','Q30_Part_24']

l = list()

for i in big_data2:

    df = india_ds_2018.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Big_Data_Products_Used_in_2018'}, inplace = True)

    l.append(df)

big_data2 = pd.concat(l).reset_index(drop=True)
big_data2.sort_values(by=['Frequency'],inplace=True,ascending=False)
big_data2['perc'] = ((big_data2['Frequency']*100)/big_data2['Frequency'].sum())
import matplotlib.pyplot as plot

ax = big_data2.plot.barh(x='Big_Data_Products_Used_in_2018', y='perc', title="Usage of big data products by data scientists in India in 2018");





# Despine

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_visible(False)



# Switch off ticks

ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")



# Draw vertical axis lines

vals = ax.get_xticks()

for tick in vals:

    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, zorder=1)



# Set x-axis label

ax.set_xlabel("Big Data products used", labelpad=20, weight='bold', size=12)



# Set y-axis label

ax.set_ylabel("Proportion of people", labelpad=20, weight='bold', size=12)



# Format y-axis label

#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

plot.show(block=True)
#Q34, 12 Parts

rel_dat = ['Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8','Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12','Q29_Part_13','Q29_Part_14','Q29_Part_15','Q29_Part_16','Q29_Part_17','Q29_Part_18','Q29_Part_19','Q29_Part_20','Q29_Part_21','Q29_Part_22','Q29_Part_23','Q29_Part_24','Q29_Part_7','Q29_Part_8','Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12','Q29_Part_13','Q29_Part_14','Q29_Part_15','Q29_Part_16','Q29_Part_17','Q29_Part_18','Q29_Part_19','Q29_Part_20','Q29_Part_21','Q29_Part_22','Q29_Part_23','Q29_Part_24','Q29_Part_25','Q29_Part_26','Q29_Part_27','Q29_Part_28']

l = list()

for i in rel_dat:

    df = india_ds_2018.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Relational_Database_Used_in_2018'}, inplace = True)

    l.append(df)

rel_dat = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
rel_dat.sort_values(by=['Frequency'],inplace=True,ascending=False)

rel_dat = rel_dat.reset_index(drop=True)
rel_dat['perc'] = ((rel_dat['Frequency']*100)/rel_dat['Frequency'].sum())
labels = rel_dat['Relational_Database_Used_in_2018']

values = rel_dat['perc']



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(

    title=" 2018 Relational Database Tools used By Data Scientists in India",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)

#Q34, 12 Parts

rel_dat = ['Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8','Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12']

l = list()

for i in rel_dat:

    df = indian_data_scientists.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Relational_Database_Used_in_2019'}, inplace = True)

    l.append(df)

rel_dat = pd.concat(l).reset_index(drop=True)

#rawn = role_at_work_num.reset_index(drop=True)
rel_dat.sort_values(by=['Frequency'],inplace=True,ascending=False)

rel_dat['perc'] = ((rel_dat['Frequency']*100)/rel_dat['Frequency'].sum())
labels = rel_dat['Relational_Database_Used_in_2019']

values = rel_dat['perc']



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(

    title="2019 Relational Database Tools used By Data Scientists in India",

    font=dict(

        family="Courier New, monospace",

        size=13,

        color="#7f7f7f"

    )

)
