# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('../input/so-survey-2017/survey_results_public.csv')
survey_sch = pd.read_csv('../input/so-survey-2017/survey_results_schema.csv')
survey_sch
df.head()
#Columns having more than 75% of missing value
print(df.columns[df.isnull().mean()>0.75].to_list())
print('\n The no. of comlumn with more than 75% of missing values',len(df.columns[df.isnull().mean()>0.75].to_list()))
edu_type = df['EducationTypes'].value_counts().reset_index()
edu_type.head()
#First of all let's clean the data
#Here we are no dealing with any null value as it will automatically not be considered

def clean_series(df,col_name):
    """
    inputs: 
    df: dataframe which you want to manipulate
    col_name: name of the column which you want to clean the data
    
    outputs:
    df2: Panda's dataframe with the unique element seperated and
         their count
    """
    temp = df[col_name]
    temp = temp.dropna().reset_index()
    temp = temp[col_name].str.split(';')
    emp_list=[]
    for i in range(len(temp)):
        emp_list += temp[i]

    for i in range(len(emp_list)):
        emp_list[i] = emp_list[i].strip()

    emp_set = list(set(emp_list))    
    emp_set

    c = Counter(emp_list)
    print(c.items())
    df2 = pd.DataFrame({'Methods':list(c.keys()),'Counts':list(c.values())})
    
    return df2
methods = clean_series(df,'EducationTypes').sort_values(by='Counts')
methods
plt.figure(figsize=(14,6))
sns.barplot(data = methods , y = 'Methods',x = 'Counts',)
self_type = df['SelfTaughtTypes']
self_type.head()
#Let's use the function created before for seperating the categorical values and count the no. of occurence

method_self = clean_series(df,'SelfTaughtTypes').sort_values(by='Counts')
method_self['percentage'] = (method_self['Counts']/np.sum(method_self['Counts']))*100
method_self

plt.figure(figsize=(14,6))
sns.barplot(data = method_self , y = 'Methods',x = 'percentage',)
# The question asked to the developers during the survey
list(survey_sch[survey_sch.Column =='CousinEducation']['Question'])
#Let's have a look at what the participants say
study = df['CousinEducation'].value_counts().reset_index()
study.head()
#Let's apply the function defined ealier to count occurence of unique responses

method_study = clean_series(df,'CousinEducation').sort_values(by='Counts')
method_study['percentage'] = (method_study['Counts']/np.sum(method_study['Counts']))*100
method_study

# method_study = df['CousinEducation']
# method_study = method_study.dropna().reset_index()
# method_study = method_study['CousinEducation'].str.split(';')
# emp_list=[]
# for i in range(len(method_study)):
#     emp_list += method_study[i]
    
# for i in range(len(emp_list)):
#     emp_list[i] = emp_list[i].strip()

# emp_set = list(set(emp_list))    
# emp_set

# c = Counter(emp_list)
# print(c.items()))

plt.figure(figsize=(14,6))
sns.barplot(data = method_study , y = 'Methods',x = 'percentage',)
lang_work = df['HaveWorkedLanguage'].value_counts().reset_index()
lang_work.head()
lang_work = clean_series(df,'HaveWorkedLanguage').sort_values(by='Counts')
lang_work.columns = ['Language','CountsWork']
lang_work.head()
lang_want = df['WantWorkLanguage'].value_counts().reset_index()
lang_want.head()
lang_want = clean_series(df,'WantWorkLanguage').sort_values(by='Counts')
lang_want.columns = ['Language','CountsWant']
lang_want.head()
#First of all lets normalize the data to visualize and compare more clearly
#Defining a normalization function, to normalize the data

def norm(df,col_name):
    """
    input:
    df: The dataframe storing the data to be normalize
    col_name: Name of the column in which the data is stored
    
    output:
    df2: Dataframe with normalize data
    """
    df[col_name] = (df[col_name] - df[col_name].mean()) / (df[col_name].max() - df[col_name].min())
    pass
#Normalizing the HaveWorkLanguage Data
norm(lang_work,'CountsWork')
lang_work = lang_work.sort_values(by='Language').reset_index().drop('index',axis=1)
lang_work.head()
#Normalizing the WantWorkLanguage Data
norm(lang_want,'CountsWant')
lang_want = lang_want.sort_values(by='Language').reset_index().drop('index',axis=1)
lang_want.head()
language = pd.concat([lang_want,lang_work],axis=1)
language = language.loc[:,~language.columns.duplicated()]
language.head()
#Now let's create a barplot comparing both parameters
# lang_work.join(lang_want,on = 'Language')
language.columns = ['Language','IndustryPreference','ParticipantPreference']
plt.figure(figsize=(12,15))
language.plot(kind="bar",figsize=(18,9),x='Language')

#The function defined below labels the horizontal bar with their size/width
def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
country = df['Country'].value_counts().reset_index()
country.columns = ['Country','Participants']
plt.figure(figsize=(12,6))
bar = sns.barplot(data=country.head(10),y ='Country',x = 'Participants')
# for index, row in country.head(10).iterrows():
#     bar.text(row.tip,row.name, round(row.Country,2), color='black', ha="center")
show_values_on_bars(bar, "h", 0.3)
country.head()
emp_status = df['EmploymentStatus'].value_counts().reset_index()
emp_status.columns = ['EmploymentStatus','Participants']
plt.figure(figsize=(16,6))
plt.title("Current Employment Status Of Participants")
bar = sns.barplot(data = emp_status,x = 'Participants',y = 'EmploymentStatus')
show_values_on_bars(bar, "h", 0.3)
edu_status = df['FormalEducation'].value_counts().reset_index()
edu_status.columns = ['FormalEducation','Participants']
plt.figure(figsize=(16,6))
plt.title("Highest formal education degree of the developers")
bar = sns.barplot(data = edu_status,x = 'Participants',y = 'FormalEducation')
show_values_on_bars(bar, "h", 0.3)
career_sat = df['CareerSatisfaction'].value_counts().reset_index()
career_sat.columns = ['CareerSatisfaction','Participants']
career_sat = career_sat.sort_values(by='CareerSatisfaction').reset_index()
career_sat['per'] = (career_sat['Participants']/np.sum(career_sat['Participants']))*100
plt.figure(figsize=(16,6))
plt.title("Career Satisfaction Of the developers")
bar = sns.barplot(data = career_sat,y = 'Participants',x = 'CareerSatisfaction')
for index, row in career_sat.iterrows():
    bar.text(row.name,row.Participants, round(row.per,2), color='black', ha="center")
salary = df['Salary']
salary = salary.dropna()
salary = salary.reset_index()
salary
plt.figure(figsize=(12,5))
plt.title("Distribution of Salary over no. of participants")
sns.distplot(a=df['Salary'], kde=False)

job_sat = df['JobSatisfaction'].value_counts().reset_index()
job_sat.columns = ['JobSatisfaction','Participants']
job_sat = job_sat.sort_values(by='JobSatisfaction').reset_index()
job_sat['per'] = (job_sat['Participants']/np.sum(job_sat['Participants']))*100
plt.figure(figsize=(16,6))
plt.title("Current Job Satisfaction Of the developers")
bar = sns.barplot(data = job_sat,y = 'Participants',x = 'JobSatisfaction')

#This code is for labeling the bars with the percentage
for index, row in job_sat.iterrows():
    bar.text(row.name,row.Participants, round(row.per,2), color='black', ha="center")
plt.figure(figsize=(12,5))
sns.jointplot(x= df['JobSatisfaction'],y=df['Salary'],kind='kde')
job_seek = df['JobSeekingStatus'].value_counts().reset_index()
job_seek.columns = ['JobSeeking','Participants']
job_seek = job_seek.sort_values(by='JobSeeking').reset_index()
job_seek['per'] = (job_seek['Participants']/np.sum(job_seek['Participants']))*100
plt.figure(figsize=(16,6))
plt.title("Current Job Satisfaction Of the developers")
bar = sns.barplot(data = job_seek,x = 'Participants',y = 'JobSeeking')
