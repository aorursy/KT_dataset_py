import numpy as np # linear algebra
import pandas as pd # data processing

#plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

#color scales
import colorlover as cl

import os
print(os.listdir("../input"))

#load multiple choice data into variable
multiple_choice = pd.read_csv("../input/multipleChoiceResponses.csv")

#dictionary used to shorten answers for major
majors_dict = {"Computer science (software engineering, etc.)":"Computer Sci & Eng",
               "Engineering (non-computer focused)":"Engineering",
               "Mathematics or statistics":"Mathematics",
               "A business discipline (accounting, economics, finance, etc.)":"Business", 
               "Physics or astronomy":"Physics", 
               "Information technology, networking, or system administration":"Info Tech", 
               "Medical or life sciences (biology, chemistry, medicine, etc.)": "Life Sciences", 
               "Social sciences (anthropology, psychology, sociology, etc.)": "Social Sciences", 
               "Humanities (history, literature, philosophy, etc.)": "Humanities", 
               "Environmental science or geology": "Earth Sciences", 
               "Fine arts or performing arts" : "Arts",
               "I never declared a major":"Undeclared"}

#dictionary used to shorten answers for degree
degree_dict = {"Some college/university study without earning a bachelor’s degree":"Some college/university",
               "No formal education past high school":"High school"}
#dictionary used to shorten answers for degree
industry_dict = {"I am a student":"Student"}

#multiple_choice is filtered by reported degrees
#columns were renamed to make code easier to use/read
#columns that were used in analysis were selected
#dictionaries listed above were used to shorten survey answers
education = (multiple_choice[(multiple_choice.Q4.notnull())
                             &(multiple_choice.Q4!="I prefer not to answer")]     
          .rename(columns={'Q2':'age',
                           'Q3':'country',
                           'Q4':'degree',
                           'Q5':'major',
                           'Q6':'job',
                           'Q7':'industry',
                           'Q8':'experience',
                           'Q9':'salary',
                           'Q12_MULTIPLE_CHOICE':'software',
                           'Q17':'language'})
          .replace({'major': majors_dict,'degree':degree_dict,'industry':industry_dict})
          .loc[:,['age','country','degree','major','job','industry',
                  'experience','salary','software','language']])[1:]

#value counts for each of the following attributes were taken and used
majors = education.major.value_counts()
degrees = education.degree.value_counts()
age = education.age.value_counts()
jobs = education.job.value_counts()
industries = education.industry.value_counts()
experience = education.experience.value_counts()
softwares = education.software.value_counts()
languages = education.language.value_counts()
salaries = education.salary.value_counts()

#plotly pie graph chart code for degrees
#all pie charts follow a similar if not exact format
degrees_layout = go.Layout(title= 'Degrees',font=dict(size=15))
data = [go.Pie(labels=degrees.index,
               values=degrees.values,
               marker=dict(colors=["crimson","goldenrod","firebrick",
                                   "darkkhaki","gray","dimgray"]))]
figure = go.Figure(data=data,layout=degrees_layout)
iplot(figure)

#plotly pie graph chart code for majors
majors_layout = go.Layout(title= 'Majors', font=dict(size=15))
data = [go.Pie(labels=majors.index,values=majors.values)]
figure = go.Figure(data=data,layout=majors_layout)

iplot(figure)
# Pre-defined functions were used in analysis for making stacked bar chart subplots
# Functions are used repeatedly for Distributions with Degree Distributions
# This is to keep the plotting code from looking messy
# Note: I could have used some for-loops within these functions

# Calculates the number of each degree for every element within a given attribute
# as a *percentage of the total count for the attribute*.
# Attributes include: age, major,job and industry
# Stacks each degree for every element within the given attribute
# As a result, stacked bar graph also shows the overall distribution of each attribute
def degree_percentage(attribute):
    phd = education[education.degree=='Doctoral degree'][attribute].value_counts().reindex(idx)/total*100
    ms = education[education.degree=='Master’s degree'][attribute].value_counts().reindex(idx)/total*100
    bs = education[education.degree=='Bachelor’s degree'][attribute].value_counts().reindex(idx)/total*100
    prof = education[education.degree=='Professional degree'][attribute].value_counts().reindex(idx)/total*100
    some = education[education.degree=='Some college/university'][attribute].value_counts().reindex(idx)/total*100
    high = education[education.degree=='High school'][attribute].value_counts().reindex(idx)/total*100
    
    phd_tr = go.Bar(y=phd.index,x=phd.values,name='Doctoral degree',marker=dict(color="firebrick"),orientation = 'h')
    ms_tr = go.Bar(y=ms.index,x=ms.values,name='Master’s degree',marker=dict(color="crimson"),orientation = 'h')
    bs_tr = go.Bar(y=bs.index,x=bs.values,name='Bachelor’s degree',marker=dict(color="goldenrod"),orientation = 'h')
    prof_tr = go.Bar(y=prof.index,x=prof.values,name='Professional degree',marker=dict(color="darkkhaki"),orientation = 'h')
    some_tr = go.Bar(y=some.index,x=some.values,name='Some college/university',marker=dict(color="gray"),orientation = 'h')
    high_tr = go.Bar(y=high.index,x=high.values,name='High school',marker=dict(color="dimgray"),orientation = 'h')
    
    data = [high_tr,some_tr,prof_tr,bs_tr,ms_tr,phd_tr]
    return data

# Performs same thing as before but as a *percentage of each element*.
# As a result, stacked bar graph shows the degree distribution for each element
def normalized_degree_percentage(attribute):
    norm_phd = (education[education.degree=='Doctoral degree'][attribute].value_counts().reindex(idx)/education[attribute].value_counts()*100)
    norm_ms = (education[education.degree=='Master’s degree'][attribute].value_counts().reindex(idx)/education[attribute].value_counts()*100)
    norm_bs = (education[education.degree=='Bachelor’s degree'][attribute].value_counts().reindex(idx)/education[attribute].value_counts()*100)
    norm_prof = (education[education.degree=='Professional degree'][attribute].value_counts().reindex(idx)/education[attribute].value_counts()*100)
    norm_some = (education[education.degree=='Some college/university'][attribute].value_counts().reindex(idx)/education[attribute].value_counts()*100)
    norm_high = (education[education.degree=='High school'][attribute].value_counts().reindex(idx)/education[attribute].value_counts()*100)
    
    phd_ntr = go.Bar(y=norm_phd.index,x=norm_phd.values,name='Doctoral degree',showlegend=False,marker=dict(color="firebrick"),orientation = 'h')
    ms_ntr = go.Bar(y=norm_ms.index,x=norm_ms.values,name='Master’s degree',showlegend=False,marker=dict(color="crimson"),orientation = 'h')
    bs_ntr = go.Bar(y=norm_bs.index,x=norm_bs.values,name='Bachelor’s degree',showlegend=False,marker=dict(color="goldenrod"),orientation = 'h')
    prof_ntr = go.Bar(y=norm_prof.index,x=norm_prof.values,name='Professional degree',showlegend=False,marker=dict(color="darkkhaki"),orientation = 'h')
    some_ntr = go.Bar(y=norm_some.index,x=norm_some.values,name='Some college/university',showlegend=False,marker=dict(color="gray"),orientation = 'h')
    high_ntr = go.Bar(y=norm_high.index,x=norm_high.values,name='High school',showlegend=False,marker=dict(color="dimgray"),orientation = 'h')

    norm_data = [high_ntr,some_ntr,prof_ntr,bs_ntr,ms_ntr,phd_ntr]
    return norm_data


# Appends degree percentage and normalized degree percentage
# to the first and second subplots respectively.
def append_degrees(figure,data,norm_data):
    figure.append_trace(data[0],1,1)
    figure.append_trace(data[1],1,1)
    figure.append_trace(data[2],1,1)
    figure.append_trace(data[3],1,1)
    figure.append_trace(data[4],1,1)
    figure.append_trace(data[5],1,1)
    
    figure.append_trace(norm_data[0],1,2)
    figure.append_trace(norm_data[1],1,2)
    figure.append_trace(norm_data[2],1,2)
    figure.append_trace(norm_data[3],1,2)
    figure.append_trace(norm_data[4],1,2)
    figure.append_trace(norm_data[5],1,2)
    return figure
# Used idx to manually sort age group strings into numerical order 
idx = ['80+','70-79','60-69','55-59','50-54','45-49','40-44','35-39','30-34','25-29','22-24','18-21']
# Defined total as sum of ages
total = age.sum()

# Stacked bar sublots for age 
# All stacked bar subplots follow same format

# Using pre-defined functions degree_percentage and normalized_degree_percentage
data = degree_percentage('age')
norm_data = normalized_degree_percentage('age')

figure = tools.make_subplots(print_grid=False,rows=1, cols=2,shared_yaxes=True)

figure['layout'].update(title='Age Distribution with Degree Distributions',font=dict(size=15), barmode='stack',
                        height=600,margin=dict(t=100,l=100),yaxis=dict(title='Age [Years]'))

figure['layout']['xaxis1'].update(title='% of Total',dtick=5,tickangle=45,ticks='outside')
figure['layout']['xaxis2'].update(title='% of Age Group',dtick=20,tickangle=45,ticks='outside')

append_degrees(figure,data,norm_data)

iplot(figure)
# Defined idx according to value_counts of major 
idx = majors.index[::-1]

# Defined total as sum of majors
total = majors.sum()

# Stacked bar sublots for major 
data = degree_percentage('major')
norm_data = normalized_degree_percentage('major')

figure = tools.make_subplots(print_grid=False,rows=1, cols=2, shared_yaxes=True)
figure['layout'].update(title='Major Distribution with Degree Distributions',font=dict(size=15), barmode='stack',
                        xaxis=dict(tickangle=45),height=700,margin=dict(t=100,b=160,l=180))

figure['layout']['xaxis1'].update(title='% of Total')
figure['layout']['xaxis2'].update(title='% of Major')

append_degrees(figure,data,norm_data)

iplot(figure)

# Used dictionary to combined reported majors 
combined_dict = {"Life Sciences":"Non-Computer Science","Social Sciences":"Non-Computer Science",
                 "Earth Sciences":"Non-Computer Science","Physics":"Non-Computer Science",
                 "Humanities":"Other","Arts":"Other"}

# Value counts taken of combined majors
combined_majors = (education
                  .replace({'major': combined_dict})
                  .major
                  .value_counts())

# Pie chart of combined majors
combined_majors_layout = go.Layout(title= 'Majors (Combined)', font=dict(size=15))

data =[go.Pie(labels=combined_majors.index,values=combined_majors.values)]

figure = go.Figure(data=data,layout=combined_majors_layout)

iplot(figure)

# Pie chart of jobs
jobs_layout = go.Layout(title= 'Jobs', font=dict(size=15))
data = [go.Pie(labels=jobs.index,values=jobs.values)]
figure = go.Figure(data=data,layout=jobs_layout)

iplot(figure)
# Pie chart of industries
industries_layout = go.Layout(title= 'Industries', font=dict(size=15))
data = [go.Pie(labels=industries.index,values=industries.values)]
figure = go.Figure(data=data,layout=industries_layout)

iplot(figure)
# Defined idx according to value_counts of jobs 
idx = jobs.index[::-1]
# Defined total as sum of jobs 
total = jobs.sum()
# Same as before, stacked bar subplots for jobs
data = degree_percentage('job')
norm_data = normalized_degree_percentage('job')

figure = tools.make_subplots(print_grid=False,rows=1,cols=2,shared_yaxes=True)

figure['layout'].update(title='Job Distribution with Degree Distributions',font=dict(size=15),
                        barmode='stack',xaxis=dict(tickangle=45),height=700,
                        margin=dict(t=100,b=160,l=200))

figure['layout']['xaxis1'].update(title='% of Total')
figure['layout']['xaxis2'].update(title='% of Occupation')

append_degrees(figure,data,norm_data)

bar = iplot(figure)
# Defined idx according to value_counts of industry
idx = industries.index[::-1]
# Defined total as sum of industries 
total = industries.sum()
# Same as before, stacked bar subplots for industries
data = degree_percentage('industry')
norm_data = normalized_degree_percentage('industry')

figure = tools.make_subplots(print_grid=False,rows=1, cols=2, shared_yaxes=True)

figure['layout'].update(title='Industry Distribution with Degree Distributions',font=dict(size=15),
                        barmode='stack',xaxis=dict(tickangle=45),height=700,
                        margin=dict(t=100,b=160,l=300))

figure['layout']['xaxis1'].update(title='% of Total')
figure['layout']['xaxis2'].update(title='% of Industry')

append_degrees(figure,data,norm_data)

bar = iplot(figure)
# Used yrs_idx to sort years of experience
yrs_idx = ['0-1','1-2','2-3','3-4','4-5','5-10', '10-15','15-20','20-25','25-30','30 +']

# exp is the value count for years of experience as a percentage of the sum
exp = (experience/experience.sum()*100).reindex(yrs_idx)

# Simple bar chart for years of experience distribution
# All simple bar charts follow similar format
experience_layout = go.Layout(title= 'Years of Experience', font=dict(size=15),
                              xaxis=dict(tickangle=45,title='Age [Years]'),
                              yaxis=dict(title='Percent of Total'),margin=dict(b=100))

data=[go.Bar(x=exp.index,
             y=exp.values,
             marker=dict(color=exp.values,
                         colorscale='Viridis',
                         showscale=True,
                         reversescale=True))]

figure = go.Figure(data=data,layout=experience_layout)

iplot(figure)
# lang is the value count for years of experience as a percentage of the sum
lang = languages/languages.sum()*100

#Simple bar chart for languages 
languages_layout = go.Layout(title= 'Languages', font=dict(size=15),xaxis=dict(tickangle=45),
                             yaxis=dict(title='Percent of Total'),margin=dict(b=160))

data=[go.Bar(x=lang.index,
             y=lang.values,
             marker=dict(color=lang.values,
                         colorscale='Jet',
                         showscale=True,
                         reversescale=True))]

figure = go.Figure(data=data,layout=languages_layout)

iplot(figure)

# Used other_dict to group all languages other than Python and R
other_dict={'SQL':'Other',
            'Java':'Other',
            'C/C++':'Other',
            'C#/.NET':'Other',
            'Javascript/Typescript':'Other',
            'MATLAB':'Other',
            'SAS/STATA':'Other',
            'PHP':'Other',
            'Visual Basic/VBA':'Other',
            'Scala':'Other',
            'Bash':'Other',
            'Ruby':'Other',
            'Go':'Other',
            'Julia':'Other'}
# Selected multiple choice data for Python and R
pythonR = (education[(education.language=='Python')|(education.language=='R')]
           .copy())
# Selected multiple choice data for other languages and unspecified languages
other = (education[(education.language!='Python')&(education.language!='R')]
         .copy()
         .replace({'language': other_dict})
         .dropna())
# Concatenated both data sets so all languages are only Python, R or Other
pro = pd.concat([pythonR,other])

# Value counts of each attribute used for this section
pro_majors = pro.major.value_counts()
pro_jobs = pro.job.value_counts()
pro_ages = pro.age.value_counts()
pro_exp = pro.experience.value_counts()

# List of index names for sorting in plots
lang_idx = ['Python', 'Other', 'R']
yrs_idx = ['0-1','1-2','2-3','3-4','4-5','5-10', '10-15','15-20','20-25','25-30','30 +']
maj_idx = majors.index
job_idx = jobs.index
# Used similar color as bar plot before
color_idx = ["blue","orange","red"]
# Used for-loop to quickly calculate the normalized percentages of each language
# for every years of experience group.
# Stored values and made a trace for each language.
# Appended each trace to list called data
# Stacked bar chart of language by years of experience
# The other two bar charts in this section follow the same format.

count=0
data=[]

for i in lang_idx:
    langs = pro[pro.language==i].experience.value_counts().reindex(yrs_idx)
    norm = (langs/pro_exp.reindex(yrs_idx)*100)
    trace = go.Bar(x=norm.index,
                   y=norm.values,
                   name=i,
                   marker=dict(color=color_idx[count]))                      
    count=count+1
    data.append(trace)

lang_age_layout = go.Layout(title= 'Language Distribution by Years of Experience',font=dict(size=15),
                            xaxis=dict(tickangle=45,title='Experience [Years]'),
                            yaxis=dict(title='Percent of Year Range'),
                            barmode='stack')
figure = go.Figure(data=data,layout=lang_age_layout)

iplot(figure)
# Stacked bar chart of languages by major

count=0
data=[]

for i in lang_idx:
    langs = pro[pro.language==i].major.value_counts().reindex(maj_idx)
    norm = (langs/pro_majors.reindex(maj_idx)*100)
    trace = go.Bar(x=norm.index,
                   y=norm.values,
                   name=i,
                   marker=dict(color=color_idx[count]))                      
    count=count+1
    data.append(trace)

lang_maj_layout = go.Layout(title= 'Language Distributions by Major',font=dict(size=15),
                            xaxis=dict(tickangle=45),yaxis=dict(title='Percent of Major'),
                            barmode='stack',margin=dict(b=160))

figure = go.Figure(data=data,layout=lang_maj_layout)

iplot(figure)
# Stacked bar chart of languages by job

count=0
data=[]

for i in lang_idx:
    langs = pro[pro.language==i].job.value_counts().reindex(job_idx)
    profession = (langs/pro_jobs.reindex(job_idx)*100)
    trace = go.Bar(x=profession.index,
                   y=profession.values,
                   name=i,
                   marker=dict(color=color_idx[count]))                      
    count=count+1
    data.append(trace)

lang_job_layout = go.Layout(title= 'Language Distributions by Job',font=dict(size=15),
                            xaxis=dict(tickangle=45),yaxis=dict(title='Percent of Job'),
                            barmode='stack',margin=dict(b=200))
figure = go.Figure(data=data,layout=lang_job_layout)
iplot(figure)
# Used yrs_dict to group years of experience into three groups
yrs_dict = {'0-1':'0-2','1-2':'0-2','2-3':'2-5','3-4':'2-5',
            '4-5':'2-5','5-10':'5+','10-15':'5+','15-20':'5+',
            '20-25':'5+','25-30':'5+','30 +':'5+'}
# Used deg_dict to group degrees into three groups
deg_dict = {'High school':'No College Degree',
            'Some college/university':'No College Degree',
            'Professional degree':'Undergraduate Degree',
            'Bachelor’s degree':'Undergraduate Degree',
            'Master’s degree':'Graduate Degree',
            'Doctoral degree':'Graduate Degree'}
sal_dict = {'0-10,000':'0-20,000',
            '10-20,000':'0-20,000',
            '20-30,000':'20-40,000',
            '30-40,000':'20-40,000',
            '40-50,000':'40-60,000',
            '50-60,000':'40-60,000',
            '60-70,000':'60-80,000',
            '70-80,000':'60-80,000',
            '80-90,000':'80-100,000',
            '90-100,000':'80-100,000',
            '200-250,000':'200,000+',
            '250-300,000':'200,000+',
            '300-400,000':'200,000+',
            '400-500,000':'200,000+',
            '500,000+':'200,000+'}

# reported_salaries is the selected data based on the criteria written 
# in the Markdown above
reported_salaries = (education[(education.salary!='I do not wish to disclose my approximate yearly compensation')
                                &(education.salary.notnull())
                                &(education.job!='Student')
                                &(education.job!='Not employed')
                                &(education.job.notnull())
                                &(education.industry!='Student')
                                &(education.country=='United States of America')]
                     .loc[:,['degree','experience','salary','job','industry']]
                     .replace({'experience': yrs_dict,'degree':deg_dict}))

reported_employment = (education[(education.job!='Student')
                                 &(education.job.notnull())
                                 &(education.industry!='Student')
                                 &(education.country=='United States of America')]
                       .loc[:,['degree','experience','salary','job','industry']])


# reported_salaries with index rename as Used
reported = (pd.Series([reported_salaries.salary.value_counts().sum()])
            .rename(index={0:'Used'}))
# Sum of all salaries in originally selected data
all_salaries_sum = (education.salary
                    .fillna('nada')
                    .value_counts()
                    .sum())
# unreported is difference between the sum of all salaries 
# and the sum of Used entries
unreported = (pd.Series([all_salaries_sum-reported_salaries.salary.value_counts().sum()])
              .rename(index={0:'Unused'}))

# Concatenated used and unused into single df
rep = pd.concat([reported,unreported])

# lists yearly incomes in numberical order 
sal_idx = ['0-10,000','10-20,000','20-30,000','30-40,000','40-50,000',
           '50-60,000','60-70,000','70-80,000','80-90,000','90-100,000',
           '100-125,000','125-150,000','150-200,000','200-250,000',
           '250-300,000','300-400,000','400-500,000','500,000+']

# lists years of experience in numerical order
yrs_idx = ['0-2','2-5','5+']
#lists degrees in chronological order
deg_idx = ['No College Degree','Undergraduate Degree','Graduate Degree']
#lists colors in order
color_idx = ["gray","goldenrod","firebrick"]

# Grouping yearly incomes into three groups 
grouped_salaries = (reported_salaries
                    .replace({'salary': sal_dict}))

# Value counts for grouped incomes
group_jobs = grouped_salaries.job.value_counts()
group_inds = grouped_salaries.industry.value_counts()

# Numerical order of grouped incomes
group_idx = ['0-20,000','20-40,000','40-60,000','60-80,000','80-100,000','100-125,000',
             '125-150,000','150-200,000','200,000+']

# Pie chat of Used and Unused data
rep_layout = go.Layout(title= 'Used & Unused Portions',font=dict(size=15))

data= [go.Pie(labels=rep.index,values=rep.values)]
figure = go.Figure(data=data,layout=rep_layout)

iplot(figure)
# Bar chart of yearly incomes selected by the before mentioned criteria
sal = ((reported_salaries.salary.value_counts()/reported_salaries.salary.value_counts().sum()*100)
       .reindex(sal_idx))

salaries_layout = go.Layout(title= 'Yearly Incomes',font=dict(size=15),
                            xaxis=dict(tickangle=45,title='Yearly Income [$]'),
                            yaxis=dict(title='Percent of Total'),margin=dict(b=200))

data=[go.Bar(x=sal.index,
             y=sal.values,
             marker=dict(color=sal.values,
                         colorscale='Viridis',
                         showscale=True,
                         reversescale=True),
             orientation='v')]


salaries_bar = iplot(go.Figure(data=data,layout=salaries_layout))
# Used for-loop to quickly calculate the distributions of yearly income
# Normalized by each years of experience group 
# Stored values and made a trace for each degree.
# Appended each trace to list called data
# Line chart of yearly income distributions for each years of experience group
# The other line charts in this section follow the same format.

count=0
data=[]

for i in yrs_idx:
    sals = (reported_salaries[reported_salaries.experience==i]
            .salary
            .value_counts()
            .reindex(sal_idx,fill_value=0))
    
    exp = sals/sals.sum()*100
    trace = go.Scatter(x=exp.index,
                       y=exp.values,
                       name=i,
                       mode='lines',
                       connectgaps=True,
                       line=dict(color=color_idx[count],
                                 width=8,
                                 shape='linear'))
                               
    count=count+1
    data.append(trace)

sal_exp_layout = go.Layout(title= 'Yearly Income Distributions for Years of Experience',
                           font=dict(size=15),margin=dict(b=160),
                           xaxis=dict(showgrid=False,tickangle=45,
                                      title=('Yearly Income [$]')),
                           yaxis=dict(title='Percent of Age Group'))

figure = go.Figure(data=data,layout=sal_exp_layout)

iplot(figure)
# Normalized by each degree group 
# Line chart of yearly income distributions for each degree group.

count=0
data=[]

for i in deg_idx:
    sals = (reported_salaries[reported_salaries.degree==i]
            .salary.value_counts()
            .reindex(sal_idx,fill_value=0))
    
    deg = sals/sals.sum()*100
    trace = go.Scatter(x=deg.index,
                       y=deg.values,
                       name=i,
                       mode='lines',
                       line=dict(color=color_idx[count],
                                 width=8,
                                 shape='linear'))
                               
    count=count+1
    data.append(trace)

sal_deg_layout = go.Layout(title= 'Yearly Income Distributions for Degrees',font=dict(size=15),
                           xaxis=dict(showgrid=False,tickangle=45,
                                      title='Yearly Income [$]'),
                           yaxis=dict(title='Percent of Degree'),
                           margin=dict(b=160))
figure = go.Figure(data=data,layout=sal_deg_layout)

iplot(figure)
#Simple bar chart of unemployment
degree_idx = ['High school','Some college/university','Professional degree',
              'Bachelor’s degree', 'Master’s degree', 'Doctoral degree']

total = reported_employment.degree.value_counts().reindex(degree_idx)
unemployment = reported_employment[reported_employment.job=='Not employed'].degree.value_counts().reindex(degree_idx)

norm = unemployment/total*100

trace = go.Bar(x=norm.index,
               y=norm.values,
               name='Unemployed')
data = [trace]

layout = go.Layout(title= 'Unemployment Rate by Degree',font=dict(size=15),
                            xaxis=dict(tickangle=45),
                            yaxis=dict(title='Percent of Year Range'),
                            barmode='stack',
                            margin=dict(b=160))

figure = go.Figure(data=data,layout=layout)

iplot(figure)
# Simialar to what was done in languages
# For incomes and jobs
color_idx = cl.scales['9']['div']['Spectral']

count=0
data=[]

for i in group_idx:
    sals = grouped_salaries[grouped_salaries.salary==i].job.value_counts().reindex(group_jobs.index)
    norm = (sals/group_jobs.reindex(group_jobs.index)*100)
    trace = go.Bar(x=norm.index,
                   y=norm.values,
                   name=i,
                   marker=dict(color=color_idx[count]))                      
    count=count+1
    data.append(trace)

layout = go.Layout(title= 'Yeary Income [$] Distribution by Job',font=dict(size=15),
                            xaxis=dict(tickangle=45),
                            yaxis=dict(title='Percent of Year Range'),
                            barmode='stack',
                            margin=dict(b=160))
figure = go.Figure(data=data,layout=layout)

iplot(figure)
# Simialar to what was done in languages
# For incomes and industries
color_idx =cl.scales['9']['div']['Spectral']

count=0
data=[]

for i in group_idx:
    sals = grouped_salaries[grouped_salaries.salary==i].industry.value_counts().reindex(group_inds.index)
    norm = (sals/group_inds.reindex(group_inds.index)*100)
    trace = go.Bar(x=norm.index,
                   y=norm.values,
                   name=i,
                   marker=dict(color=color_idx[count]))                      
    count=count+1
    data.append(trace)

lang_age_layout = go.Layout(title= 'Yeary Income [$] Distribution by Industry',font=dict(size=15),
                            xaxis=dict(tickangle=45,tickfont=dict(size=12)),
                            yaxis=dict(title='Percent of Year Range'),
                            barmode='stack',
                            margin=dict(b=200))
figure = go.Figure(data=data,layout=lang_age_layout)

iplot(figure)