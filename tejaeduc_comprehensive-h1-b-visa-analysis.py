import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
init_notebook_mode(connected=True)
from plotly import tools
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../input/H-1B_Disclosure_Data_FY17.csv")
df.head()
df.VISA_CLASS.value_counts()
df.EMPLOYER_COUNTRY.value_counts()
df = df[df.VISA_CLASS == 'H-1B']
df= df[df.EMPLOYER_COUNTRY == 'UNITED STATES OF AMERICA']
df.apply(lambda x:len(x.unique()))
df.isnull().sum()[df.isnull().sum() > 0]
to_select = ['CASE_STATUS', 'EMPLOYMENT_START_DATE','EMPLOYER_NAME', 'EMPLOYER_STATE','JOB_TITLE', 'SOC_NAME','FULL_TIME_POSITION',
            'PREVAILING_WAGE','PW_UNIT_OF_PAY','WORKSITE_STATE']
df = df[to_select]
df.isnull().sum()[df.isnull().sum() > 0]
df = df[df['EMPLOYMENT_START_DATE'].notnull()]
df = df[df['JOB_TITLE'].notnull()]
df = df[df['SOC_NAME'].notnull()]
df = df[df['FULL_TIME_POSITION'].notnull()]
df = df[df['PW_UNIT_OF_PAY'].notnull()]
df = df[df['WORKSITE_STATE'].notnull()]
df = df[df['EMPLOYER_NAME'].notnull()]
df.isnull().sum()[df.isnull().sum() > 0]
df.head()
df['EMPLOYMENT_START_DATE'] = pd.to_datetime(df['EMPLOYMENT_START_DATE'])
df.groupby(['FULL_TIME_POSITION','PW_UNIT_OF_PAY']).describe()['PREVAILING_WAGE']
for i in df.index:   
        if df.loc[i,'PW_UNIT_OF_PAY'] == 'Month':
            df.loc[i,'PREVAILING_WAGE'] = df.loc[i,'PREVAILING_WAGE'] * 12
        if df.loc[i,'PW_UNIT_OF_PAY'] == 'Week':
            df.loc[i,'PREVAILING_WAGE'] = df.loc[i,'PREVAILING_WAGE'] * 48
        if df.loc[i,'PW_UNIT_OF_PAY'] == 'Bi-Weekly':
            df.loc[i,'PREVAILING_WAGE'] = df.loc[i,'PREVAILING_WAGE'] * 24
df.PW_UNIT_OF_PAY.replace(['Bi-Weekly','Month','Week'],['Year','Year','Year'], inplace=True)
df.groupby(['FULL_TIME_POSITION','PW_UNIT_OF_PAY']).describe()['PREVAILING_WAGE']
df['countvar'] = 1
dftop = df.groupby('EMPLOYER_NAME',as_index=False).count()
dftop = dftop.sort_values('countvar',ascending= False)[['EMPLOYER_NAME','countvar']][0:30]
t1 = go.Bar(x=dftop.EMPLOYER_NAME.values,y=dftop.countvar.values,name='top30')
layout = go.Layout(dict(title= "TOP EMPLOYERS SPONSORING",yaxis=dict(title="Num of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)
dftop1 = df.groupby(['EMPLOYER_NAME','CASE_STATUS'],as_index=False).count()
dftop1=dftop1[dftop1.EMPLOYER_NAME.isin(dftop.EMPLOYER_NAME)]
t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'CERTIFIED'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'CERTIFIED'].sort_values('countvar',ascending= False)['countvar'].values,name='CERTIFIED')
t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'CERTIFIED-WITHDRAWN'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'CERTIFIED-WITHDRAWN'].sort_values('countvar',ascending= False)['countvar'].values,name='CERTIFIED-WITHDRAWN')
t3 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'DENIED'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'DENIED'].sort_values('countvar',ascending= False)['countvar'].values,name='DENIED')
t4 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'WITHDRAWN'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'WITHDRAWN'].sort_values('countvar',ascending= False)['countvar'].values,name='WITHDRAWN')

data = [t1,t2,t3,t4]
layout = go.Layout(
    barmode='stack'
)

fig =go.Figure(data,layout)
iplot(fig)
dfempst = df.groupby('EMPLOYER_STATE',as_index=False).count()[['EMPLOYER_STATE','countvar']].sort_values('countvar',ascending=False)
t1 = go.Bar(x=dfempst.EMPLOYER_STATE.values,y=dfempst.countvar.values,name='Employerstate')
layout = go.Layout(dict(title= "NUMBER OF APPLICATIONS PER STATE",xaxis=dict(title="STATES"),yaxis=dict(title="Num of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)
data=[dict(
    type='choropleth',
    locations = dfempst.EMPLOYER_STATE,
    z = dfempst.countvar,
    locationmode = 'USA-states',marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Number of applications")
)]
layout= dict(title="2011-2018 H1B VISA APPLICATIONS ( EMPLOYER STATE)",geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict( data=data, layout=layout )
iplot(fig)
dfjob = df.groupby('JOB_TITLE',as_index=False).count()[['JOB_TITLE','countvar']].sort_values('countvar',ascending=False)[0:20]
t1 = go.Bar(x=dfjob.JOB_TITLE.values,y=dfjob.countvar.values,name='jobtitle')
layout = go.Layout(dict(title= "TOP 20 JOBS",yaxis=dict(title="Num of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)
df['year'] = df.EMPLOYMENT_START_DATE.apply(lambda x: x.year)
dfyear = df.groupby('year',as_index=False).count()[['year','countvar']]
t1 = go.Scatter(
    x=dfyear.year,
    y=dfyear.countvar
)
layout = go.Layout(dict(title= " NUMBER OF APPLICATIONS PER YEAR",xaxis=dict(title="YEARS"),yaxis=dict(title="Num of applications")))
data = [t1]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
t1 = go.Bar(x=df.groupby('CASE_STATUS').count().index,y=df.groupby('CASE_STATUS').count()['countvar'],name='CASESTATUSWISE')
data = [t1]
iplot(data)
t1 = go.Bar(x=df[df.FULL_TIME_POSITION == 'Y'].groupby('CASE_STATUS').count().index,y=df[df.FULL_TIME_POSITION == 'Y'].groupby('CASE_STATUS').count()['countvar'],name='FULL-TIME ')
t2 = go.Bar(x=df[df.FULL_TIME_POSITION == 'N'].groupby('CASE_STATUS').count().index,y=df[df.FULL_TIME_POSITION == 'N'].groupby('CASE_STATUS').count()['countvar'],name='PART-TIME ')
data = [t1,t2]
layout = go.Layout(barmode='stack')
fig = go.Figure(data =data,layout =layout)
iplot(fig)
df.PREVAILING_WAGE.describe()
df.PW_UNIT_OF_PAY.value_counts()
dum = df[(df.FULL_TIME_POSITION == 'Y') & (df.PW_UNIT_OF_PAY == 'Year')]
ind1 = dum[(dum.PREVAILING_WAGE > 270000) | (dum.PREVAILING_WAGE < 40000)].index
df = df.drop(ind1,axis=0)
dum = df[(df.FULL_TIME_POSITION == 'N') & (df.PW_UNIT_OF_PAY == 'Year')]
ind1 = dum[(dum.PREVAILING_WAGE > 150000) | (dum.PREVAILING_WAGE < 32000)].index
df = df.drop(ind1,axis=0)
dum = df[(df.PW_UNIT_OF_PAY == 'Hour')]
ind1 = dum[(dum.PREVAILING_WAGE > 110) | (dum.PREVAILING_WAGE < 15)].index
df = df.drop(ind1,axis=0)
k = df[(df.PW_UNIT_OF_PAY == 'Hour') & (df.FULL_TIME_POSITION == 'Y')].index
df.loc[k,'PREVAILING_WAGE'] = df.loc[k,'PREVAILING_WAGE'] * 1920
k = df[(df.PW_UNIT_OF_PAY == 'Hour') & (df.FULL_TIME_POSITION == 'N')].index
df.loc[k,'PREVAILING_WAGE'] = df.loc[k,'PREVAILING_WAGE'] * 1440
df=df.drop(['PW_UNIT_OF_PAY'],axis=1)
t1 = go.Scatter(
    x=df.groupby('year').mean().index,
    y=df.groupby('year').mean().PREVAILING_WAGE
)

layout = go.Layout(dict(title= " AVERAGE ANNUAL PAY vs YEAR",xaxis=dict(title="YEARS"),yaxis=dict(title="AVERAGE ANNUAL PAY")))
data = [t1]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
dum = df[["EMPLOYER_STATE","JOB_TITLE"]]
dum = dum.groupby(["EMPLOYER_STATE","JOB_TITLE"]).size().reset_index()
dum.columns = ['EMPLOYER_STATE', 'JOB_TITLE', "COUNT"]
dum = dum.groupby(['EMPLOYER_STATE', 'JOB_TITLE']).agg({'COUNT':sum})
dum = dum['COUNT'].groupby(level=0, group_keys=False)
dum = dum.apply(lambda x: x.sort_values(ascending=False).head(1))
dum = pd.DataFrame(dum).reset_index()
data=[dict(
    type='choropleth',
    locations = dum.EMPLOYER_STATE,
    z = dum.COUNT,
    locationmode = 'USA-states',
    text = dum.JOB_TITLE,
    marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Number of application")
)]
layout= dict(title="Top job title in the state",geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict( data=data, layout=layout )
iplot(fig)
dum = df.groupby('EMPLOYER_STATE',as_index=False).mean()[['EMPLOYER_STATE','PREVAILING_WAGE']]
data=[dict(
    type='choropleth',
    locations = dum.EMPLOYER_STATE,
    z = dum.PREVAILING_WAGE,
    locationmode = 'USA-states',
    marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Avg salary in USD")
)]
layout= dict(title="Average salaries per state",geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict( data=data, layout=layout )
iplot(fig)
df['OCCUPATION'] = np.nan
df['SOC_NAME'] = df['SOC_NAME'].str.lower()
df.OCCUPATION[df['SOC_NAME'].str.contains('computer','programmer')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('data scientist','data analyst')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('data engineer','data base')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('machine learning','artifical intelligence')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('spark','apache')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('hadoop','big data')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('sql','cyber')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('developer','full stack')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('fullstack','etl')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('data','network')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('software tester','cloud')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('information','informatica')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('jira','programmer')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('software','web developer')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('database')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('math','statistic')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('predictive model','stats')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('teacher','linguist')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('professor','Teach')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('school principal')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('medical','doctor')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('physician','dentist')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('Health','Physical Therapists')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('surgeon','nurse')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('psychiatr')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('chemist','physicist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('biology','scientist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('biologi','clinical research')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('public relation','manage')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('management','operation')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('chief','plan')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('executive')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('advertis','marketing')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('promotion','market research')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('business','business analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('business systems analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('accountant','finance')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('financial')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('engineer','architect')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('surveyor','carto')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('technician','drafter')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('information security','information tech')] = 'Architecture & Engineering'
df['OCCUPATION']= df.OCCUPATION.replace(np.nan, 'Others', regex=True)
df['SOC_NAME'] = df['SOC_NAME'].str.upper()

df.head()
df.OCCUPATION.value_counts()
dum = df.groupby('OCCUPATION',as_index = False).mean()[['OCCUPATION','PREVAILING_WAGE']]
t1 =go.Bar(x=dum.OCCUPATION,y=dum.PREVAILING_WAGE,name='wageperoccuaption')
layout = go.Layout(dict(title= " AVERAGE ANNUAL PAY vs OCCUPATION",yaxis=dict(title="AVERAGE ANNUAL PAY")))
data = [t1]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
dfcomp = df[df.OCCUPATION == 'Computer Occupations']
dum = dfcomp.groupby('EMPLOYER_STATE',as_index=False).mean()[['EMPLOYER_STATE','PREVAILING_WAGE']]
data=[dict(
    type='choropleth',
    locations = dum.EMPLOYER_STATE,
    z = dum.PREVAILING_WAGE,
    locationmode = 'USA-states',
    marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Avg salary in USD")
)]
layout= dict(title="Average salaries of TECH(IT) per state",geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict( data=data, layout=layout )
iplot(fig)
dum = df.groupby('SOC_NAME',as_index=False).mean()[['SOC_NAME','PREVAILING_WAGE']]
dum.sort_values('PREVAILING_WAGE',ascending= False).head(20)
dum = dfcomp.groupby('SOC_NAME',as_index=False).mean()[['SOC_NAME','PREVAILING_WAGE']]
dum.sort_values('PREVAILING_WAGE',ascending= False).head(20)
df['DS'] = np.nan
df.DS[df['JOB_TITLE'].str.contains('DATA SCIENTIST')] = 'DATA SCIENTIST'
df.DS[df['JOB_TITLE'].str.contains('DATA ANALYST')] = 'DATA ANALYST'
df.DS[df['JOB_TITLE'].str.contains('MACHINE LEARNING')] = 'MACHINE LEARNING'
df.DS[df['JOB_TITLE'].str.contains('BUSINESS ANALYST')] = 'BUSINESS ANALYST'
df.DS[df['JOB_TITLE'].str.contains('DEEP LEARNING')] = 'DEEP LEARNING'
df.DS[df['JOB_TITLE'].str.contains('ARTIFICIAL INTELLIGENCE')] = 'ARTIFICIAL INTELLIGENCE'
df.DS[df['JOB_TITLE'].str.contains('BIG DATA')] = 'BIG DATA'
df.DS[df['JOB_TITLE'].str.contains('HADOOP')] = 'HADOOP'
df.DS[df['JOB_TITLE'].str.contains('DATA ENGINEER')] = 'DATA ENGINEER'
df['DS']= df.DS.replace(np.nan, 'Others', regex=True)
df.DS.value_counts()
dum = df.groupby('DS',as_index=False).mean()[['DS','PREVAILING_WAGE']]
t1 =go.Bar(x=dum.DS,y=dum.PREVAILING_WAGE,name='DataScience')
data = [t1]
iplot(data)
dum = df.groupby(['year','DS']).count().reset_index()[['year','DS','countvar']]
data = []
for i in dum.DS.unique():
    if i != 'Others':
        data.append(go.Scatter(x = dum[dum.DS == i].year,y= dum[dum.DS == i].countvar,name=i))

layout = go.Layout(dict(title= "GROWTH IN DATA SCIENCE",xaxis=dict(title="YEARS"),yaxis=dict(title="Number of applications")))
        
fig = go.Figure(data,layout)    
iplot(fig)    

dum = df[["DS","EMPLOYER_STATE"]]
dum = dum.groupby(["DS","EMPLOYER_STATE"]).size().reset_index()
dum.columns = ["DS","EMPLOYER_STATE", "COUNT"]
dum = dum.groupby(["DS","EMPLOYER_STATE"]).agg({'COUNT':sum})
dum = dum['COUNT'].groupby(level=0, group_keys=False)
dum = dum.apply(lambda x: x.sort_values(ascending=False).head(1))
dum = pd.DataFrame(dum).reset_index()
dum[0:-1]
dum = df[["DS","EMPLOYER_NAME"]]
dum = dum.groupby(["DS","EMPLOYER_NAME"]).size().reset_index()
dum.columns = ["DS","EMPLOYER_NAME", "COUNT"]
dum = dum.groupby(["DS","EMPLOYER_NAME"]).agg({'COUNT':sum})
dum = dum['COUNT'].groupby(level=0, group_keys=False)
dum = dum.apply(lambda x: x.sort_values(ascending=False).head(1))
dum = pd.DataFrame(dum).reset_index()
dum[0:-1]
dfvadc = df[(df.EMPLOYER_STATE == 'VA') | (df.EMPLOYER_STATE == 'DC')]
dfvadc = dfvadc[dfvadc.DS != 'Others']
dum = dfvadc[["DS","EMPLOYER_NAME"]]
dum = dum.groupby(["DS","EMPLOYER_NAME"]).size().reset_index()
dum.columns = ["DS","EMPLOYER_NAME", "COUNT"]
dum = dum.groupby(["DS","EMPLOYER_NAME"]).agg({'COUNT':sum})
dum = dum['COUNT'].groupby(level=0, group_keys=False)
newdum = dum.apply(lambda x: x.sort_values(ascending=False).head(1))
newdum = pd.DataFrame(newdum).reset_index()
newdum[0:-1]
pd.DataFrame(dum.apply(lambda x: x.sort_values(ascending=False).head(15))).reset_index()['EMPLOYER_NAME']