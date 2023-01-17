import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import squarify as sq

import seaborn as sns

import plotly.graph_objects as go

import warnings

import plotly.express as px

warnings.filterwarnings("ignore")



%matplotlib inline
survey18 = pd.read_csv("/kaggle/input/stack-overflow-2018-developer-survey/survey_results_public.csv")
survey18.columns
country = survey18.groupby('Country').count()[['Respondent']].sort_values(by=['Respondent'], ascending=False).dropna().head(45)



fig = plt.gcf()

ax = fig.add_subplot()

fig.set_size_inches(20, 10)

norm = mpl.colors.Normalize(vmin=min(country.Respondent), vmax=max(country.Respondent))

colors = [mpl.cm.YlGnBu(norm(value)) for value in country.Respondent]





sq.plot(label=country.index,sizes=country.Respondent,color = colors, alpha=.8)

plt.axis('off')

plt.title('Countries from where overall respondents come from')

plt.show()
country.head()
fm = survey18.copy()

fm['count'] = fm.groupby('Country')['Country'].transform('count')

fm = fm[(fm['count']>200)]

fm = fm[(fm['Gender'] == 'Female') | (fm['Gender']=='Male')]

fm = fm.pivot_table(index='Country', columns='Gender', aggfunc='size', fill_value=0).reset_index()

sums = fm[['Female', 'Male']].sum(axis=1)

fm['FemaleRatio'] = fm['Female'] / sums

fm['MaleRatio'] = fm['Male'] / sums

fm = fm.sort_values(by='FemaleRatio', ascending=False)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 10))

ax = sns.barplot(x=fm.Country.head(30), y=fm.FemaleRatio.head(30), data=fm)

plt.xticks(rotation=45)

plt.title('Top 30 Countries ordered with better Female-to-Male Ratio')

fm = fm.sort_values(by='FemaleRatio', ascending=True)

sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 10))

ax = sns.barplot(x=fm.Country.head(30), y=fm.FemaleRatio.head(30), data=fm)

plt.xticks(rotation=45)

plt.title('Top 30 Countries ordered with worse Female-to-Male Ratio')
fm_age = survey18.copy()

fm_age = fm_age[(fm_age['Gender'] == 'Female') | (fm_age['Gender']=='Male')]

ax = sns.catplot(x="Age", col = 'Gender', data=fm_age, kind="count",height=9, aspect=1.1)

ax.set_axis_labels("Age", "Count")
fig,ax =  plt.subplots(figsize=(20,10))

ax = sns.countplot(x="Age", hue="Gender", data=fm_age, ax=ax )

plt.title('Age by Male vs Female - Stack Overflow Members')
fm_gen = survey18.copy()

s = fm_gen['Gender'].str.split(';').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1) # to line up with df's index

s.name = 'NewGender'

fm_gen = fm_gen.join(s)

fm_gen['gen_count'] = fm_gen.groupby('NewGender')['NewGender'].transform('count')

gen = fm_gen.filter(items=['NewGender', 'gen_count'])

gen = gen.drop_duplicates().dropna()

gen['perc']= gen['gen_count']/gen['gen_count'].sum()

gen
fig = go.Figure(go.Waterfall( name = "20", orientation = "v",

    x = gen.NewGender, textposition = "outside",y = gen.gen_count, connector = {"line":{"color":"rgb(63, 63, 63)"}},

))



fig.update_layout(title = "Gender - Stack Overflow Members",showlegend = True)

fig.show()
sx = survey18.copy()

s = sx['SexualOrientation'].str.split(';').apply(pd.Series, 1).stack()

#print(s)

s.index = s.index.droplevel(-1) 

s.name = 'NewSexualOrientation'

sx = sx.join(s)

sx['sx_count'] = sx.groupby('NewSexualOrientation')['NewSexualOrientation'].transform('count')

sxo = sx.filter(items=['NewSexualOrientation', 'sx_count'])

sxo = sxo.drop_duplicates().dropna()
fig1 = go.Figure(go.Waterfall( name = "20", orientation = "v",

    x = sxo.NewSexualOrientation, textposition = "outside",y = sxo.sx_count, connector = {"line":{"color":"rgb(63, 63, 63)"}},

))

fig1.update_layout(title = "Sexual Orientation - Stack Overflow Members",showlegend = True)

fig1.show()
fig, ax = plt.subplots(1,2, figsize=(20,10))

sns.countplot(x="Student", data=survey18, ax=ax[1]).set_title('Student vs Others') 

sns.countplot(y="Employment", data=survey18, ax=ax[0]).set_title('Type of Employement')
survey18.groupby('Student').count()[['Respondent']]
survey18.groupby('Employment').count()[['Respondent']]
sal1 = survey18[survey18['Employment']=='Employed full-time']

sal1 = sal1.dropna(subset=['ConvertedSalary'])

fig,ax = plt.subplots(1,2, figsize=(20,10))



a = sns.distplot(sal1.ConvertedSalary, kde=False, ax=ax[0],axlabel = 'Annual Salary in USD')

a.set_ylabel('Frequency Count')

a.set_title('Annual Salary in USD - Distribution')

b = sns.distplot(sal1.ConvertedSalary,kde=False, ax=ax[1], axlabel = 'Log of Annual Salary in USD')

b.set_xscale('log')

b.set_ylabel('Frequency Count')

b.set_title('with Log')
countries_salary = survey18[survey18['Employment']=='Employed full-time']

countries_salary['count'] = countries_salary.groupby('Country')['Respondent'].transform('count')

countries_salary = countries_salary[countries_salary['count']>500]
fig, ax =  plt.subplots(figsize=(25,18))

ax =sns.violinplot(x=countries_salary.ConvertedSalary, y=countries_salary.Country, data = countries_salary, scale="width", palette="Set2")

ax.set_ylabel('Countries')

ax.set_title('Annual Salary in USD - Distribution by Country - More than 500 respondents')
countries_salary1 = survey18[survey18['Employment']=='Employed full-time']

countries_salary1 = countries_salary1[(countries_salary1['Gender'] == 'Female') | (countries_salary1['Gender']=='Male')]

countries_salary1['count'] = countries_salary1.groupby('Country')['Respondent'].transform('count')

countries_salary1 = countries_salary1[countries_salary1['count']>500]
fig, ax =  plt.subplots(figsize=(25,15))

ax = sns.boxplot(x="ConvertedSalary", y="Country", hue="Gender", data=countries_salary1, palette="Set3")

ax.set_ylabel('Countries')

ax.set_title('Annual Salary in USD - Male vs Female - by Country')
countries_salary = survey18[survey18['Employment']=='Employed full-time']

countries_salary['count'] = countries_salary.groupby('Country')['Respondent'].transform('count')

countries_salary = countries_salary[(countries_salary['count'] > 200) & (countries_salary['count'] < 500)]



fig, ax =  plt.subplots(figsize=(25,18))

ax = sns.violinplot(x=countries_salary.ConvertedSalary, y=countries_salary.Country, data = countries_salary, scale="width", palette="Set2")

ax.set_ylabel('Countries')

ax.set_title('Annual Salary in USD - Distribution by Country - b/w 200 & 500 respondents')
countries_salary1 = survey18[survey18['Employment']=='Employed full-time']

countries_salary1 = countries_salary1[(countries_salary1['Gender'] == 'Female') | (countries_salary1['Gender']=='Male')]

countries_salary1['count'] = countries_salary1.groupby('Country')['Respondent'].transform('count')

countries_salary1 = countries_salary1[(countries_salary1['count'] > 100) & (countries_salary1['count'] < 500)]



fig, ax =  plt.subplots(figsize=(25,15))

ax = sns.boxplot(x="ConvertedSalary", y="Country", hue="Gender", data=countries_salary1, palette="Set3", ax = ax)

ax.set_ylabel('Countries')

ax.set_title('Annual Salary in USD - Male vs Female - by Country')
salary = survey18[survey18['Employment']=='Employed full-time']

salary['count'] = salary.groupby('Country')['Respondent'].transform('count')

salary = salary[salary['count']>500]

salary['med'] = salary.groupby('Country')['ConvertedSalary'].transform('median')

salary['med_med'] = salary['med'].median()

salary['diff'] = salary['med']-salary['med_med']
fig,ax = plt.subplots(figsize=(20,10))

ax =sns.barplot(x="diff", y="Country", data=salary.sort_values(by=['diff'], ascending=False))

ax.set_xlabel('Median Salary difference with Median of Median Salaries of Top Countries')

ax.set_ylabel('Countries')

ax.set_title('Salary Difference')
salary = survey18[survey18['Employment']=='Employed full-time']

salary['count'] = salary.groupby('Country')['Respondent'].transform('count')

salary = salary[(salary['count'] > 100) & (salary['count'] < 500)]

salary['med'] = salary.groupby('Country')['ConvertedSalary'].transform('median')

salary['med_med'] = salary['med'].median()

salary['diff'] = salary['med']-salary['med_med']



fig,ax = plt.subplots(figsize=(20,10))

ax =sns.barplot(x="diff", y="Country", data=salary.sort_values(by=['diff'], ascending=False))

ax.set_xlabel('Median Salary difference with Median of Median Salaries of 2nd bracket Countries')

ax.set_ylabel('Countries')

ax.set_title('Salary Difference')
devtype = survey18[survey18['Employment']=='Employed full-time']

devtype = devtype.filter(items=['DevType', 'ConvertedSalary', 'Gender'])

devtype = devtype.dropna(subset=['DevType'])

d = devtype['DevType'].str.split(';').apply(pd.Series, 1).stack()

d.index = d.index.droplevel(-1) # to line up with df's index

d.name = 'NewDevType'

devtype = devtype.join(d)

devtype['count'] = devtype.groupby('NewDevType')['NewDevType'].transform('count')

devtype = devtype.filter(items=['NewDevType', 'ConvertedSalary', 'Gender',  'count'])

devtype = devtype.drop_duplicates().dropna()

devtype['med'] = devtype.groupby('NewDevType')['ConvertedSalary'].transform('median')

devtype = devtype.sort_values(by='med', ascending=False)
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 20))

ax = sns.barplot(x=devtype.NewDevType, y=devtype.ConvertedSalary, data=devtype, errwidth=0)

plt.xticks(rotation=45)

plt.ylabel('Median Salary')

plt.xlabel('Dev Type')

plt.title('Median Salary by Developer Type')

plt.tight_layout() 
devtype = survey18[survey18['Employment']=='Employed full-time']

devtype = devtype.filter(items=['DevType', 'ConvertedSalary', 'Gender'])

devtype = devtype.dropna(subset=['DevType'])

devtype = devtype[(devtype['Gender'] == 'Female') | (devtype['Gender']=='Male')]





d = devtype['DevType'].str.split(';').apply(pd.Series, 1).stack()

d.index = d.index.droplevel(-1) # to line up with df's index

d.name = 'NewDevType'

devtype = devtype.join(d)



devtype['count'] = devtype.groupby('NewDevType')['NewDevType'].transform('count')

devtype = devtype.filter(items=['NewDevType', 'ConvertedSalary', 'Gender',  'count'])

devtype = devtype.drop_duplicates().dropna()

devtype['med'] = devtype.groupby('NewDevType')['ConvertedSalary'].transform('median')

devtype = devtype.sort_values(by='med', ascending=False)
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.barplot(x="NewDevType", y="ConvertedSalary", hue="Gender",errwidth=0.0, data=devtype, palette="Set2")

plt.xticks(rotation=45)

plt.ylabel('Median Salary')

plt.xlabel('Dev Type')

plt.title('Median Salary by Developer Type wrt Gender')

plt.tight_layout() 
devtype = survey18[survey18['Employment']=='Employed full-time']

devtype = devtype.filter(items=['DevType', 'Gender'])

devtype = devtype.dropna(subset=['DevType'])

devtype = devtype[(devtype['Gender'] == 'Female') | (devtype['Gender']=='Male')]



d = devtype['DevType'].str.split(';').apply(pd.Series, 1).stack()

d.index = d.index.droplevel(-1) 

d.name = 'NewDevType'

devtype = devtype.join(d)
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))



ax = sns.countplot(x="NewDevType", hue="Gender", data=devtype,palette="Set3" )

plt.xticks(rotation=45)

plt.yscale("log")

plt.ylabel('Count')

plt.xlabel('Dev Type')

plt.title('Developer Type Count by Gender')

plt.tight_layout() 
coun = survey18.filter(items=['StackOverflowConsiderMember', 'Country'])

coun = coun.dropna(subset=['StackOverflowConsiderMember', 'Country'])

coun = coun[(coun['StackOverflowConsiderMember'] == 'Yes') | (coun['StackOverflowConsiderMember']=='No')]

coun['count'] = coun.groupby('Country')['StackOverflowConsiderMember'].transform('count')

coun = coun[coun['count']>150]

coun = coun.drop(['count'], axis=1)



coun = coun.pivot_table(index='Country', columns='StackOverflowConsiderMember', aggfunc='size', fill_value=0).reset_index()

coun['No_to_Yes'] = coun['No']/coun['Yes']

coun['No_to_Yes'] = round(coun['No_to_Yes']*100)

coun = coun.sort_values(by=['No_to_Yes'], ascending=False)

coun = coun.head(20)

coun = coun.assign(code = ['FIN','NOR','NZL','CAN','GBR','SWE','CZE','DEU','USA','AUS', 'DNK','HUN','PRT','CHE','POL','IRL','NLD','BEL','JPN','ARG'])
fig = go.Figure(data=go.Choropleth(

    locations = coun['code'],

    z = coun['No_to_Yes'],

    text = coun['Country'],

    colorscale = 'Blues',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.8,

    colorbar_title = '',

))

fig.update_layout(

    title_text='Countries - Ratio of No to Yes (feeling part of SO)',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    )

)



fig.show()
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))



ax = sns.countplot(x="StackOverflowJobs", data=survey18.dropna(subset=['StackOverflowJobs']) ,palette="Set3" )

plt.title('Stack Overflow Jobs Awareness')
nps = survey18[['StackOverflowJobsRecommend']]

nps = nps.replace(to_replace=r'10 (Very Likely)', value='10', regex=False)

nps = nps.replace(to_replace=r'0 (Not Likely)', value='0', regex=False)



nps = nps.dropna(subset=['StackOverflowJobsRecommend']).sort_values(by =['StackOverflowJobsRecommend'], ascending=True)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))



ax = sns.countplot(x="StackOverflowJobsRecommend", data=nps ,palette="Set3" )

plt.title('Net Promoter Score - Recommend SO Jobs?\nHow likely is it that you would recommend SO Jobs to a friend or colleague? ')

plt.ylabel('Count of Respondents')

plt.xlabel('')
nps2 = survey18[['StackOverflowJobsRecommend']]

nps2 = nps2.replace(to_replace=r'10 (Very Likely)', value='10', regex=False)

nps2 = nps2.replace(to_replace=r'0 (Not Likely)', value='0', regex=False)

nps2 = nps2.dropna(subset=['StackOverflowJobsRecommend']).sort_values(by =['StackOverflowJobsRecommend'], ascending=True)

nps2['StackOverflowJobsRecommend'] = nps2['StackOverflowJobsRecommend'].astype(int)

nps2.loc[nps2.StackOverflowJobsRecommend <7, 'StackOverflowJobsRecommend_1'] = 'Detracter'

nps2.loc[nps2.StackOverflowJobsRecommend >7, 'StackOverflowJobsRecommend_1'] = 'Passive'

nps2.loc[nps2.StackOverflowJobsRecommend >=9, 'StackOverflowJobsRecommend_1'] = 'Promoter'

nps2['sum'] = nps2.groupby('StackOverflowJobsRecommend_1').transform('sum')

nps2['perc']= nps2['sum']/nps2['sum'].sum()*1000000

nps2.tail()



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.barplot(x="StackOverflowJobsRecommend_1", y="perc", errwidth=0.0, data=nps2, palette="Set2")

plt.title('Net Promoter Score - Stack Overflow Jobs')

plt.ylabel('Percentage')

plt.xlabel('NPS')
adbloc = survey18[['AdBlocker']]



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))



ax = sns.countplot(x="AdBlocker", data=adbloc.dropna(subset=['AdBlocker']) ,palette="Set3" )

plt.title('Adblocker Usage')
adbloc_mf = survey18[['AdBlocker', 'Gender']]

adbloc_mf = adbloc_mf.dropna(subset=['AdBlocker', 'Gender'])

adbloc_mf = adbloc_mf[(adbloc_mf['Gender'] == 'Female') | (adbloc_mf['Gender']=='Male')]





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))



ax = sns.countplot(x="AdBlocker", hue='Gender', data=adbloc_mf ,palette="Set2" )

plt.title('Adblocker Usage')

adbloc_reason = survey18[['AdBlockerReasons']]



a = adbloc_reason['AdBlockerReasons'].str.split(';').apply(pd.Series, 1).stack()

a.index = a.index.droplevel(-1) 

a.name = 'NewReason'

adbloc_reason = adbloc_reason.join(a)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))



ax = sns.countplot(y="NewReason", data=adbloc_reason ,palette="Set2" )

#plt.xticks(rotation=45)

plt.ylabel('Reasons')

plt.title('Adblocker Unblocking Top Reasons')

plt.tight_layout() 
ad_agree = survey18[['AdsAgreeDisagree1']]

ad_agree = ad_agree.dropna(subset=['AdsAgreeDisagree1'])

ad_agree['count'] = ad_agree.groupby('AdsAgreeDisagree1')['AdsAgreeDisagree1'].transform('count')

ad_agree['perc'] = ad_agree['count']/ad_agree['count'].sum()*100



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.barplot(y="AdsAgreeDisagree1", x="perc", errwidth=0.0, data=ad_agree, palette="Set2")

plt.title('Online advertising can be valuable when it is relevant to me - Do you agree?')

plt.ylabel('Response')

plt.xlabel('Percentage')
ad_agree = survey18[['AdsAgreeDisagree2']]

ad_agree = ad_agree.dropna(subset=['AdsAgreeDisagree2'])

ad_agree['count'] = ad_agree.groupby('AdsAgreeDisagree2')['AdsAgreeDisagree2'].transform('count')

ad_agree['perc'] = ad_agree['count']/ad_agree['count'].sum()*100



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.barplot(y="AdsAgreeDisagree2", x="perc", errwidth=0.0, data=ad_agree, palette="Set2")

plt.title(' I enjoy seeing online updates from companies that I like - Do you agree?')

plt.ylabel('Response')

plt.xlabel('Percentage')
ad_agree = survey18[['AdsAgreeDisagree3']]

ad_agree = ad_agree.dropna(subset=['AdsAgreeDisagree3'])

ad_agree['count'] = ad_agree.groupby('AdsAgreeDisagree3')['AdsAgreeDisagree3'].transform('count')

ad_agree['perc'] = ad_agree['count']/ad_agree['count'].sum()*100



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.barplot(y="AdsAgreeDisagree3", x="perc", errwidth=0.0, data=ad_agree, palette="Set2")

plt.title(' I fundamentally dislike the concept of advertising - Do you agree?')

plt.ylabel('Response')

plt.xlabel('Percentage')
ide = survey18[['IDE']]



i = ide['IDE'].str.split(';').apply(pd.Series, 1).stack()

i.index = i.index.droplevel(-1) 

i.name = 'NewIDE'

ide = ide.join(i)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewIDE", data=ide, order = ide['NewIDE'].value_counts().index ,palette="Set2" )

plt.ylabel('IDE')

plt.title('IDE Preference')

plt.tight_layout()
ide_gen = survey18.filter(items=['IDE', 'Gender'])

ide_gen = ide_gen.dropna(subset=['IDE'])

ide_gen = ide_gen[(ide_gen['Gender'] == 'Female') | (ide_gen['Gender']=='Male')]





i = ide_gen['IDE'].str.split(';').apply(pd.Series, 1).stack()

i.index = i.index.droplevel(-1) 

i.name = 'IDE_NEW'

ide_gen = ide_gen.join(i)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="IDE_NEW", hue="Gender", data=ide_gen, order = ide_gen['IDE_NEW'].value_counts().index ,palette="Set3" )

plt.xscale("log")

plt.ylabel('IDE Preferences')

plt.title('IDE Preferences by Gender')

plt.tight_layout() 
os = survey18[['OperatingSystem']]

os = os.dropna(subset=['OperatingSystem'])



o = os['OperatingSystem'].str.split(';').apply(pd.Series, 1).stack()

o.index = o.index.droplevel(-1) 

o.name = 'NewOs'

os = os.join(o)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewOs", data=os, order = os['NewOs'].value_counts().index ,palette="Set2" )

plt.ylabel('OS')

plt.xlabel('Count')

plt.title('Operating System Preference')

plt.tight_layout()
os_gen = survey18.filter(items=['OperatingSystem', 'Gender'])

os_gen = os_gen.dropna(subset=['OperatingSystem'])

os_gen = os_gen[(os_gen['Gender'] == 'Female') | (os_gen['Gender']=='Male')]



o = os_gen['OperatingSystem'].str.split(';').apply(pd.Series, 1).stack()

o.index = o.index.droplevel(-1) 

o.name = 'NewOs'

os_gen = os_gen.join(o)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewOs", hue="Gender", data=os_gen, order = os_gen['NewOs'].value_counts().index ,palette="Set3" )

plt.xscale("log")

plt.ylabel('OS')

plt.xlabel('Count')

plt.title('Operating System Preferences by Gender')

plt.tight_layout()
lang = survey18[['LanguageWorkedWith']]

lang = lang.dropna(subset=['LanguageWorkedWith'])



l= lang['LanguageWorkedWith'].str.split(';').apply(pd.Series, 1).stack()

l.index = l.index.droplevel(-1) 

l.name = 'NewLang'

lang = lang.join(l)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewLang", data=lang, order = lang['NewLang'].value_counts().index ,palette="Set2" )

plt.ylabel('Languages')

plt.xlabel('Count')

plt.title('LanguageWorkedWith Preference')

plt.tight_layout()
lang_gen = survey18.filter(items=['LanguageWorkedWith', 'Gender'])

lang_gen = lang_gen.dropna(subset=['LanguageWorkedWith'])

lang_gen = lang_gen[(lang_gen['Gender'] == 'Female') | (lang_gen['Gender']=='Male')]



l= lang['LanguageWorkedWith'].str.split(';').apply(pd.Series, 1).stack()

l.index = l.index.droplevel(-1) 

l.name = 'NewLang'

lang_gen = lang_gen.join(l)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewLang", hue="Gender", data=lang_gen, order = lang_gen['NewLang'].value_counts().index ,palette="Set2" )

plt.xscale("log")

plt.ylabel('Languages')

plt.xlabel('Count')

plt.title('Languages Worked With - by Gender')

plt.tight_layout()
deslang = survey18[['LanguageDesireNextYear']]

deslang = deslang.dropna(subset=['LanguageDesireNextYear'])



l= deslang['LanguageDesireNextYear'].str.split(';').apply(pd.Series, 1).stack()

l.index = l.index.droplevel(-1) 

l.name = 'NewDesLang'

deslang = deslang.join(l)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewDesLang", data=deslang, order = deslang['NewDesLang'].value_counts().index ,palette="Set2" )

plt.ylabel('Languages')

plt.xlabel('Count')

plt.title('LanguageDesireNextYear Preference')

plt.tight_layout()
deslang_gen = survey18.filter(items=['LanguageDesireNextYear', 'Gender'])

deslang_gen = deslang_gen.dropna(subset=['LanguageDesireNextYear'])

deslang_gen = deslang_gen[(deslang_gen['Gender'] == 'Female') | (deslang_gen['Gender']=='Male')]



l= deslang_gen['LanguageDesireNextYear'].str.split(';').apply(pd.Series, 1).stack()

l.index = l.index.droplevel(-1) 

l.name = 'NewDesLang'

deslang_gen = deslang_gen.join(l)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewDesLang", hue="Gender", data=deslang_gen, order = deslang_gen['NewDesLang'].value_counts().index ,palette="Set2" )

plt.xscale("log")

plt.ylabel('Languages')

plt.xlabel('Count')

plt.title('LanguageDesireNextYear - by Gender')

plt.tight_layout()
framework = survey18[['FrameworkWorkedWith']]

framework = framework.dropna(subset=['FrameworkWorkedWith'])



f= framework['FrameworkWorkedWith'].str.split(';').apply(pd.Series, 1).stack()

f.index = f.index.droplevel(-1) 

f.name = 'NewFrameworkWorkedWith'

framework = framework.join(f)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewFrameworkWorkedWith", data=framework, order = framework['NewFrameworkWorkedWith'].value_counts().index ,palette="Set2" )

plt.ylabel('Frameworks')

plt.xlabel('Count')

plt.title('FrameworkWorkedWith Preferences')

plt.tight_layout()
framework_gen = survey18.filter(items=['FrameworkWorkedWith', 'Gender'])

framework_gen = framework_gen.dropna(subset=['FrameworkWorkedWith'])

framework_gen = framework_gen[(framework_gen['Gender'] == 'Female') | (framework_gen['Gender']=='Male')]



f= framework_gen['FrameworkWorkedWith'].str.split(';').apply(pd.Series, 1).stack()

f.index = f.index.droplevel(-1) 

f.name = 'NewFramework'

framework_gen = framework_gen.join(f)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewFramework", hue="Gender", data=framework_gen, order = framework_gen['NewFramework'].value_counts().index ,palette="Set2" )

plt.xscale("log")

plt.ylabel('Framework')

plt.xlabel('Count')

plt.title('FrameworkWorkedWith - by Gender')

plt.tight_layout()
nframework = survey18[['FrameworkDesireNextYear']]

nframework = nframework.dropna(subset=['FrameworkDesireNextYear'])



f= nframework['FrameworkDesireNextYear'].str.split(';').apply(pd.Series, 1).stack()

f.index = f.index.droplevel(-1) 

f.name = 'NewFrameworkDesireNextYear'

nframework = nframework.join(f)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewFrameworkDesireNextYear", data=nframework, order = nframework['NewFrameworkDesireNextYear'].value_counts().index ,palette="Set2" )

plt.ylabel('Frameworks')

plt.xlabel('Count')

plt.title('FrameworkDesireNextYear Preferences')

plt.tight_layout()
nframework_gen = survey18.filter(items=['FrameworkDesireNextYear', 'Gender'])

nframework_gen = nframework_gen.dropna(subset=['FrameworkDesireNextYear'])

nframework_gen = nframework_gen[(nframework_gen['Gender'] == 'Female') | (nframework_gen['Gender']=='Male')]



f= nframework_gen['FrameworkDesireNextYear'].str.split(';').apply(pd.Series, 1).stack()

f.index = f.index.droplevel(-1) 

f.name = 'NewFramework'

nframework_gen = nframework_gen.join(f)





sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewFramework", hue="Gender", data=nframework_gen, order = nframework_gen['NewFramework'].value_counts().index ,palette="Set2" )

plt.xscale("log")

plt.ylabel('Framework')

plt.xlabel('Count')

plt.title('FrameworkDesireNextYear - by Gender')

plt.tight_layout()
version = survey18[['VersionControl']]

version = version.dropna(subset=['VersionControl'])



v= version['VersionControl'].str.split(';').apply(pd.Series, 1).stack()

v.index = v.index.droplevel(-1) 

v.name = 'NewVersionControl'

version = version.join(v)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewVersionControl", data=version, order = version['NewVersionControl'].value_counts().index ,palette="Set2" )

plt.ylabel('VersionControl')

plt.xlabel('Count')

plt.title('VersionControl Preferences')

plt.tight_layout()
version_gen = survey18.filter(items=['VersionControl', 'Gender'])

version_gen = version_gen.dropna(subset=['VersionControl'])

version_gen = version_gen[(version_gen['Gender'] == 'Female') | (version_gen['Gender']=='Male')]



v= version_gen['VersionControl'].str.split(';').apply(pd.Series, 1).stack()

v.index = v.index.droplevel(-1) 

v.name = 'NewVersionControl'

version_gen = version_gen.join(v)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(y="NewVersionControl", hue='Gender', data=version_gen, order = version_gen['NewVersionControl'].value_counts().index ,palette="Set2" )

plt.ylabel('VersionControl')

plt.xlabel('Count')

plt.title('VersionControl Preferences')

plt.tight_layout()
kinship = survey18[['AgreeDisagree1']]

kinship = kinship.replace(to_replace=r'1_Strongly disagree', value='Strongly disagree', regex=False)

kinship = kinship.replace(to_replace=r'2_Disagree', value='Disagree', regex=False)

kinship = kinship.replace(to_replace=r'3_Neither Agree nor Disagree', value='Neither Agree nor Disagree', regex=False)

kinship = kinship.replace(to_replace=r'4_Agree', value='Agree', regex=False)

kinship = kinship.replace(to_replace=r'5_Strongly Agree', value='Strongly Agree', regex=False)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="AgreeDisagree1", data=kinship, order = kinship['AgreeDisagree1'].value_counts().index ,palette="Set2" )

plt.xlabel('Kinship towards Fellow Developers')

plt.title('Kinship towards Fellow Developers')

plt.tight_layout()
kinship_gen = survey18.filter(items=['AgreeDisagree1', 'Gender'])

kinship_gen = kinship_gen.dropna(subset=['AgreeDisagree1'])

kinship_gen = kinship_gen[(kinship_gen['Gender'] == 'Female') | (kinship_gen['Gender']=='Male')]



kinship_gen = kinship_gen.replace(to_replace=r'1_Strongly disagree', value='Strongly disagree', regex=False)

kinship_gen = kinship_gen.replace(to_replace=r'2_Disagree', value='Disagree', regex=False)

kinship_gen = kinship_gen.replace(to_replace=r'3_Neither Agree nor Disagree', value='Neither Agree nor Disagree', regex=False)

kinship_gen = kinship_gen.replace(to_replace=r'4_Agree', value='Agree', regex=False)

kinship_gen = kinship_gen.replace(to_replace=r'5_Strongly Agree', value='Strongly Agree', regex=False)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="AgreeDisagree1", data=kinship_gen, hue='Gender',order = kinship_gen['AgreeDisagree1'].value_counts().index ,palette="Set2" )

plt.xlabel('Kinship towards Fellow Developers - Male vs Female')

plt.yscale('log')

plt.title('Kinship towards Fellow Developers - Male vs Female')

plt.tight_layout()
compete = survey18[['AgreeDisagree2']]

compete = compete.replace(to_replace=r'1_Strongly disagree', value='Strongly disagree', regex=False)

compete = compete.replace(to_replace=r'2_Disagree', value='Disagree', regex=False)

compete = compete.replace(to_replace=r'3_Neither Agree nor Disagree', value='Neither Agree nor Disagree', regex=False)

compete = compete.replace(to_replace=r'4_Agree', value='Agree', regex=False)

compete = compete.replace(to_replace=r'5_Strongly Agree', value='Strongly Agree', regex=False)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="AgreeDisagree2", data=compete, order = compete['AgreeDisagree2'].value_counts().index ,palette="Set2" )

plt.xlabel('Competing Feeling towards Fellow Developers')

plt.title('Competing Feeling towards Fellow Developers')

plt.tight_layout()
compete_gen = survey18.filter(items=['AgreeDisagree2', 'Gender'])

compete_gen = compete_gen.dropna(subset=['AgreeDisagree2'])

compete_gen = compete_gen[(compete_gen['Gender'] == 'Female') | (compete_gen['Gender']=='Male')]



compete_gen = compete_gen.replace(to_replace=r'1_Strongly disagree', value='Strongly disagree', regex=False)

compete_gen = compete_gen.replace(to_replace=r'2_Disagree', value='Disagree', regex=False)

compete_gen = compete_gen.replace(to_replace=r'3_Neither Agree nor Disagree', value='Neither Agree nor Disagree', regex=False)

compete_gen = compete_gen.replace(to_replace=r'4_Agree', value='Agree', regex=False)

compete_gen = compete_gen.replace(to_replace=r'5_Strongly Agree', value='Strongly Agree', regex=False)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="AgreeDisagree2", data=compete_gen, hue='Gender',order = compete_gen['AgreeDisagree2'].value_counts().index ,palette="Set2" )

plt.xlabel('Competing Feeling towards Fellow Developers')

plt.yscale('log')

plt.title('Competing Feeling towards Fellow Developers - Male vs Female')

plt.tight_layout()
imposter = survey18[['AgreeDisagree3']]

imposter = imposter.replace(to_replace=r'1_Strongly disagree', value='Strongly disagree', regex=False)

imposter = imposter.replace(to_replace=r'2_Disagree', value='Disagree', regex=False)

imposter = imposter.replace(to_replace=r'3_Neither Agree nor Disagree', value='Neither Agree nor Disagree', regex=False)

imposter = imposter.replace(to_replace=r'4_Agree', value='Agree', regex=False)

imposter = imposter.replace(to_replace=r'5_Strongly Agree', value='Strongly Agree', regex=False)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="AgreeDisagree3", data=imposter, order = imposter['AgreeDisagree3'].value_counts().index ,palette="Set2" )

plt.xlabel('Feeling not good at programming as Peers')

plt.title('Feeling not good at programming as Peers')

plt.tight_layout()
imposter_gen = survey18.filter(items=['AgreeDisagree3', 'Gender'])

imposter_gen = imposter_gen.dropna(subset=['AgreeDisagree3'])

imposter_gen = imposter_gen[(imposter_gen['Gender'] == 'Female') | (imposter_gen['Gender']=='Male')]



imposter_gen = imposter_gen.replace(to_replace=r'1_Strongly disagree', value='Strongly disagree', regex=False)

imposter_gen = imposter_gen.replace(to_replace=r'2_Disagree', value='Disagree', regex=False)

imposter_gen = imposter_gen.replace(to_replace=r'3_Neither Agree nor Disagree', value='Neither Agree nor Disagree', regex=False)

imposter_gen = imposter_gen.replace(to_replace=r'4_Agree', value='Agree', regex=False)

imposter_gen = imposter_gen.replace(to_replace=r'5_Strongly Agree', value='Strongly Agree', regex=False)



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="AgreeDisagree3", data=imposter_gen, hue='Gender',order = imposter_gen['AgreeDisagree3'].value_counts().index ,palette="Set2" )

plt.xlabel('Feeling not good at programming as Peers')

plt.yscale('log')

plt.title('Feeling not good at programming as Peers - by Gender')

plt.tight_layout()
peer = survey18[['HypotheticalTools1']]

peer = peer.dropna(subset=['HypotheticalTools1'])



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools1", data=peer, order = peer['HypotheticalTools1'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.xlabel('Peer Mentoring System')

plt.title('A Peer Mentoring System - Overall')

plt.tight_layout()
peer_gen = survey18.filter(items=['HypotheticalTools1', 'Gender'])

peer_gen = peer_gen.dropna(subset=['HypotheticalTools1'])

peer_gen = peer_gen[(peer_gen['Gender'] == 'Female') | (peer_gen['Gender']=='Male')]



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools1", hue = 'Gender', data=peer_gen, order = peer_gen['HypotheticalTools1'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.yscale('log')

plt.xlabel('Peer Mentoring System')

plt.title('A Peer Mentoring System - by Gender')

plt.tight_layout()
pvt = survey18[['HypotheticalTools2']]

pvt = pvt.dropna(subset=['HypotheticalTools2'])



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools2", data=pvt, order = pvt['HypotheticalTools2'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.xlabel('Responses')

plt.title('A private area for people new to programming')

plt.tight_layout()
pvt_gen = survey18.filter(items=['HypotheticalTools2', 'Gender'])

pvt_gen = pvt_gen.dropna(subset=['HypotheticalTools2'])

pvt_gen = pvt_gen[(pvt_gen['Gender'] == 'Female') | (pvt_gen['Gender']=='Male')]



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools2", hue = 'Gender', data=pvt_gen, order = pvt_gen['HypotheticalTools2'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.yscale('log')

plt.xlabel('Responses')

plt.title('A private area for people new to programming - by Gender')

plt.tight_layout()
prog = survey18[['HypotheticalTools3']]

prog = prog.dropna(subset=['HypotheticalTools3'])



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools3", data=prog, order = prog['HypotheticalTools3'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.xlabel('Responses')

plt.title('A programming-oriented blog platform')

plt.tight_layout()
prog_gen = survey18.filter(items=['HypotheticalTools3', 'Gender'])

prog_gen = prog_gen.dropna(subset=['HypotheticalTools3'])

prog_gen = prog_gen[(prog_gen['Gender'] == 'Female') | (prog_gen['Gender']=='Male')]



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools3", hue = 'Gender', data=prog_gen, order = prog_gen['HypotheticalTools3'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.yscale('log')

plt.xlabel('Responses')

plt.title('A programming-oriented blog platform - Male Vs Female')

plt.tight_layout()
emp = survey18[['HypotheticalTools4']]

emp = emp.dropna(subset=['HypotheticalTools4'])



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools4", data=emp, order = emp['HypotheticalTools4'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.xlabel('Responses')

plt.title('An employer or job review system')

plt.tight_layout()
emp_gen = survey18.filter(items=['HypotheticalTools4', 'Gender'])

emp_gen = emp_gen.dropna(subset=['HypotheticalTools4'])

emp_gen = emp_gen[(emp_gen['Gender'] == 'Female') | (emp_gen['Gender']=='Male')]



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools4", hue = 'Gender', data=emp_gen, order = emp_gen['HypotheticalTools4'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.yscale('log')

plt.xlabel('Responses')

plt.title('An employer or job review system - Male Vs Female')

plt.tight_layout()
qanda = survey18[['HypotheticalTools5']]

qanda = qanda.dropna(subset=['HypotheticalTools5'])



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools5", data=qanda, order = qanda['HypotheticalTools5'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.xlabel('Responses')

plt.title('An area for Q&A related to career growth - Overall')

plt.tight_layout()
qanda_gen = survey18.filter(items=['HypotheticalTools5', 'Gender'])

qanda_gen = qanda_gen.dropna(subset=['HypotheticalTools5'])

qanda_gen = qanda_gen[(qanda_gen['Gender'] == 'Female') | (qanda_gen['Gender']=='Male')]



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(20, 15))

ax = sns.countplot(x="HypotheticalTools5", hue = 'Gender', data=qanda_gen, order = qanda_gen['HypotheticalTools5'].value_counts().index ,palette="Set2" )

plt.ylabel('Count')

plt.yscale('log')

plt.xlabel('Responses')

plt.title('An area for Q&A related to career growth - Male Vs Female')

plt.tight_layout()