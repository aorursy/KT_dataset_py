#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from matplotlib import cm
cs=cm.Set1(np.arange(40)/40.)
data = pd.read_csv("../input/survey_results_public.csv")
data.head()
data['DevType'].value_counts()[:20].plot(kind='barh', figsize=(8,6))
#For the Full Stack Dev's
a = data.loc[data.DevType == 'Full-stack developer','IDE']
df = pd.DataFrame(a)
df = df.dropna()
b = pd.DataFrame(df.IDE.str.split(';').tolist()).stack()
b.columns = ['COUNT', 'IDE']
b.value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Popular IDE's amongst Full Stack developers")
#For back-end dev's
a1 = data.loc[data.DevType == 'Back-end developer','IDE']
df1 = pd.DataFrame(a1)
df1 = df1.dropna()
b1 = pd.DataFrame(df1.IDE.str.split(';').tolist()).stack()
b1.columns = ['COUNT', 'IDE']
b1.value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Popular IDE's amongst back end developers")
#For Mobile dev's
a2 = data.loc[data.DevType == 'Mobile developer','IDE']
df2 = pd.DataFrame(a2)
df2 = df2.dropna()
b2 = pd.DataFrame(df2.IDE.str.split(';').tolist()).stack()
b2.columns = ['COUNT', 'IDE']
b2.value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Popular IDE's amongst Mobile Developers")
#For the ML folks
a3 = data.loc[data.DevType == 'Data scientist or machine learning specialist','IDE']
df3 = pd.DataFrame(a3)
df3 = df3.dropna()
b3 = pd.DataFrame(df3.IDE.str.split(';').tolist()).stack()
b3.columns = ['COUNT', 'IDE']
b3.value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Popular IDE's amongst ML specialists and Data Scientists")
data['Hobby'].value_counts().plot(kind='pie', figsize=(8,8), title = '% of developers who code as a hobby', autopct='%1.0f%%', explode=(0, 0.1),
                                 colors = ['#7FFF00', '#FF7256'], shadow=True)
x = data.loc[data.Hobby == 'Yes','Country']
df_x = pd.DataFrame(x)
df_x = df_x.dropna()
df_x['Country'].value_counts()[:20].plot(kind='bar', figsize=(8,6), title = '% of developers in the country who code as a hobby')
us = data.loc[data.Country == 'United States','Hobby']
df_us = pd.DataFrame(us)
df_us = df_us.dropna()
df_us['Hobby'].value_counts().plot(kind='pie', figsize=(8,8), title = '% of developers in the US who code as a hobby', autopct='%1.0f%%', explode=(0.1, 0),
                                  shadow=True)
indh = data.loc[data.Country == 'India','Hobby']
df_indh = pd.DataFrame(indh)
df_indh = df_indh.dropna()
df_indh['Hobby'].value_counts().plot(kind='pie', figsize=(8,8), title = '% of developers in India who code as a hobby', autopct='%1.0f%%', explode=(0.1, 0),
                                  shadow=True)
y = data.loc[data.OpenSource == 'Yes','Country']
df_y = pd.DataFrame(y)
df_y = df_y.dropna()
df_y['Country'].value_counts()[:20].plot(kind='bar', figsize=(8,6), title = '% of developers in the country who contribute towards open source')
us1 = data.loc[data.Country == 'United States','OpenSource']
df_us1 = pd.DataFrame(us1)
df_us1 = df_us1.dropna()
df_us1['OpenSource'].value_counts().plot(kind='pie', figsize=(8,8), title = '% of developers in the US who contribute to Open Source', autopct='%1.0f%%',
                                         explode=(0.1, 0),
                                  shadow=True)
ind = data.loc[data.Country == 'India','OpenSource']
df_in = pd.DataFrame(ind)
df_in = df_in.dropna()
df_in['OpenSource'].value_counts().plot(kind='pie', figsize=(8,8), title = '% of developers in India who contribute to Open Source', autopct='%1.0f%%',
                                         explode=(0.1, 0), colors = ['#7FFF00', '#FF7256'],
                                  shadow=True)
data['Employment'].value_counts().plot(kind='pie', figsize=(9,9), title = 'Employment status of developers', autopct='%1.0f%%',
                                       shadow=True,explode=(0.1, 0, 0, 0, 0, 0.1))
data['JobSatisfaction'].value_counts().plot(kind='pie', figsize=(9,9), title = 'Job Satisfaction of developers', autopct='%1.0f%%',
                                           explode=(0.1, 0, 0, 0.1, 0, 0, 0), shadow=True)
data['CareerSatisfaction'].value_counts().plot(kind='pie', figsize=(9,9), title = 'Career Satisfaction of developers', autopct='%1.0f%%',
                                              explode=(0.1, 0, 0, 0.1, 0, 0, 0), shadow=True)
p = data.loc[data.JobSatisfaction == 'Extremely dissatisfied','Country']
df_p = pd.DataFrame(p)
df_p = df_p.dropna()
df_p['Country'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title = '% of developers in the country who are extremely dissatisfied with their job')
usj = data.loc[data.Country == 'United States','JobSatisfaction']
df_usj = pd.DataFrame(usj)
df_usj = df_usj.dropna()
df_usj['JobSatisfaction'].value_counts().plot(kind='pie', figsize=(8,8), title = 'Job satisfaction of US devs',
                                                  autopct='%1.0f%%', shadow=True, explode=(0.1, 0.015, 0, 0, 0, 0, 0))
inj = data.loc[data.Country == 'India','JobSatisfaction']
df_inj = pd.DataFrame(inj)
df_inj = df_inj.dropna()
df_inj['JobSatisfaction'].value_counts().plot(kind='pie', figsize=(8,8), title = 'Job satisfaction of Indian devs',
                                                  autopct='%1.0f%%', shadow=True, explode=(0.1, 0.1, 0, 0, 0, 0, 0),
                                                  colors=['yellow', 'cyan', 'green', 'orange', 'magenta', 'pink', 'red'])
q = data.loc[data.CareerSatisfaction == 'Extremely dissatisfied','Country']
df_q = pd.DataFrame(q)
df_q = df_q.dropna()
df_q['Country'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title = '% of developers in the country who are extremely dissatisfied with their career')
usc = data.loc[data.Country == 'United States','CareerSatisfaction']
df_usc = pd.DataFrame(usc)
df_usc = df_usc.dropna()
df_usc['CareerSatisfaction'].value_counts().plot(kind='pie', figsize=(8,8), title = 'Career satisfaction of US devs',
                                                  autopct='%1.0f%%', shadow=True, explode=(0.1, 0.015, 0, 0, 0, 0, 0))
inc = data.loc[data.Country == 'India','CareerSatisfaction']
df_inc = pd.DataFrame(inc)
df_inc = df_inc.dropna()
df_inc['CareerSatisfaction'].value_counts().plot(kind='pie', figsize=(8,8), title = 'Career satisfaction of Indian devs',
                                                  autopct='%1.0f%%', shadow=True, explode=(0.1, 0.015, 0, 0, 0, 0, 0),
                                                colors=['yellow', 'cyan', 'green', 'orange', 'magenta', 'pink', 'red'])
data['OpenSource'].value_counts().plot(kind='pie', figsize=(9,9), title = 'Career Satisfaction of developers', autopct='%1.0f%%',
                                              explode=(0.04, 0), shadow=True, colors=['#FFFF00', '#33A1C9'])
x1 = data.loc[data.DevType == 'Data scientist or machine learning specialist','UndergradMajor']
df_x1 = pd.DataFrame(x1)
df_x1 = df_x1.dropna()
df_x1['UndergradMajor'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Subjects in which ML specialists took Majors")
x2 = data.loc[data.DevType == 'Data scientist or machine learning specialist','FormalEducation']
df_x2 = pd.DataFrame(x2)
df_x2 = df_x2.dropna()
df_x2['FormalEducation'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Formal Education of ML specialists")
data['HopeFiveYears'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Expectation of dev's in upcoming years")
x3 = data.loc[data.HopeFiveYears == "Working in a different or more specialized technical role than the one I'm in now",'CareerSatisfaction']
df_x3 = pd.DataFrame(x3)
df_x3 = df_x3.dropna()
df_x3['CareerSatisfaction'].value_counts()[:20].plot(kind='bar', figsize=(8,6), title="Satisfaction Level of those who want to work in different roles")
x4 = data.loc[data.HopeFiveYears == "Working in a different or more specialized technical role than the one I'm in now",'JobSatisfaction']
df_x4 = pd.DataFrame(x4)
df_x4 = df_x4.dropna()
df_x4['JobSatisfaction'].value_counts()[:20].plot(kind='bar', figsize=(8,6), title="Satisfaction Level of those who want to work in different roles")
data['JobSearchStatus'].value_counts()[:20].plot(kind='pie', figsize=(8,8), title="Job Search Status of Dev's", autopct='%1.0f%%',
                                                colors=['#FFFF00', '#33A1C9', '#98FB98'], explode=(0.1, 0, 0.1))
x5 = data.loc[data.JobSearchStatus == "I am actively looking for a job",'DevType']
df_x5 = pd.DataFrame(x5)
df_x5 = df_x5.dropna()
df_x5['DevType'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Dev description from which most active job searchers emerged")
data['OperatingSystem'].value_counts().plot(kind='pie', figsize=(9,9), title = 'OS most used by developers', autopct='%1.0f%%',
                                              explode=(0.015, 0.015, 0, 0))
x6  = data.loc[data.OperatingSystem == "MacOS",'DevType']
df_x6 = pd.DataFrame(x6)
df_x6 = df_x6.dropna()
df_x6['DevType'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="DevType's who mostly use MacOS")
x7  = data.loc[data.OperatingSystem == "Linux-based",'DevType']
df_x7 = pd.DataFrame(x7)
df_x7 = df_x7.dropna()
df_x7['DevType'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="DevType's who mostly use Linux-based Systems")
data['LanguageWorkedWith'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Some popular Languages amogst Developers")
x8  = data.loc[data.DevType == "Data scientist or machine learning specialist",'LanguageWorkedWith']
df_x8 = pd.DataFrame(x8)
df_x8 = df_x8.dropna()
b3 = pd.DataFrame(df_x8.LanguageWorkedWith.str.split(';').tolist()).stack()
b3.columns = ['COUNT', 'LanguageWorkedWith']
b3.value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Language preferred by the ML folks")
data['FrameworkWorkedWith'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Some popular Frameworks amogst Developers")
x9  = data.loc[data.DevType == "Data scientist or machine learning specialist",'FrameworkWorkedWith']
df_x9 = pd.DataFrame(x9)
df_x9 = df_x9.dropna()
b4 = pd.DataFrame(df_x9.FrameworkWorkedWith.str.split(';').tolist()).stack()
b4.columns = ['COUNT', 'FrameworkWorkedWith']
b4.value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Framework preferred by the ML folks")
data['DatabaseWorkedWith'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Some popular Databases amogst Developers")
data['PlatformWorkedWith'].value_counts()[:20].plot(kind='barh', figsize=(8,6), title="Some popular Platforms amogst Developers")
x10  = data.loc[data.DevType == "Data scientist or machine learning specialist",'PlatformWorkedWith']
df_x10 = pd.DataFrame(x10)
df_x10 = df_x10.dropna()
b5 = pd.DataFrame(df_x10.PlatformWorkedWith.str.split(';').tolist()).stack()
b5.columns = ['COUNT', 'PlatformWorkedWith']
b5.value_counts()[:20].plot(kind='bar', figsize=(8,6), title="Platforms preferred by the ML folks")
data['Gender'].value_counts().plot(kind='barh', figsize=(8,6), title="Gender proportions in the developer survey")
f  = data.loc[data.Gender == "Female",'AgreeDisagree1']
df_f = pd.DataFrame(f)
df_f = df_f.dropna()
df_f['AgreeDisagree1'].value_counts().plot(kind='pie', figsize=(8,8), title="% agreement with AgreeDisagree1 by females", autopct='%1.0f%%')
m  = data.loc[data.Gender == "Male",'AgreeDisagree1']
df_m = pd.DataFrame(m)
df_m = df_m.dropna()
df_m['AgreeDisagree1'].value_counts().plot(kind='pie', figsize=(8,8), title="% agreement with AgreeDisagree1 by males", autopct='%1.0f%%')
f1  = data.loc[data.Gender == "Female",'AgreeDisagree3']
df_f1 = pd.DataFrame(f1)
df_f1 = df_f1.dropna()
df_f1['AgreeDisagree3'].value_counts().plot(kind='pie', figsize=(8,8), title="% agreement with AgreeDisagree3 by females", autopct='%1.0f%%')
m1  = data.loc[data.Gender == "Male",'AgreeDisagree3']
df_m1 = pd.DataFrame(m1)
df_m1 = df_m1.dropna()
df_m1['AgreeDisagree3'].value_counts().plot(kind='pie', figsize=(8,8), title="% agreement with AgreeDisagree3 by males", autopct='%1.0f%%')
f2  = data.loc[data.Gender == "Female",'AgreeDisagree2']
df_f2 = pd.DataFrame(f2)
df_f2 = df_f2.dropna()
df_f2['AgreeDisagree2'].value_counts().plot(kind='pie', figsize=(8,8), title="% agreement with AgreeDisagree2 by females", autopct='%1.0f%%')
m2  = data.loc[data.Gender == "Male",'AgreeDisagree2']
df_m2 = pd.DataFrame(m2)
df_m2 = df_m2.dropna()
df_m2['AgreeDisagree2'].value_counts().plot(kind='pie', figsize=(8,8), title="% agreement with AgreeDisagree2 by males", autopct='%1.0f%%')
et = data['EducationTypes']
df_et = pd.DataFrame(et)
df_et = df_et.dropna()
d_et = pd.DataFrame(df_et.EducationTypes.str.split(';').tolist()).stack()
d_et.columns = ['COUNT', 'EducationTypes']
d_et.value_counts().plot(kind='pie', figsize=(8,8), title="Methods used by devs to teach themselves", autopct='%1.0f%%',
                        explode=(0.1, 0.1, 0, 0, 0, 0, 0, 0, 0), shadow=True)
mind  = data.loc[data.Country == "India",'EducationTypes']
df_mind = pd.DataFrame(mind)
df_mind = df_mind.dropna()
b_mind = pd.DataFrame(df_mind.EducationTypes.str.split(';').tolist()).stack()
b_mind.value_counts().plot(kind='pie', figsize=(8,8), title="Popularity of Moocs in India", autopct='%1.0f%%', explode=(0.1,0.1,0,0,0,0,0,0,0),
                          shadow=True)
mus  = data.loc[data.Country == "United States",'EducationTypes']
df_mus = pd.DataFrame(mus)
df_mus = df_mus.dropna()
b_mus = pd.DataFrame(df_mus.EducationTypes.str.split(';').tolist()).stack()
b_mus.value_counts().plot(kind='pie', figsize=(8,8), title="Popularity of Moocs in US", autopct='%1.0f%%', explode=(0.1,0.1,0,0,0,0,0,0,0),
                          shadow=True)
muk  = data.loc[data.Country == "United Kingdom",'EducationTypes']
df_muk = pd.DataFrame(muk)
df_muk = df_muk.dropna()
b_muk = pd.DataFrame(df_muk.EducationTypes.str.split(';').tolist()).stack()
b_muk.value_counts().plot(kind='pie', figsize=(8,8), title="Popularity of Moocs in UK", autopct='%1.0f%%', explode=(0.1,0.1,0,0,0,0,0,0,0),
                          shadow=True)

data['AIDangerous'].value_counts().plot(kind='pie', figsize=(8,8), title="Aspects of AI which scares devs the most", autopct='%1.0f%%',
                                       explode=(0.1, 0.01, 0, 0), shadow=True, colors=['blue','orange','green', 'cyan'])
data['AIFuture'].value_counts().plot(kind='pie', figsize=(8,8), title="Devs view on the future of AI", autopct='%1.0f%%',
                                       explode=(0.1, 0, 0), shadow=True, colors=['yellow','orange', 'cyan'])
data['AIResponsible'].value_counts().plot(kind='pie', figsize=(8,8), title="Who should be responsible for AI", autopct='%1.0f%%',
                                       explode=(0.1, 0, 0, 0), shadow=True)
data['AIInteresting'].value_counts().plot(kind='pie', figsize=(8,8), title="Interesting aspects of AI", autopct='%1.0f%%',
                                       explode=(0.1, 0, 0, 0), shadow=True)
mx  = data.loc[data.Gender == "Male",'EthicsChoice']
df_mx = pd.DataFrame(mx)
df_mx = df_mx.dropna()
df_mx['EthicsChoice'].value_counts().plot(kind='pie', figsize=(8,8), title="What % of males would write a code they consider unethical", 
                                          explode=(0.1, 0, 0), autopct='%1.0f%%', shadow=True)
my  = data.loc[data.Gender == "Female",'EthicsChoice']
df_my = pd.DataFrame(my)
df_my = df_my.dropna()
df_my['EthicsChoice'].value_counts().plot(kind='pie', figsize=(8,8), title="What % of females would write a code they consider unethical", 
                                          explode=(0.1, 0, 0), autopct='%1.0f%%', shadow=True)
mx1  = data.loc[data.Gender == "Male",'EthicsReport']
df_mx1 = pd.DataFrame(mx1)
df_mx1 = df_mx1.dropna()
df_mx1['EthicsReport'].value_counts().plot(kind='pie', figsize=(8,8), title="What % of males would report unethical code", 
                                           explode=(0.1, 0, 0, 0),autopct='%1.0f%%', shadow=True)
fx1  = data.loc[data.Gender == "Female",'EthicsReport']
df_fx1 = pd.DataFrame(fx1)
df_fx1 = df_fx1.dropna()
df_fx1['EthicsReport'].value_counts().plot(kind='pie', figsize=(8,8), title="What % of females would report unethical code", 
                                           explode=(0.1, 0, 0, 0),autopct='%1.0f%%', shadow=True)
mx2  = data.loc[data.Gender == "Male",'EthicsResponsible']
df_mx2 = pd.DataFrame(mx2)
df_mx2 = df_mx2.dropna()
df_mx2['EthicsResponsible'].value_counts().plot(kind='pie', figsize=(8,8), title="Who is responsible for unethical code as per male devs", 
                                           autopct='%1.0f%%', shadow=True, explode=(0.1,0,0))
fx2  = data.loc[data.Gender == "Female",'EthicsResponsible']
df_fx2 = pd.DataFrame(fx2)
df_fx2 = df_fx2.dropna()
df_fx2['EthicsResponsible'].value_counts().plot(kind='pie', figsize=(8,8), title="Who is responsible for unethical code as per female devs", 
                                           autopct='%1.0f%%', shadow=True, explode=(0.1,0,0))
